---
title: Hft Environment Server
emoji: 🏸
colorFrom: green
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Hft Environment

A limit-order-book execution environment built on OpenEnv. It simulates market microstructure scenarios (including stressed conditions) and lets an agent place limit, market, and cancel actions through a persistent client connection.

## Quick Start

The simplest way to use the Hft environment is through the `HftEnv` class:

```python
from hft import HftAction, HftEnv

try:
    # Start and connect to the Dockerized environment
    hftenv = HftEnv.from_docker_image("hft-env:latest")

    # Reset episode
    result = hftenv.reset()
    print(f"Initial time: {result.observation.time}")

    # Submit a few trading actions
    actions = [
        HftAction(limit_buy_price=99.95, limit_buy_size=100),
        HftAction(limit_ask_price=100.05, limit_ask_size=100),
        HftAction(market_buy_size=50),
    ]

    for action in actions:
        result = hftenv.step(action)
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")
        print(f"Active orders: {result.observation.active_orders}")

finally:
    # Always clean up
    hftenv.close()
```

That's it! The `HftEnv.from_docker_image()` method handles:

- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t hft-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can deploy this OpenEnv environment to Hugging Face Spaces with `openenv push`:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:

1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Build a Hugging Face-compatible Docker Space image
3. Upload and publish the environment with the configured app entrypoint

### Prerequisites

- Authenticate with Hugging Face. If needed, the CLI prompts for login.

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:

- **Web Interface** at `/web` - Interactive controls for environment interaction
- **API Documentation** at `/docs` - OpenAPI docs for HTTP endpoints
- **Health Check** at `/health` - Liveness and readiness checks
- **WebSocket** at `/ws` - Persistent low-latency episode sessions

## Environment Details

### Action

**HftAction** supports order placement and cancellation:

- `limit_buy_price` (float, optional) - Limit price for buy order
- `limit_buy_size` (int, optional) - Size for buy limit order
- `limit_ask_price` (float, optional) - Limit price for ask order
- `limit_ask_size` (int, optional) - Size for ask limit order
- `market_buy_size` (int, optional) - Size for market buy order
- `market_ask_size` (int, optional) - Size for market ask order
- `cancel_order_id` (str, optional) - Existing order id to cancel

### Observation

**HftObservation** contains market and episode feedback:

- `time` (float) - Current simulation timestamp
- `reward` (float) - Reward from the current transition
- `done` (bool) - Episode termination flag
- `history` (list) - Lookback window of book/market state snapshots
- `active_orders` (list) - Agent order ids currently resting in the book

### Reward

Reward is produced by the simulator after each step and is designed to balance execution quality, inventory risk, and completion urgency. Higher rewards generally correspond to better fills and better inventory control under the active scenario.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Hft environment server running, you can connect directly:

```python
from hft import HftAction, HftEnv

# Connect to existing server
hftenv = HftEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = hftenv.reset()
result = hftenv.step(HftAction(market_ask_size=25))
```

Note: When connecting to an existing server, `hftenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from hft import HftAction, HftEnv

# Connect with context manager (auto-connects and closes)
with HftEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Initial step time: {result.observation.time}")
    # Multiple steps with low latency
    for _ in range(3):
        result = env.step(HftAction(limit_buy_price=99.9, limit_buy_size=10))
        print(f"Reward: {result.reward}, done={result.done}")
```

The client uses WebSocket connections for:

- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - tune concurrent session capacity
app = create_app(
    create_HftEnvironment,
    HftAction,
    HftObservation,
    max_concurrent_envs=10,
)
```

Then multiple clients can connect simultaneously:

```python
from hft import HftAction, HftEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with HftEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(HftAction(limit_buy_price=99.9 + i * 0.01, limit_buy_size=5))
        return client_id, result.reward

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test environment logic directly without starting the HTTP server:

```bash
# From project root
python -c "from server.hft_environment import HftEnvironment; env=HftEnvironment(); print(env.reset())"
```

This verifies that:

- Environment resets correctly
- Step accepts valid actions
- State progression works
- Reward and termination signals are emitted

### Running Locally

Run the server locally for development:

```bash
uv run --project . server --port 8000
```

## Project Structure

```
hft/
├── .env                  # Runtime environment overrides
├── .gitignore            # Git ignore rules
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── client.py              # HftEnv client
├── Inference.py           # LLM-driven inference runner
├── models.py              # Action and Observation models
├── validate.py            # Validation helpers / placeholder
├── openenv_hft.egg-info/  # Packaging metadata
└── server/
    ├── __init__.py        # Server module exports
    ├── hft_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    ├── Dockerfile         # Container image definition
    ├── config/            # Scenario configuration YAML files
    └── core/              # Matching engine and trader implementations
```
