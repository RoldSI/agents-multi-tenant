# Custom DockerExecutor for smolagents

This project demonstrates how to use a custom `DockerExecutor` with the `smolagents` library by modifying the library to accept a custom executor class.

## Modifications Made

We modified the `smolagents` library to enhance the `CodeAgent` class to accept a custom executor class:

1. Added a new `executor_class` parameter to `CodeAgent.__init__()` method
2. Updated `create_python_executor()` to use the provided executor class if available
3. Expanded the type hints and documentation

This allows us to pass our own executor implementation instead of relying on the built-in executor types.

## Key Files

- `multi_tenant/DockerExecutor.py` - Our custom Docker-based executor implementation
- `demo/use_docker_executor.py` - Example script showing how to use the custom executor

## How It Works

### 1. DockerExecutor Implementation

Our `DockerExecutor` implements the `PythonExecutor` interface with these key features:

- Creates a Docker container with a Jupyter kernel
- Communicates with the kernel via WebSockets
- Properly handles state management between code executions
- Supports dynamic port allocation for multiple executor instances

### 2. Modified smolagents

The modified library allows passing a custom executor class:

```python
agent = CodeAgent(
    tools=[],
    model=model,
    executor_class=DockerExecutor,  # Pass the class type, not an instance
    executor_kwargs={...},          # Parameters for the executor
    additional_authorized_imports=["numpy", "pandas"],
)
```

### 3. Installation

We installed our modified version locally in development mode:

```bash
# Clone the repository
git clone https://github.com/huggingface/smolagents.git

# Make modifications to src/smolagents/agents.py (adding executor_class)

# Install in development mode
pip install -e .
```

## Usage Instructions

1. Build the Docker image once at application startup:
   ```python
   DockerExecutor.setup()
   ```

2. Create a CodeAgent with the DockerExecutor class:
   ```python
   agent = CodeAgent(
       tools=[],
       model=model,
       executor_class=DockerExecutor,
       executor_kwargs={"host": "127.0.0.1"}
   )
   ```

3. Run tasks with the agent:
   ```python
   result = agent.run("Your task here")
   ```

## Benefits

- **Isolation**: Each execution happens in a separate Docker container
- **Security**: Code runs in an isolated environment, not on the host
- **Customization**: Full control over the execution environment
- **Persistence**: Each executor instance maintains its own container 