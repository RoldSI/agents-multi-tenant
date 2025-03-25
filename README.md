# Multi-Tenant Agent Framework

A wrapper around smolagents that provides a multi-tenant tool and LLM management system.

## Overview

This framework provides a scalable, multi-tenant solution for managing tools and language models for use with smolagents. It uses SQLite for persistent storage of tool metadata and LLM configurations.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/agents-multi-tenant.git
   cd agents-multi-tenant
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

   This will install the required dependencies including:
   - smolagents (>= 0.5.0)
   - openai (>= 1.0.0)
   - python-dotenv (>= 1.0.0)

## Testing

To test that everything is working correctly:

1. Create a `.env` file with your API key:
   ```bash
   cp .env.template .env
   ```
   
   Then edit the `.env` file to add your API key:
   ```
   LLM_API_KEY=your_actual_api_key_here
   ```

   You can also customize the model name and API base URL:
   ```
   MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
   API_BASE_URL=https://fmapi.swissai.cscs.ch
   ```

2. Run the test script:
   ```bash
   python test.py
   ```

   This script will:
   - Create a simple greeting tool
   - Configure an LLM using your API key
   - Run a query that uses the greeting tool
   - Verify that database files and tool files were created correctly

## Usage

### Tool Management

```python
from multi_tenant import add_tool, get_tool, get_tools

# Add a new tool
tool_id = add_tool(
    name="calculator",
    description="A simple calculator tool",
    code="""
def calculate(operation: str, a: float, b: float) -> str:
    '''
    Perform a basic arithmetic operation on two numbers.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: The first number
        b: The second number
    
    Returns:
        The result of the operation as a string
    '''
    if operation == 'add':
        return f"{a} + {b} = {a + b}"
    elif operation == 'subtract':
        return f"{a} - {b} = {a - b}"
    elif operation == 'multiply':
        return f"{a} * {b} = {a * b}"
    elif operation == 'divide':
        if b == 0:
            return "Error: Division by zero"
        return f"{a} / {b} = {a / b}"
    else:
        return f"Unknown operation: {operation}"
""",
    dependencies=[]
)

# Get tool information
tool_info = get_tool(tool_id)
print(tool_info)

# List all tools
tools = get_tools()
print(tools)
```

### LLM Management

```python
from multi_tenant import add_llm, get_llm_config, get_llms

# Add a new LLM configuration
llm_id = add_llm(
    base_url="https://fmapi.swissai.cscs.ch",  # Swiss AI API
    api_key="YOUR_API_KEY",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    llm_name="llama-model"
)

# Get LLM information
llm_info = get_llm_config(llm_id)
print(llm_info)

# List all LLMs
llms = get_llms()
print(llms)
```

### Query Execution

```python
from multi_tenant import run_query

# Run a query using tools and LLM
result = run_query(
    tool_ids=[tool_id],
    llm_id=llm_id,
    query="What is 25 multiplied by 16?",
    use_docker=True  # Enable Docker execution (default: True)
)

print(result)
```

### Docker Support

The framework supports executing tools and code in a Docker container for enhanced security and isolation:

```python
from multi_tenant import run_query

# Run a query using Docker (if available)
result = run_query(
    tool_ids=[tool_id],
    llm_id=llm_id,
    query="Your query here",
    use_docker=True  # Default is True
)

# Force local execution
result = run_query(
    tool_ids=[tool_id],
    llm_id=llm_id,
    query="Your query here",
    use_docker=False
)
```

#### Docker Configuration

You can customize Docker execution through environment variables:

```
# In your .env file
DOCKER_IMAGE=python:3.9-slim  # Docker image to use
DOCKER_ENV_VARS=API_KEY,OTHER_VAR  # Comma-separated env vars to pass to container
```

#### Requirements for Docker Support

1. Docker must be installed and running on your system
2. The Python `docker` package must be installed:
   ```bash
   pip install docker
   ```

3. Your user must have permissions to use Docker

If Docker is not available, the framework will automatically fall back to local execution with a warning message.

### Using with Environment Variables

If you want to load configuration from environment variables:

```python
import os
from dotenv import load_dotenv
from multi_tenant import add_llm, run_query

# Load variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.environ.get("LLM_API_KEY")
model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
api_base = os.environ.get("API_BASE_URL", "https://fmapi.swissai.cscs.ch")

# Configure LLM
llm_id = add_llm(
    base_url=api_base,
    api_key=api_key,
    model_name=model_name,
    llm_name="env-llm"
)

# Use the LLM
result = run_query(
    tool_ids=[tool_id],
    llm_id=llm_id,
    query="Your query here"
)
```

### Full Example

Check out the example files:
- `example.py`: Basic usage example
- `test.py`: Testing the framework using environment variables

## Architecture

### Core Components

1. **Tool Management**
   - Tools are Python classes in the `/tools` directory inheriting from `smolagents.Tool`
   - Each tool is a separate file with filename matching the tool_id
   - Tool metadata stored in SQLite
   - Tools are validated against smolagents.Tool interface

2. **LLM Management**
   - Supports multiple LLM providers via smolagents.models
   - Uses smolagents.models.OpenAIServerModel for OpenAI-compatible APIs
   - Configurations stored in SQLite
   - Flexible model selection per request

3. **Storage Layer**
   - SQLite database with schema for tools and LLMs
   - Simple, file-based storage for easy deployment

## Directory Structure

```
/
├── multi_tenant/
│   ├── __init__.py         # Main module with public API
│   ├── base.py             # Base configuration and utilities
│   ├── tool_manager.py     # Tool file and metadata management
│   ├── llm_manager.py      # LLM configuration management
│   ├── docker_utils.py     # Docker support utilities
│   └── storage.py          # SQLite interface
├── tools/                  # Tool implementations
│   └── README.md           # Tool documentation
├── database/
│   └── schema.sql          # Database schema
├── .env.template           # Template for environment variables
├── example.py              # Usage example
├── test.py                 # Test script
└── README.md
```

## Dependencies

- Python 3.8+
- SQLite3
- smolagents >= 0.5.0
- openai >= 1.0.0
- python-dotenv >= 1.0.0
- docker (optional, for Docker support)

## Future Considerations

1. **Scaling**
   - Migration path to distributed databases
   - Caching layer for frequent queries

2. **Monitoring**
   - Usage analytics
   - Cost tracking

3. **Extended Features**
   - Tool versioning
   - Support for more model types from smolagents.models
   - LLM fallback configurations
   - Custom tool parameters
   - API key encryption
