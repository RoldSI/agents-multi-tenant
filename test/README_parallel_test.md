# Parallel Test Tool for Multi-Tenant Agent Framework

This tool allows you to execute multiple agent queries in parallel at a specified rate using a shared Docker executor. It's useful for testing the performance and behavior of the multi-tenant agent framework under load.

## Features

- Execute multiple queries in parallel with rate limiting
- Use a shared Docker executor instance to efficiently manage resources
- Dynamically create tools from simple function definitions
- Support for multiple functions in a single tool definition
- Track execution time and success/failure for each query
- Output detailed results to a JSON file

## Usage

### Input Format

The input file should be a JSON array where each object contains:

- `tools`: String containing one or more function definitions
- `query`: The query string to execute with these tools

Example:
```json
[
  {
    "tools": "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers\"\"\"\n    return a + b",
    "query": "Calculate 42 + 58"
  }
]
```

The tool definition should be a proper Python function with type hints and docstring. Multiple functions can be defined in a single "tools" string.

### Running the Tool

Basic usage:
```bash
python test/parallel_test.py test/sample_queries.json
```

With options:
```bash
python test/parallel_test.py test/sample_queries.json --rate 0.5 --output results.json
```

### Command Line Arguments

- `input_file`: Path to JSON file containing tasks to execute
- `--rate`: Number of queries to execute per second (default: 1.0)
- `--output`: Output file to write results (default: results.json)
- `--setup`: Run DockerExecutor.setup() before execution (builds/prepares Docker image)

### Example Output

```json
[
  {
    "task_id": 0,
    "query": "Calculate 42 + 58",
    "response": "The sum of 42 and 58 is 100.",
    "duration": 2.45,
    "success": true,
    "error": null
  }
]
```

## Requirements

- Python 3.8+
- Docker running locally
- OpenAI API key set in environment

## Environment Setup

Make sure to set your OPENAI_API_KEY in your environment or in a .env file:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

Or create a .env file:
```
OPENAI_API_KEY=your_openai_api_key
``` 