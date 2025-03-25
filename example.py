#!/usr/bin/env python
# coding=utf-8

"""
Example usage of the multi-tenant agent framework.

This example demonstrates how to use the multi-tenant framework
with the smolagents package installed via pip.
"""

import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import from our multi-tenant framework
from multi_tenant import (
    add_tool,
    get_tool,
    get_tools,
    add_llm,
    get_llm_config,
    get_llms,
    run_query
)

def create_example_tool():
    """Create an example tool for calculation."""
    tool_id = add_tool(
        name="calculator",
        description="A simple calculator tool that can perform basic arithmetic operations",
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
    print(f"Created calculator tool with ID: {tool_id}")
    return tool_id

def create_example_llm():
    """Create an example LLM configuration."""
    # Get from environment variables or use defaults
    api_key = os.environ.get("LLM_API_KEY")
    
    if not api_key:
        print("ERROR: LLM_API_KEY environment variable not set.")
        print("Please create a .env file with LLM_API_KEY=your-api-key")
        sys.exit(1)
    
    # Get optional environment variables
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    api_base_url = os.environ.get("API_BASE_URL", "https://fmapi.swissai.cscs.ch")
    
    llm_id = add_llm(
        base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        llm_name="llama-model"
    )
    print(f"Created LLM configuration with ID: {llm_id}")
    return llm_id

def main():
    """Run the example."""
    print("Multi-tenant Agent Framework Example")
    print("===================================")
    print("Using smolagents installed via pip\n")
    
    # Verify smolagents is installed
    try:
        import smolagents
        print(f"smolagents version: {smolagents.__version__}")
    except (ImportError, AttributeError):
        print("Warning: smolagents is not installed or version info unavailable.")
        print("Please install with: pip install smolagents>=0.5.0")
    
    # Print configuration from .env
    print("\nConfiguration:")
    print(f"- API Base URL: {os.environ.get('API_BASE_URL', 'https://fmapi.swissai.cscs.ch')}")
    print(f"- Model Name: {os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.3-70B-Instruct')}")
    print(f"- API Key: {'Set' if os.environ.get('LLM_API_KEY') else 'Not Set'}")
    
    # Create example tool and LLM
    tool_id = create_example_tool()
    llm_id = create_example_llm()
    
    # List tools and LLMs
    print("\nAvailable Tools:")
    pprint(get_tools())
    
    print("\nAvailable LLMs:")
    pprint(get_llms())
    
    # Get tool and LLM details
    print("\nTool Details:")
    pprint(get_tool(tool_id))
    
    print("\nLLM Details:")
    try:
        llm_info = get_llm_config(llm_id)
        # Hide API key in output
        llm_info['api_key'] = "****" + llm_info['api_key'][-4:] if len(llm_info['api_key']) > 4 else "****"
        pprint(llm_info)
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Note: You need to set the LLM_API_KEY environment variable.")
        return
    
    # Run a query
    print("\nRunning a query...")
    
    query = "What is 25 multiplied by 16? Use the calculator tool."
    print(f"Query: {query}")
    
    # By default, Docker execution is enabled if Docker is available
    # If Docker is not available, it will automatically fall back to local execution
    result = run_query(
        tool_ids=[tool_id],
        llm_id=llm_id,
        query=query,
        use_docker=True  # Set to False to force local execution
    )
    
    print(f"Result: {result}")
    
    # Optional: Test with local execution explicitly
    print("\nRunning a query with local execution...")
    local_query = "What is 10 plus 20? Use the calculator tool."
    print(f"Query: {local_query}")
    
    local_result = run_query(
        tool_ids=[tool_id],
        llm_id=llm_id,
        query=local_query,
        use_docker=False  # Force local execution
    )
    
    print(f"Result: {local_result}")

if __name__ == "__main__":
    main() 