#!/usr/bin/env python
# coding=utf-8

"""
Test script for the multi-tenant agent framework.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from multi_tenant import add_tool, add_llm, run_query, get_tool, get_llm_config

def run_test():
    print("Testing multi-tenant agent framework...")
    
    # Check if environment variable exists
    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key:
        print("ERROR: LLM_API_KEY environment variable not set. Create a .env file with LLM_API_KEY=your-api-key")
        return False
    
    # Get optional environment variables
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    api_base_url = os.environ.get("API_BASE_URL", "https://fmapi.swissai.cscs.ch")
    
    # 1. Create a simple test tool
    tool_id = add_tool(
        name="hello_world",
        description="A simple greeting tool",
        code="""
def greet(name: str) -> str:
    '''
    Greet someone by name.
    
    Args:
        name: The name to greet
    
    Returns:
        A greeting message
    '''
    return f"Hello, {name}! Nice to meet you."
""",
        dependencies=[]
    )
    print(f"Created tool with ID: {tool_id}")
    
    # 2. Create LLM config with the specified model
    llm_id = add_llm(
        base_url=api_base_url,
        api_key=llm_api_key, 
        model_name=model_name,
        llm_name="llama-model"
    )
    print(f"Created LLM config with ID: {llm_id}")
    
    # Print the LLM configuration
    llm_config = get_llm_config(llm_id)
    # Hide the API key in output
    safe_config = llm_config.copy()
    safe_config['api_key'] = "****" + safe_config['api_key'][-4:] if len(safe_config['api_key']) > 4 else "****"
    print(f"LLM Configuration: {safe_config}")
    
    # 3. Run a simple query
    query = "Say hello to Alice and then to Bob using the greet tool."
    print(f"\nRunning query: {query}")
    
    result = run_query(
        tool_ids=[tool_id],
        llm_id=llm_id,
        query=query
    )
    
    print(f"\nResult: {result}")
    
    # Check if result contains expected content (more flexible check)
    if "Hello" in result and ("Alice" in result or "Bob" in result):
        print("\nTEST PASSED: The framework is working correctly!")
        return True
    else:
        print("\nTEST FAILED: The response doesn't contain expected greetings.")
        return False

def check_database_and_files():
    """Check if database and tool files were created correctly."""
    print("\nVerifying database and tool files...")
    
    # Check database
    db_path = Path("database/multi_tenant.db")
    if db_path.exists():
        print(f"✓ Database exists: {db_path}")
    else:
        print(f"✗ Database not found: {db_path}")
    
    # Check tools directory
    tools_dir = Path("tools")
    if tools_dir.exists() and tools_dir.is_dir():
        print(f"✓ Tools directory exists")
        
        # Count .py files
        tool_files = list(tools_dir.glob("*.py"))
        if tool_files:
            print(f"✓ Found {len(tool_files)} tool files:")
            for tool_file in tool_files:
                print(f"  - {tool_file.name}")
        else:
            print("✗ No tool files found in tools directory")
    else:
        print("✗ Tools directory not found")

def main():
    # Print test environment
    print("Test Environment:")
    print(f"- API Base URL: {os.environ.get('API_BASE_URL', 'https://fmapi.swissai.cscs.ch')}")
    print(f"- Model Name: {os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.3-70B-Instruct')}")
    print(f"- API Key: {'Set' if os.environ.get('LLM_API_KEY') else 'Not Set'}")
    print("-" * 40)
    
    # Run the tests
    success = run_test()
    check_database_and_files()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 