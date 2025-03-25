#!/usr/bin/env python
# coding=utf-8

"""
Test script for Docker functionality in the multi-tenant agent framework.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import from our multi-tenant framework
from multi_tenant import (
    add_tool,
    add_llm,
    run_query,
    check_docker_available,
    test_docker_setup
)

def test_docker():
    """Test Docker functionality."""
    print("Multi-tenant Agent Framework - Docker Test")
    print("==========================================")
    
    # Step 1: Check if Docker is available
    print("\n1. Checking if Docker is available...")
    docker_available, error_msg = check_docker_available()
    
    if not docker_available:
        print(f"❌ Docker is not available: {error_msg}")
        print("Please install Docker and the Python docker package to use Docker functionality.")
        print("You can still use the framework with local execution.")
        return False
    
    print("✅ Docker is available")
    
    # Step 2: Test Docker setup with a simple container
    print("\n2. Testing Docker execution...")
    test_success, test_output = test_docker_setup()
    
    if not test_success:
        print(f"❌ Docker test failed: {test_output}")
        print("You can still use the framework with local execution.")
        return False
    
    print(f"✅ Docker test successful: {test_output}")
    
    # Step 3: Create a simple tool and run with Docker
    print("\n3. Running a test query with Docker...")
    
    # Get API key from environment
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("❌ LLM_API_KEY not set in .env file")
        print("Please create a .env file with your API key")
        return False
    
    # Create a simple echo tool
    tool_id = add_tool(
        name="echo",
        description="A simple echo tool that returns the input",
        code="""
def echo(message: str) -> str:
    '''
    Echo back the input message.
    
    Args:
        message: The message to echo
    
    Returns:
        The same message
    '''
    return f"Echo: {message}"
""",
        dependencies=[]
    )
    
    # Create LLM configuration
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    api_base_url = os.environ.get("API_BASE_URL", "https://fmapi.swissai.cscs.ch")
    
    llm_id = add_llm(
        base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        llm_name="test-llm"
    )
    
    # Run a query using Docker
    query = "Echo this message: Hello from Docker!"
    print(f"Query: {query}")
    
    try:
        result = run_query(
            tool_ids=[tool_id],
            llm_id=llm_id,
            query=query,
            use_docker=True
        )
        
        print(f"Result: {result}")
        print("\n✅ Docker execution successful")
        return True
    
    except Exception as e:
        print(f"❌ Error during Docker execution: {str(e)}")
        print("You can still use the framework with local execution.")
        return False

if __name__ == "__main__":
    success = test_docker()
    sys.exit(0 if success else 1) 