#!/usr/bin/env python
# coding=utf-8

"""
Simple test script to verify smolagents works with Docker.
"""

import os
import sys
from dotenv import load_dotenv
from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel
from smolagents.tools import tool

@tool
def calculate(a: int, b: int) -> str:
    '''
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum as a string
    '''
    return f"{a} + {b} = {a + b}"


def test_smolagents_docker():
    """Test smolagents with Docker execution."""
    print("Smolagents Docker Test")
    print("=====================")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("❌ LLM_API_KEY not set in .env file")
        print("Please create a .env file with your API key")
        return False
    
    # Create LLM client
    model = OpenAIServerModel(
        # api_base=os.environ.get("API_BASE_URL", "https://fmapi.swissai.cscs.ch"),
        # model_id=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
        api_base=os.environ.get("API_BASE_URL", "http://127.0.0.1:1234/v1"),
        model_id="phi-4",
        api_key=api_key
    )
    
    # Create agent with Docker execution
    print("\nCreating CodeAgent with Docker execution...")
    agent = CodeAgent(
        tools=[calculate],
        model=model,
        add_base_tools=True,
        # executor_type="docker"  # Force Docker execution
    )
    
    # Run a test query
    print("\nRunning test query...")
    query = "What is 5 plus 3? Use the calculate tool."
    print(f"Query: {query}")
    
    try:
        result = agent.run(query)
        print(f"Result: {result}")
        print("\n✅ Smolagents Docker test successful")
        return True
    except Exception as e:
        print(f"❌ Error during Docker execution: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_smolagents_docker()
    sys.exit(0 if success else 1) 