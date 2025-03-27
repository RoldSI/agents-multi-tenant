#!/usr/bin/env python3
"""
Example of using a custom DockerExecutor with smolagents CodeAgent.

This script demonstrates how to use our custom Docker-based Python code execution
environment with the smolagents library by passing the executor_class parameter.
"""

import os
from typing import Type

from smolagents import CodeAgent, HfApiModel
from smolagents.local_python_executor import PythonExecutor
from multi_tenant.DockerExecutor import DockerExecutor

def main():
    """Run a simple example with our custom DockerExecutor."""
    
    # Build the Docker image once (this only needs to happen once in your application)
    print("Setting up DockerExecutor...")
    DockerExecutor.setup(force_rebuild=True)  # Force rebuild to use latest fixes
    
    # Create a model - using HfApiModel as an example
    # You can use any model supported by smolagents
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # Change to your preferred model
        provider="ollama",  # Change to your provider: openai, anthropic, together, etc.
    )
    
    # Create a CodeAgent with our custom DockerExecutor class
    agent = CodeAgent(
        tools=[],  # Add any tools you need
        model=model,
        executor_class=DockerExecutor,  # Pass the class type, not an instance
        executor_kwargs={
            "host": "127.0.0.1",  # Docker host
            # Add any other parameters for DockerExecutor initialization
        },
        additional_authorized_imports=["numpy", "pandas"],  # Add any imports the agent can use
    )
    
    # Run the agent with a task
    print("\nRunning agent with DockerExecutor...")
    result = agent.run("Create a pandas DataFrame with 5 rows of sample data and calculate the mean of each column.")
    
    print("\nFinal result:")
    print(result)

if __name__ == "__main__":
    main() 