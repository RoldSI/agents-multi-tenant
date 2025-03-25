#!/usr/bin/env python
# coding=utf-8

"""
Multi-tenant agent framework for smolagents.
"""

import os
from typing import Dict, List, Any, Optional, Union

from .storage import initialize_database
from .tool_manager import add_tool, get_tool, get_tools, load_tool
from .llm_manager import add_llm, get_llm_config, get_llms, create_llm_client
from .base import TOOLS_DIR
from .docker_utils import check_docker_available, get_docker_config, test_docker_setup

# Initialize the database on import
initialize_database()

def run_query(
    tool_ids: List[str],
    llm_id: str,
    query: str,
    use_docker: bool = True
) -> str:
    """
    Execute a query using specified tools and LLM.
    
    Args:
        tool_ids: List of tool IDs to use
        llm_id: The LLM ID to use
        query: The query to execute
        use_docker: Whether to use Docker for execution (default: True)
    
    Returns:
        Response from the agent
    """
    try:
        # Import smolagents - if it fails, raise a helpful error
        try:
            from smolagents import CodeAgent
        except ImportError:
            raise ImportError(
                "smolagents is required to run queries. "
                "Install it with: pip install smolagents"
            )
        
        # Load tools
        tools = []
        for tool_id in tool_ids:
            tool = load_tool(tool_id)
            if tool is None:
                raise ValueError(f"Tool with ID {tool_id} could not be loaded")
            tools.append(tool)
        
        # Create LLM client
        llm_client = create_llm_client(llm_id)
        
        # Set up execution environment
        executor_type = "local"  # Default to local
        executor_kwargs = {}
        
        # If using Docker, check availability and set up Docker configuration
        if use_docker:
            docker_available, error_msg = check_docker_available()
            if docker_available:
                executor_type = "docker"
                # executor_kwargs = get_docker_config()
            else:
                print(f"Docker is not available: {error_msg}. Falling back to local execution.")
        
        # Create agent with specified executor
        agent = CodeAgent(
            tools=tools,
            model=llm_client,
            add_base_tools=True,  # Include default tools
            executor_type=executor_type  # Specify executor type
            # executor_kwargs=executor_kwargs  # Docker configuration if applicable
        )
        
        # Execute query
        response = agent.run(query)
        
        return response
    
    except Exception as e:
        # Return error message
        return f"Error executing query: {str(e)}"

# Export public API
__all__ = [
    'add_tool',
    'get_tool',
    'get_tools',
    'add_llm',
    'get_llm_config',
    'get_llms',
    'run_query',
    'check_docker_available',
    'test_docker_setup'
] 