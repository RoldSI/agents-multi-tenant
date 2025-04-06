#!/usr/bin/env python
# coding=utf-8

"""
Multi-tenant agent framework for smolagents.
"""

import os
from typing import Dict, List, Any, Optional, Union
from smolagents import CodeAgent, OpenAIServerModel
from smolagents.tools import Tool


def run_query(
    query: str,
    tools: List[Tool]
) -> str:
    """
    Execute a query using specified tools and LLM.
    
    Args:
        query: The query to execute
        use_docker: Whether to use Docker for execution (default: False, not used)
    
    Returns:
        Response from the agent
    """
    try:
        # Create OpenAI model client using environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        model = OpenAIServerModel(
            model="gpt-4o-mini",
            api_key=api_key
        )
        
        # Create agent with local execution
        agent = CodeAgent(
            tools=[],  # No custom tools for now
            model=model,
            add_base_tools=True  # Include default tools
        )
        
        # Execute query
        response = agent.run(query)
        
        return response
    
    except Exception as e:
        # Return error message
        return f"Error executing query: {str(e)}"

# Export public API
__all__ = [
    'run_query'
] 