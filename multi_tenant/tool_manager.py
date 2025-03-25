#!/usr/bin/env python
# coding=utf-8

"""
Tool management module for the multi-tenant agent framework.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Union

from smolagents import Tool

from .base import TOOLS_DIR, generate_id, validate_dependencies, is_valid_smolagents_tool, function_to_tool_class
from .storage import save_tool_metadata, get_tool_metadata, list_tools, delete_tool

def add_tool(
    name: str,
    description: str,
    code: str,
    dependencies: List[str]
) -> str:
    """
    Add a new tool to the system.
    
    Args:
        name: Tool name
        description: Tool description
        code: Tool function code
        dependencies: List of dependencies
    
    Returns:
        The tool_id
    """
    # Validate inputs
    if not name:
        raise ValueError("Tool name is required")
    if not description:
        raise ValueError("Tool description is required")
    if not code:
        raise ValueError("Tool code is required")
    
    # Validate that code is a valid tool function
    if not is_valid_smolagents_tool(code):
        raise ValueError(
            "Tool code must be a valid function with docstring and type annotations"
        )
    
    # Validate dependencies
    validated_deps = validate_dependencies(dependencies)
    
    # Generate tool ID (using the name as a base)
    sanitized_name = name.lower().replace(' ', '_')
    tool_id = generate_id(f"{sanitized_name}_")
    
    # Convert function to Tool class
    tool_class_code = function_to_tool_class(code, name, description, validated_deps)
    
    # Save to file system
    tool_file_path = os.path.join(TOOLS_DIR, f"{tool_id}.py")
    with open(tool_file_path, 'w') as f:
        f.write(tool_class_code)
    
    # Save metadata to database
    save_tool_metadata(tool_id, name, description, validated_deps)
    
    return tool_id

def get_tool(tool_id: str) -> Dict[str, Any]:
    """
    Retrieve tool information.
    
    Args:
        tool_id: The tool ID
    
    Returns:
        Dictionary with tool metadata
    """
    # Retrieve metadata from database
    tool_metadata = get_tool_metadata(tool_id)
    
    if tool_metadata is None:
        raise ValueError(f"Tool with ID {tool_id} not found")
    
    return {
        'tool_id': tool_metadata['tool_id'],
        'name': tool_metadata['name'],
        'description': tool_metadata['description'],
        'dependencies': tool_metadata['dependencies']
    }

def get_tools() -> List[Dict[str, str]]:
    """
    List all available tools.
    
    Returns:
        List of dictionaries with tool_id, name, and description
    """
    return list_tools()

def load_tool(tool_id: str) -> Optional[Tool]:
    """
    Load a tool from the filesystem.
    
    Args:
        tool_id: The tool ID
    
    Returns:
        An instance of the tool, or None if not found
    """
    # Check if tool exists in metadata
    tool_metadata = get_tool_metadata(tool_id)
    
    if tool_metadata is None:
        return None
    
    # Check if tool file exists
    tool_file_path = os.path.join(TOOLS_DIR, f"{tool_id}.py")
    
    if not os.path.exists(tool_file_path):
        return None
    
    try:
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(f"tools.{tool_id}", tool_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the tool instance
        # Each tool file should have a variable named {toolname}_tool
        tool_name = tool_metadata['name'].lower()
        tool_var_name = f"{tool_name}_tool"
        
        if hasattr(module, tool_var_name):
            return getattr(module, tool_var_name)
        
        # Alternative: find any variable that is an instance of Tool
        for var_name in dir(module):
            var = getattr(module, var_name)
            if isinstance(var, Tool):
                return var
        
        return None
    except Exception as e:
        # Log the error
        print(f"Error loading tool {tool_id}: {str(e)}")
        return None 