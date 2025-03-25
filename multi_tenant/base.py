#!/usr/bin/env python
# coding=utf-8

"""
Base utility functions and configurations for the multi-tenant agent framework.
"""

import os
import json
import uuid
import importlib
import inspect
import ast
import sys
from typing import Dict, List, Any, Optional, Union, Type

# Constants
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')

# Make sure tools directory exists
os.makedirs(TOOLS_DIR, exist_ok=True)

# Utility functions
def generate_id(prefix: str = '') -> str:
    """Generate a random UUID with an optional prefix."""
    return f"{prefix}{uuid.uuid4().hex}"

def validate_dependencies(dependencies: List[str]) -> List[str]:
    """Validate and normalize dependencies list."""
    if not isinstance(dependencies, list):
        raise ValueError("Dependencies must be a list of strings")
    
    # Ensure all dependencies are strings and not empty
    validated = []
    for dep in dependencies:
        if not isinstance(dep, str):
            raise ValueError(f"Dependency {dep} is not a string")
        if not dep.strip():
            continue
        validated.append(dep.strip())
    
    return validated

def is_valid_smolagents_tool(tool_code: str) -> bool:
    """
    Analyze the code to check if it defines a function that can be converted
    to a valid smolagents Tool.
    """
    try:
        # Parse the code to find function definitions
        tree = ast.parse(tool_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Found a function, verify it has a docstring and parameters
                docstring = ast.get_docstring(node)
                if not docstring:
                    return False
                
                # Function should have at least one parameter
                if not node.args.args:
                    return False
                
                # Check for type annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        return False
                
                return True
        
        # No valid function found
        return False
    except SyntaxError:
        # Invalid Python code
        return False

def function_to_tool_class(
    function_code: str, 
    name: str, 
    description: str, 
    dependencies: List[str]
) -> str:
    """
    Convert a function to a smolagents Tool class.
    
    Args:
        function_code: The function code as a string
        name: The name of the tool
        description: A description of what the tool does
        dependencies: List of package dependencies
    
    Returns:
        A string containing the Tool class code
    """
    try:
        # Parse the function to extract information
        tree = ast.parse(function_code)
        function_node = None
        
        # Find the function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_node = node
                break
        
        if function_node is None:
            raise ValueError("No function definition found in code")
        
        # Get function name and parameters
        func_name = function_node.name
        
        # Extract parameters and their types
        parameters = []
        for arg in function_node.args.args:
            if arg.arg == 'self':
                continue
            
            arg_name = arg.arg
            arg_type = "Any"  # Default type
            
            # Extract type annotation if available
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle complex types like List[str]
                    if isinstance(arg.annotation.value, ast.Name):
                        arg_type = arg.annotation.value.id
                        # TODO: Handle the subscript value if needed
            
            parameters.append((arg_name, arg_type))
        
        # Get docstring and extract parameter descriptions
        docstring = ast.get_docstring(function_node)
        param_descriptions = {}
        
        if docstring:
            # Extract parameter descriptions from docstring
            lines = docstring.split('\n')
            in_args = False
            current_param = None
            
            for line in lines:
                line = line.strip()
                if not in_args and line.lower().startswith('args:'):
                    in_args = True
                    continue
                
                if in_args:
                    if line.startswith(':param') or line.startswith('@param'):
                        # Handle :param style
                        parts = line.split(':', 2) if ':' in line else line.split(' ', 2)
                        if len(parts) >= 3:
                            param_name = parts[1].strip().split(' ')[0]
                            param_descriptions[param_name] = parts[2].strip()
                    elif line and not line.startswith(':') and not line.startswith('@') and ':' in line:
                        # Handle "param_name: description" style
                        parts = line.split(':', 1)
                        param_name = parts[0].strip()
                        param_descriptions[param_name] = parts[1].strip()
                    elif line.startswith(':') or line.startswith('@') or not line:
                        # End of Args section
                        in_args = False
        
        # Create the inputs dictionary for the Tool class
        inputs_dict = {}
        for param_name, param_type in parameters:
            # Map Python types to smolagents types
            type_mapping = {
                'str': 'string',
                'int': 'integer',
                'float': 'number',
                'bool': 'boolean',
                'list': 'array',
                'dict': 'object',
                'Any': 'any',
                'None': 'null',
            }
            
            type_str = type_mapping.get(param_type, param_type)
            
            # Get description from docstring or use a default
            param_desc = param_descriptions.get(param_name, f"Parameter {param_name}")
            
            inputs_dict[param_name] = {
                "type": type_str,
                "description": param_desc
            }
        
        # Determine output type from return annotation if available
        output_type = "string"  # Default output type
        if function_node.returns:
            if isinstance(function_node.returns, ast.Name):
                return_type = function_node.returns.id
                type_mapping = {
                    'str': 'string',
                    'int': 'integer',
                    'float': 'number',
                    'bool': 'boolean',
                    'list': 'array',
                    'dict': 'object',
                    'Any': 'any',
                    'None': 'null',
                }
                output_type = type_mapping.get(return_type, return_type)
        
        # Generate the Tool class code with the original function included
        # and the forward method simply calling the original function
        class_code = f"""
from smolagents import Tool
from typing import Any, Dict, List, Optional, Union

{function_code}

class {name.capitalize()}Tool(Tool):
    name = "{name}"
    description = "{description}"
    inputs = {json.dumps(inputs_dict, indent=4)}
    output_type = "{output_type}"
    
    def forward(self, {', '.join(f'{name}: {type_}' for name, type_ in parameters)}):
        return {func_name}({', '.join(name for name, _ in parameters)})

# Create an instance of the tool
{name}_tool = {name.capitalize()}Tool()
"""
        
        return class_code
    
    except Exception as e:
        raise ValueError(f"Failed to convert function to Tool class: {str(e)}") 