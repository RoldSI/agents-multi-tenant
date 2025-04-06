#!/usr/bin/env python

"""
Base utility functions and configurations for the multi-tenant agent framework.
"""

import json
import ast
import inspect
import textwrap
import types
from typing import Dict, List, Any, Optional, Union, Type

from smolagents.tools import Tool
from smolagents._function_type_hints_utils import get_json_schema, TypeHintParsingException

def function_to_tool_class(
    function_code: str
) -> Tool:
    """
    Convert a string of a Python function into a Tool class instance.
    
    Args:
        function_code (str): String representation of a Python function.
            The function should have type hints for each input and a type hint for the output.
            It should also have a docstring including the description of the function
            and an 'Args:' part where each argument is described.
            
    Returns:
        Tool: An instance of a dynamically created Tool subclass.
    """
    # Create a temporary module to execute the function code
    temp_module = types.ModuleType('temp_function_module')
    
    # Execute the function code in the module's namespace
    exec(function_code, temp_module.__dict__)
    
    # Find the function in the module
    function_objects = [obj for name, obj in inspect.getmembers(temp_module) 
                       if inspect.isfunction(obj) and name != '<lambda>']
    
    if not function_objects:
        raise ValueError("No function found in the provided code string.")
    
    # Use the first function found
    tool_function = function_objects[0]
    
    # Get the function's schema
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        raise TypeHintParsingException("Tool return type not found: make sure your function has a return type hint!")

    # Create a SimpleTool class
    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]
    
    # Bind the tool function to the forward method
    SimpleTool.forward = staticmethod(tool_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    
    # Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    
    # Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create the forward method source, including def line and indentation
    tool_source_body = "\n".join(function_code.split("\n")[1:])
    tool_source_body = textwrap.dedent(tool_source_body)
    forward_method_source = f"def forward{str(new_sig)}:\n{textwrap.indent(tool_source_body, '    ')}"
    
    # Create the class source
    class_source = (
        textwrap.dedent(f'''
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        ''')
        + textwrap.indent(forward_method_source, "    ")  # indent for class method
    )
    
    # Store the source code on both class and method for inspection
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    return SimpleTool()