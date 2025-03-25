# Tools Directory

This directory contains tool implementations for the multi-tenant agent framework.

## Structure

Each tool is stored as a separate Python file with a file name matching its `tool_id`.

## Tool Format

Tools are implemented as subclasses of `smolagents.Tool` following this format:

```python
from smolagents import Tool

class ExampleTool(Tool):
    name = "example"
    description = "This is an example tool"
    inputs = {
        "input_param": {
            "type": "string",
            "description": "Input parameter description"
        }
    }
    output_type = "string"
    
    def forward(self, input_param: str):
        # Implementation
        return "Example result"

# Create an instance of the tool
example_tool = ExampleTool()
```

## Adding Tools

Do not manually add files to this directory. Instead, use the `add_tool` function from the multi-tenant framework:

```python
from multi_tenant import add_tool

tool_id = add_tool(
    name="example",
    description="This is an example tool",
    code="""
def example_tool(input_param: str):
    '''
    Example tool function.
    
    Args:
        input_param: An input parameter
    
    Returns:
        String result
    '''
    # Tool implementation
    return f"Processed {input_param}"
    """,
    dependencies=["pandas", "numpy"]
) 