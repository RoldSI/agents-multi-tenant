import os

from smolagents.tools import Tool, tool
from multi_tenant.DockerExecutor import DockerExecutor
from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv

from multi_tenant.base import function_to_tool_class

load_dotenv()

def test_docker_executor():
    """Test the DockerExecutor without rebuilding the image"""
    # DockerExecutor setup is done outside this function
    executor = DockerExecutor()

    # Create a simple test with the executor
    result, output = executor._execute_code("print('Hello, world!')")
    print(f"Docker test result: {result}")
    print(f"Docker test output: {output}")

# @tool
# def add(a: int, b: int) -> int:
#     """Add two numbers together
    
#     Args:
#         a: The first number to add
#         b: The second number to add
        
#     Returns:
#         int: The sum of a and b
#     """
#     return a + b

asStr = """def add(a: int, b: int) -> int:
    \"\"\"Add two numbers together
    
    Args:
        a: The first number to add
        b: The second number to add
        
    Returns:
        int: The sum of a and b
    \"\"\"
    return a + b
"""

def fibonacci2(n: int) -> int:
    """
    Calculate the nth Fibonacci number
    
    Args:
        n: The position in the Fibonacci sequence (starting from 0)
        
    Returns:
        int: The nth Fibonacci number
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

asStr2 = """def fibonacci(n: int) -> int:
    \"\"\"
    Calculate the nth Fibonacci number
    
    Args:
        n: The position in the Fibonacci sequence (starting from 0)
        
    Returns:
        int: The nth Fibonacci number
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
"""

# class DivTool(Tool):
#     def __init__(self):
#         super().__init__()
#         self.name = "div"
#         self.description = "Divide two numbers together"
#         self.inputs = {"a": {'type': 'integer', 'description': 'a'}, "b": {'type': 'integer', 'description': 'b'}}
#         self.output_type = 'integer'

#     def forward(self, a: int, b: int) -> int:
#         """
#         Args:
#             a: a
#             b: b
#         """
#         return a / b

def test_smolagents():
    """Test smolagents agent without using DockerExecutor"""
    add = function_to_tool_class(asStr)
    fibonacci = function_to_tool_class(asStr2)
    print(add(1, 2))
    print(fibonacci(3))
    # Create an OpenAI server model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create an agent with the model
    agent = CodeAgent(
        tools=[fibonacci],
        model=model,
        executor_class=DockerExecutor
    )
    
    # Run the agent with a simple query
    response = agent.run("what is the 10th fibonacci number?")
    
    print(f"Smolagents agent response: {response}")
    return response

DockerExecutor.setup(force_rebuild=False)
# test_docker_executor()
test_smolagents()
print("DockerExecutor setup complete")

