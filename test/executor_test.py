import os

from smolagents.tools import tool
from multi_tenant.DockerExecutor import DockerExecutor
from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv

load_dotenv()

def test_docker_executor():
    """Test the DockerExecutor without rebuilding the image"""
    # DockerExecutor setup is done outside this function
    executor = DockerExecutor()

    # Create a simple test with the executor
    result, output = executor._execute_code("print('Hello, world!')")
    print(f"Docker test result: {result}")
    print(f"Docker test output: {output}")

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together
    
    Args:
        a: The first number to add
        b: The second number to add
        
    Returns:
        int: The sum of a and b
    """
    return a + b

def test_smolagents_without_docker():
    """Test smolagents agent without using DockerExecutor"""
    # Create an OpenAI server model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create an agent with the model
    agent = CodeAgent(
        tools=[add],
        model=model,
        executor_class=DockerExecutor
    )
    
    # Run the agent with a simple query
    response = agent.run("Use the add tool to compute 41+3+2")
    
    print(f"Smolagents agent response: {response}")
    return response

DockerExecutor.setup(force_rebuild=False)
# test_docker_executor()
test_smolagents_without_docker()
print("DockerExecutor setup complete")

