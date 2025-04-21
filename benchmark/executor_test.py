import os

from smolagents.tools import Tool, tool
from multi_tenant.DockerExecutor import DockerExecutor
from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv

from multi_tenant.base import function_to_tool_class

load_dotenv()

@tool
def execute_command(command: str) -> str:
    """Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute.
        
    Returns:
        str: The output of the command.
    """
    import os  # Ensure the required module is imported for isolated execution
    try:
        result = os.popen(command).read().strip()
        return result
    except Exception as e:
        return f"Error executing command: {e}"

@tool
def execute_shell_script(script: str) -> str:
    """Execute a shell script provided as a string and return its output.
    
    Args:
        script: The shell script to execute.
        
    Returns:
        str: The output of the script.
    """
    import subprocess  # Use subprocess for better control over script execution
    try:
        result = subprocess.run(
            script, 
            shell=True, 
            text=True, 
            capture_output=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error executing script: {result.stderr.strip()}"
    except Exception as e:
        return f"Error executing script: {e}"

def test_smolagents():
    """Test smolagents agent without using DockerExecutor"""
    model = OpenAIServerModel(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        api_base="https://fmapi.swissai.cscs.ch",
        api_key="sk-rc-UQRkeJAH8zmt9Pm-QeEEfg"
    )
    
    # Create an agent with the model
    agent = CodeAgent(
        tools=[execute_command],
        model=model,
        executor_class=DockerExecutor()
    )
    
    # Run the agent with a simple query
    # response = agent.run("Benchmark the os interface. Run as complicated scripts. those shouldn't generate much output")
    response = agent.run("write code that imports numpy and that executes a command via the tool")
    
    print(f"Smolagents agent response: {response}")
    return response

DockerExecutor.setup(force_rebuild=False)
test_smolagents()
print("DockerExecutor setup complete")

