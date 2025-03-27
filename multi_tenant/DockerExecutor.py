import base64
import json
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

import docker
import requests
from docker.errors import DockerException, ImageNotFound
from websocket import create_connection

from smolagents.local_python_executor import BASE_BUILTIN_MODULES, PythonExecutor, DEFAULT_MAX_LEN_OUTPUT
from smolagents.tools import Tool


class DockerExecutor(PythonExecutor):
    """
    Executes Python code in an isolated Docker container running a Jupyter kernel.
    
    This executor builds a Docker image once (setup method) and then
    creates a persistent container for each executor instance.
    Code is executed by sending requests to the Jupyter kernel.
    
    Args:
        additional_authorized_imports (List[str]): Additional allowed packages
        logger (Optional): Logger to use for reporting status
        max_print_outputs_length (Optional[int]): Maximum length of print outputs
        image_name (str): Docker image name to use
        host (str): Host to bind container ports to
        port (int): Port to use for the container (0 for automatic assignment)
    """
    
    _image_built = False
    _image_name = "jupyter-kernel-executor"
    # Track used ports to avoid conflicts between instances
    _used_ports = set()
    
    _dockerfile_content = """
FROM python:3.12-slim

# Install Jupyter and common dependencies
RUN pip install jupyter-kernel-gateway numpy pandas matplotlib seaborn scikit-learn requests ipykernel

# Create workspace directory
WORKDIR /workspace

# Expose port for Jupyter
EXPOSE 8888

# Create config file
RUN mkdir -p /root/.jupyter && \\
    echo "c.KernelGatewayApp.allow_origin = '*'" > /root/.jupyter/jupyter_kernel_gateway_config.py && \\
    echo "c.KernelGatewayApp.allow_credentials = True" >> /root/.jupyter/jupyter_kernel_gateway_config.py && \\
    echo "c.JupyterWebsocketPersonality.list_kernels = True" >> /root/.jupyter/jupyter_kernel_gateway_config.py && \\
    echo "c.KernelGatewayApp.auth_token = ''" >> /root/.jupyter/jupyter_kernel_gateway_config.py

# Start Jupyter Kernel Gateway
CMD ["jupyter", "kernelgateway", "--ip=0.0.0.0", "--port=8888", "--no-auth"]
"""

    @classmethod
    def setup(cls, image_name=None, force_rebuild=False):
        """
        Build the Docker image needed for execution.
        This only needs to be done once per session.
        
        Args:
            image_name (str, optional): Custom name for the Docker image
            force_rebuild (bool, optional): Force rebuilding the image even if it exists
        """
        if cls._image_built and not force_rebuild:
            return
            
        if image_name:
            cls._image_name = image_name
            
        try:
            # Check if Docker is available
            client = docker.from_env()
            
            # Check if the image already exists
            try:
                if force_rebuild:
                    print(f"Force rebuilding Docker image '{cls._image_name}'...")
                    # Try to remove the existing image
                    try:
                        client.images.remove(cls._image_name, force=True)
                        print(f"Removed existing image '{cls._image_name}'")
                    except:
                        pass
                else:
                    client.images.get(cls._image_name)
                    cls._image_built = True
                    print(f"Docker image '{cls._image_name}' already exists.")
                    return
            except ImageNotFound:
                # Image doesn't exist, we'll build it
                pass
                
            # Create a temporary directory to build the Docker image
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create Dockerfile
                dockerfile_path = Path(temp_dir) / "Dockerfile"
                with open(dockerfile_path, "w") as f:
                    f.write(cls._dockerfile_content.strip())
                
                # Build the Docker image
                print(f"Building Docker image '{cls._image_name}'...")
                client.images.build(
                    path=temp_dir, 
                    tag=cls._image_name,
                    rm=True
                )
                
            cls._image_built = True
            print(f"Docker image '{cls._image_name}' built successfully.")
                
        except DockerException as e:
            print(f"Error setting up Docker: {str(e)}")
            print("Make sure Docker is installed and running.")
            raise

    @staticmethod
    def find_free_port(host="127.0.0.1", start_port=8000, end_port=9000):
        """
        Find a free port on the host system that's not in use or already allocated 
        to another DockerExecutor.
        
        Args:
            host (str): Host to check
            start_port (int): Start of port range to check
            end_port (int): End of port range to check
            
        Returns:
            int: Available port number
        """
        # First check ports that are not in our used_ports set
        for port in range(start_port, end_port):
            if port in DockerExecutor._used_ports:
                continue
                
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    DockerExecutor._used_ports.add(port)
                    return port
                except socket.error:
                    continue
        
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
    
    def __init__(
        self,
        additional_authorized_imports: List[str],
        logger=None,
        max_print_outputs_length: Optional[int] = None,
        image_name: str = None,
        host: str = "127.0.0.1",
        port: int = 0,  # 0 means auto-assign port
        **kwargs
    ):
        """
        Initialize the Docker executor with a persistent Jupyter kernel.
        
        Args:
            additional_authorized_imports (List[str]): Additional allowed packages
            logger: Logger for reporting status
            max_print_outputs_length (int, optional): Maximum length of print outputs
            image_name (str, optional): Custom Docker image name
            host (str, optional): Host to bind the container port to
            port (int, optional): Port to use for Jupyter (0 for auto-assignment)
        """
        self.logger = logger
        self.custom_tools = {}
        self.state = {"_print_outputs": ""}
        self.max_print_outputs_length = max_print_outputs_length or DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.static_tools = None
        
        # Docker-specific configuration
        self.image_name = image_name or self._image_name
        self.host = host
        
        # Assign a port if not explicitly provided
        if port == 0:
            self.port = self.find_free_port(host)
            if self.logger:
                self.logger.log(f"Auto-assigned port {self.port}")
        else:
            # If port is explicitly provided, check if it's available
            if port in self._used_ports:
                raise ValueError(f"Port {port} is already in use by another DockerExecutor")
                
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    self._used_ports.add(port)
                    self.port = port
                except socket.error:
                    raise ValueError(f"Port {port} is already in use by another process")
        
        self.container = None
        self.kernel_id = None
        self.ws = None
        self.final_answer_pattern = re.compile(r"final_answer\((.*)\)", re.DOTALL)
        
        # Ensure image is built
        if not self._image_built:
            self.setup(image_name=self.image_name)
            
        # Initialize Docker client and start container
        try:
            self.client = docker.from_env()
            self._start_container()
        except DockerException as e:
            if self.logger:
                self.logger.log_error(f"Failed to connect to Docker: {str(e)}")
            raise RuntimeError("Could not connect to Docker daemon. Make sure Docker is running.") from e

    def _start_container(self):
        """Start the persistent Docker container with Jupyter kernel."""
        try:
            if self.logger:
                self.logger.log(f"Starting container on {self.host}:{self.port}...", level=1)
            
            print(f"Docker version: {self.client.version()}")
                
            # Launch container
            self.container = self.client.containers.run(
                self.image_name,
                ports={f"8888/tcp": (self.host, self.port)},
                detach=True,
                name=f"jupyter-kernel-{self.port}-{uuid.uuid4().hex[:6]}",  # Unique name to avoid conflicts
                environment={
                    "JUPYTER_ALLOW_INSECURE_WRITES": "1",  # Allow insecure writes
                    "JUPYTER_ALLOW_ORIGIN": "*",
                    "JUPYTER_TOKEN": "",
                    "JUPYTER_PASSWORD": ""
                }
            )
            
            # Wait for container to be ready
            retries = 0
            max_retries = 20  # Increased from 10
            while retries < max_retries:
                self.container.reload()
                if self.container.status == "running":
                    break
                time.sleep(1)
                retries += 1
                
            if self.container.status != "running":
                raise RuntimeError(f"Container failed to start: {self.container.status}")
                
            # Wait for Jupyter to be ready
            retries = 0
            max_retries = 30  # Increased from 10
            print(f"Waiting for Jupyter kernel gateway on {self.host}:{self.port}")
            
            while retries < max_retries:
                try:
                    url = f"http://{self.host}:{self.port}/api/kernels"
                    print(f"Trying to connect to {url}...")
                    response = requests.get(url, timeout=2)
                    print(f"Response status: {response.status_code}")
                    print(f"Response content: {response.text[:100]}...")
                    
                    if response.status_code == 200:
                        print(f"Successfully connected to Jupyter at {self.host}:{self.port}")
                        break
                except Exception as e:
                    if retries % 5 == 0:  # Only log every 5th retry to reduce verbosity
                        print(f"Waiting for Jupyter to be ready (attempt {retries+1}/{max_retries}): {str(e)}")
                        # Get container logs for debugging
                        logs = self.container.logs().decode('utf-8')
                        print(f"Container logs: {logs[-500:]}")
                        if self.logger:
                            self.logger.log(f"Container logs: {logs[-500:]}", level=1)  # Last 500 chars of logs
                time.sleep(2)  # Increased from 1
                retries += 1
                
            if retries >= max_retries:
                logs = self.container.logs().decode('utf-8')
                error_msg = f"Jupyter kernel gateway failed to start. Container logs: {logs[-1000:]}"
                raise RuntimeError(error_msg)
                
            # Create a new kernel
            print(f"Creating kernel...")
            kernel_url = f"http://{self.host}:{self.port}/api/kernels"
            print(f"POST request to {kernel_url}")
            response = requests.post(kernel_url, timeout=5)
            print(f"Kernel creation response status: {response.status_code}")
            print(f"Kernel creation response text: {response.text}")
            
            if response.status_code != 201:
                raise RuntimeError(f"Failed to create kernel: {response.text}")
                
            # Store the kernel ID and connect to websocket
            kernel_data = response.json()
            self.kernel_id = kernel_data["id"]
            print(f"Created kernel with ID: {self.kernel_id}")
            
            self.ws_url = f"ws://{self.host}:{self.port}/api/kernels/{self.kernel_id}/channels"
            print(f"Connecting to WebSocket at {self.ws_url}")
            
            # Connect with retry
            ws_retries = 0
            max_ws_retries = 5
            while ws_retries < max_ws_retries:
                try:
                    self.ws = create_connection(self.ws_url, timeout=10)
                    print(f"Successfully connected to WebSocket")
                    break
                except Exception as ws_e:
                    print(f"WebSocket connection attempt {ws_retries+1}/{max_ws_retries} failed: {str(ws_e)}")
                    ws_retries += 1
                    time.sleep(2)
            
            if ws_retries >= max_ws_retries:
                raise RuntimeError(f"Failed to connect to WebSocket after {max_ws_retries} attempts")
            
            if self.logger:
                self.logger.log(f"Connected to Jupyter kernel {self.kernel_id} in container {self.container.short_id}", level=1)
                
            # Install additional packages
            if self.additional_authorized_imports:
                packages = " ".join(self.additional_authorized_imports)
                print(f"Installing packages: {packages}")
                self._execute_code(f"!pip install {packages}")
                
        except Exception as e:
            # Get container logs if available
            logs = ""
            if hasattr(self, 'container') and self.container:
                try:
                    logs = self.container.logs().decode('utf-8')
                    logs = f"\nContainer logs: {logs[-1000:]}"  # Last 1000 chars of logs
                except:
                    pass
            
            self.cleanup()
            raise RuntimeError(f"Failed to start Jupyter kernel: {e}{logs}") from e

    def _execute_code(self, code: str) -> Tuple[Any, str]:
        """
        Execute code in the Jupyter kernel and return result and output.
        
        Args:
            code (str): Python code to execute
            
        Returns:
            Tuple[Any, str]: (result, output)
        """
        if not self.ws or not self.kernel_id:
            self._start_container()
            
        # Generate a unique message ID
        msg_id = str(uuid.uuid4())
        
        # Create execute request
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }
        
        # Send the request
        self.ws.send(json.dumps(execute_request))
        
        # Collect output and results
        outputs = []
        result = None
        
        # Process messages until execution is complete
        while True:
            try:
                response = self.ws.recv()
                msg = json.loads(response)
                
                # Only process messages related to our request
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")
                if parent_msg_id != msg_id:
                    continue
                    
                msg_type = msg.get("msg_type", "")
                
                if msg_type == "stream":
                    # Output from print statements
                    text = msg["content"]["text"]
                    outputs.append(text)
                elif msg_type == "execute_result" or msg_type == "display_data":
                    # Results of execution
                    data = msg["content"].get("data", {})
                    if "text/plain" in data:
                        result = data["text/plain"]
                elif msg_type == "error":
                    # Error during execution
                    ename = msg["content"].get("ename", "")
                    evalue = msg["content"].get("evalue", "")
                    traceback = msg["content"].get("traceback", [])
                    error_msg = f"{ename}: {evalue}\n{''.join(traceback)}"
                    outputs.append(error_msg)
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    # Execution is complete
                    break
            except Exception as e:
                outputs.append(f"Error receiving message: {str(e)}")
                break
                
        return result, "".join(outputs)

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """
        Execute the provided code in the Jupyter kernel.
        
        Args:
            code_action (str): Python code to execute
            
        Returns:
            Tuple[Any, str, bool]: (result, logs, is_final_answer)
        """
        try:
            # Check if this is a final answer
            is_final_answer = bool(self.final_answer_pattern.search(code_action))
            final_answer_value = None
            
            # If we have tools or state, set them up in the kernel
            if self.state:
                state_code = f"""
# Set up state
state = {repr(self.state)}
locals().update(state)
"""
                self._execute_code(state_code)
                
            # Execute the main code
            if is_final_answer:
                # Extract the code before final_answer
                match = self.final_answer_pattern.search(code_action)
                pre_final_answer_code = code_action[:match.start()].strip()
                final_answer_expr = match.group(1).strip()
                
                # Execute the code first
                if pre_final_answer_code:
                    self._execute_code(pre_final_answer_code)
                    
                # Then evaluate the final answer expression
                result, output = self._execute_code(f"print(repr({final_answer_expr}))")
                final_answer_value = eval(output.strip())
            else:
                result, output = self._execute_code(code_action)
            
            # After execution, retrieve any updated state variables
            state_result, _ = self._execute_code("""
# Get updated state
import json
print(json.dumps({key: repr(value) for key, value in locals().items() 
                 if not key.startswith('_') and key != 'get_ipython'}))
""")
            
            try:
                # Update state with variables from the kernel
                updated_state = json.loads(state_result)
                for key, value_repr in updated_state.items():
                    try:
                        # Convert repr back to Python object
                        self.state[key] = eval(value_repr)
                    except:
                        pass
            except:
                pass
                
            # Update print outputs
            self.state["_print_outputs"] = output
            
            # Truncate if needed
            if len(self.state["_print_outputs"]) > self.max_print_outputs_length:
                self.state["_print_outputs"] = self.state["_print_outputs"][:self.max_print_outputs_length] + "\n... (truncated)"
            
            return final_answer_value if is_final_answer else result, output, is_final_answer
            
        except Exception as e:
            error_message = f"Error executing code in Docker: {str(e)}"
            if self.logger:
                self.logger.log_error(error_message)
            self.state["_print_outputs"] = error_message
            raise

    def send_variables(self, variables: dict):
        """
        Update the state with new variables.
        
        Args:
            variables (dict): Variables to add to the state
        """
        self.state.update(variables)
        
        # If we have an active kernel, send the variables there too
        if self.kernel_id:
            var_code = []
            for name, value in variables.items():
                var_code.append(f"{name} = {repr(value)}")
            
            if var_code:
                self._execute_code("\n".join(var_code))
    
    def send_tools(self, tools: Dict[str, Tool]):
        """
        Set the tools available to the code.
        
        Args:
            tools (Dict[str, Tool]): Tools available to the code
        """
        self.static_tools = tools
        
        # Send tool definitions to kernel
        if self.kernel_id and tools:
            # For real tool implementation in a kernel, we would need more complex
            # serialization. This is a simplified version that just makes tool
            # names available as dummy functions.
            tool_code = []
            for name in tools:
                tool_code.append(f"""
def {name}(*args, **kwargs):
    print(f"Tool '{name}' was called with args={{args}} and kwargs={{kwargs}}")
    print("Note: Tools cannot be directly executed in the Docker container.")
    return f"({name} called)"
""")
            
            if tool_code:
                self._execute_code("\n".join(tool_code))
    
    def cleanup(self):
        """
        Clean up resources (container, kernel, etc.).
        """
        try:
            # Close websocket connection
            if hasattr(self, 'ws') and self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
                
            # Stop and remove container
            if hasattr(self, 'container') and self.container:
                try:
                    self.container.stop(timeout=2)
                    self.container.remove()
                except:
                    pass
                self.container = None
                
            # Release the port
            if hasattr(self, 'port') and self.port and self.port in self._used_ports:
                self._used_ports.remove(self.port)
                
            if self.logger:
                self.logger.log("DockerExecutor cleanup completed", level=1)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """
        Ensure cleanup when the executor is destroyed.
        """
        self.cleanup()
