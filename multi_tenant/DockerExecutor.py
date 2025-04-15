import base64
import json
import pickle
import re
import socket
import tempfile
import threading
import time
import uuid
import os
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

import docker
import requests
from docker.errors import DockerException, ImageNotFound
from smolagents.tool_validation import validate_tool_attributes
from smolagents.utils import instance_to_source
from websocket import create_connection

from smolagents.local_python_executor import BASE_BUILTIN_MODULES, PythonExecutor, DEFAULT_MAX_LEN_OUTPUT
from smolagents.tools import Tool, get_tools_definition_code
import websocket


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
        run_id (str, optional): Unique identifier for this executor instance to track metrics
    """
    
    _image_name = "jupyter-kernel-executor"
    # Track used ports to avoid conflicts between instances
    _used_ports = set()
    # Lock for accessing _used_ports to avoid race conditions
    _port_lock = threading.Lock()
    
    _dockerfile_content = """
FROM python:3.12-slim

# Install dependencies
RUN pip install jupyter-kernel-gateway ipykernel

RUN python -m ipykernel install --name python3 --display-name "Python 3"

# Expose the default port
EXPOSE 8888

CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
"""

    @classmethod
    def setup(cls, force_rebuild=False):
        """
        Build the Docker image needed for execution.
        This only needs to be done once per session.
        
        Args:
            image_name (str, optional): Custom name for the Docker image
            dockerfile_path (str, optional): Path to the Dockerfile to use
            force_rebuild (bool, optional): Force rebuilding the image even if it exists
        """
        if not force_rebuild:
            return
            
        try:
            # Check if Docker is available
            client = docker.from_env()
            
            # Check if the image already exists
            try:
                print(f"Force rebuilding Docker image '{cls._image_name}'...")
                # Try to remove the existing image
                client.images.remove(cls._image_name, force=True)
                print(f"Removed existing image '{cls._image_name}'")
            except Exception as e:
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
        # Acquire lock to prevent concurrent access to the port selection
        with DockerExecutor._port_lock:
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
        additional_authorized_imports: List[str] = [],
        logger=None,
        max_print_outputs_length: Optional[int] = None,
        host: str = "127.0.0.1",
        run_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Docker executor with a persistent Jupyter kernel.
        
        Args:
            additional_authorized_imports (List[str]): Additional allowed packages
            logger: Logger for reporting status
            max_print_outputs_length (int, optional): Maximum length of print outputs
            host (str, optional): Host to bind the container port to
            run_id (str, optional): Unique identifier for tracking metrics
        """
        self.logger = logger
        self.custom_tools = {}
        self.max_print_outputs_length = max_print_outputs_length or DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        
        # Docker-specific configuration
        self.host = host
        
        # Performance tracking
        self.run_id = run_id or str(uuid.uuid4())
        self.metrics = {
            "container_startup_time": 0,
            "code_executions": [],
            "total_execution_time": 0,
            "num_executions": 0
        }
        
        # Assign a port if not explicitly provided
        self.port = self.find_free_port(host)
        if self.logger:
            self.logger.log(f"Auto-assigned port {self.port}", level="debug")
        
        self.container = None
        self.kernel_id = None
        self.ws = None
        # Use the same regex pattern as in remote_executors.py
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$", re.M)
            
        # Initialize Docker client and start container
        try:
            self.client = docker.from_env()
            self._start_container()
        except DockerException as e:
            if self.logger:
                self.logger.log_error(f"Failed to connect to Docker: {str(e)}")
            raise RuntimeError("Could not connect to Docker daemon. Make sure Docker is running.") from e

    def _save_metrics(self):
        """Save performance metrics to a JSON file."""
        if not self.run_id:
            return
            
        # Create results directory if it doesn't exist
        os.makedirs("test/results", exist_ok=True)
        
        # Save metrics to file
        metrics_file = f"test/results/{self.run_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        if self.logger:
            self.logger.log(f"Metrics saved to {metrics_file}", level="debug")

    def _start_container(self):
        """Start the persistent Docker container with Jupyter kernel."""
        start_time = time.time()
        try:
            if self.logger:
                self.logger.log(f"Starting container on {self.host}:{self.port}...", level="info")
            
            if self.logger:
                self.logger.log(f"Docker version: {self.client.version()}", level="debug")
                
            # Launch container
            self.container = self.client.containers.run(
                self._image_name,
                ports={f"8888/tcp": (self.host, self.port)},
                detach=True,
                name=f"jupyter-kernel-{self.port}-{self.run_id[:6]}"  # Use run_id in container name

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

            # Verify container is responsive by polling the /api endpoint
            base_url = f"http://{self.host}:{self.port}"
            api_url = f"{base_url}/api"
            http_retries = 0
            max_http_retries = 20
            while http_retries < max_http_retries:
                try:
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        if self.logger:
                            self.logger.log("Container is responsive.", level="debug")
                        break
                except Exception as e:
                    if self.logger:
                        self.logger.log(f"Waiting for container to become responsive... ({e})", level="debug")
                time.sleep(1)
                http_retries += 1
            else:
                raise RuntimeError("Container did not become responsive in time.")

            # Create a new kernel by POSTing to /api/kernels
            kernel_url = f"{base_url}/api/kernels"
            response = requests.post(kernel_url)
            if self.logger:
                self.logger.log(f"Kernel response: {response.status_code} {response.text}", level="debug")
                
            if response.status_code != 201:
                raise RuntimeError(f"Failed to create kernel: {response.status_code} {response.text}")
            kernel_info = response.json()
            self.kernel_id = kernel_info.get("id")
            if self.logger:
                self.logger.log(f"Kernel created with ID: {self.kernel_id}", level="debug")

            # Construct and store the WebSocket URL for kernel channels
            self.ws_url = f"ws://{self.host}:{self.port}/api/kernels/{self.kernel_id}/channels"
            if self.logger:
                self.logger.log(f"WebSocket URL: {self.ws_url}", level="debug")
            
            # Create WebSocket connection
            self.ws = websocket.create_connection(self.ws_url)
            if self.logger:
                self.logger.log("WebSocket connection established.", level="debug")

            # Install additional authorized imports (packages) via pip inside the container
            if getattr(self, "additional_authorized_imports", None):
                packages = " ".join(self.additional_authorized_imports)
                exec_command = f"pip install {packages}"
                
                if self.logger:
                    self.logger.log(f"Installing additional packages: {packages}", level="info")
                    
                exit_code, output = self.container.exec_run(exec_command)
                if exit_code != 0:
                    raise RuntimeError(f"Failed to install packages: {output.decode('utf-8')}")
                    
                if self.logger:
                    self.logger.log("Additional packages installed.", level="debug")
             
            # Record container startup time
            container_startup_time = time.time() - start_time
            self.metrics["container_startup_time"] = container_startup_time
            
            if self.logger:
                self.logger.log(f"Container startup took {container_startup_time:.2f} seconds", level="info")
                
            # Save initial metrics
            self._save_metrics()
                
        except Exception as e:
            # Get container logs if available
            if self.logger:
                self.logger.log_error(f"Error starting container: {e}")
            
            self.cleanup()
            raise RuntimeError(f"Failed to start Jupyter kernel: {e}") from e

    def _execute_code(self, code: str) -> Tuple[Any, str]:
        """
        Execute code in the Jupyter kernel and return result and output.
        
        Args:
            code (str): Python code to execute
            
        Returns:
            Tuple[Any, str]: (result, output)
        """
        if not self.ws or not self.kernel_id:
            raise RuntimeError("Container with kernel not started")
            
        # Generate a unique message ID
        msg_id = str(uuid.uuid4())

        # Start timing
        start_time = time.time()
        
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
        
        # Record execution time
        execution_time = time.time() - start_time
        
        # Update metrics
        if self.run_id:
            code_snippet = code[:50] + "..." if len(code) > 50 else code
            execution_record = {
                "time": execution_time,
                "timestamp": time.time(),
                "code_snippet": code_snippet,
                "success": not any(line.startswith(("Exception", "Error")) for line in outputs)
            }
            self.metrics["code_executions"].append(execution_record)
            self.metrics["total_execution_time"] += execution_time
            self.metrics["num_executions"] += 1
            
            # Save updated metrics
            self._save_metrics()
                
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
            output = ""
                
            if is_final_answer:
                # Extract the code before final_answer and the final answer expression
                match = self.final_answer_pattern.search(code_action)
                pre_final_answer_code = code_action[:match.start()].strip()
                final_answer_expr = match.group(1).strip()
                
                # Execute any code that comes before the final_answer call
                if pre_final_answer_code:
                    _, pre_output = self._execute_code(pre_final_answer_code)
                    output += pre_output
                
                # Evaluate the final answer expression with the executor
                _, expr_output = self._execute_code(f"print(repr({final_answer_expr}))")
                output += expr_output
                
                # Extract the evaluated result
                try:
                    # Get the last line which should contain our result
                    result_line = expr_output.strip().split('\n')[-1]
                    final_answer_value = eval(result_line)
                except Exception as e:
                    # If evaluation fails, log the error and use the expression as is
                    if self.logger:
                        self.logger.log_error(f"Failed to evaluate final answer: {str(e)}")
                    final_answer_value = final_answer_expr
            else:
                # Regular code execution
                result, output = self._execute_code(code_action)
            
            # Truncate output if needed
            if len(output) > self.max_print_outputs_length:
                output = output[:self.max_print_outputs_length] + "\n... (truncated)"
            
            return final_answer_value if is_final_answer else result, output, is_final_answer
            
        except Exception as e:
            error_message = f"Error executing code in Docker: {str(e)}"
            if self.logger:
                self.logger.log_error(error_message)
            raise

    def send_variables(self, variables: dict):
        """
        Send variables to the kernel namespace using pickle.
        """
        pickled_vars = base64.b64encode(pickle.dumps(variables)).decode()
        code = f"""
import pickle, base64
vars_dict = pickle.loads(base64.b64decode('{pickled_vars}'))
locals().update(vars_dict)
"""
        self._execute_code(code)
    
    def send_tools(self, tools: Dict[str, Tool]):
        """
        Set the tools available to the code by defining them in the kernel.
        
        This method installs any required packages for the tools and then
        defines the tool functions in the kernel's namespace.
        
        Args:
            tools (Dict[str, Tool]): Dictionary mapping tool names to Tool objects
                                     that will be made available to the code running
                                     in the container.
        
        Raises:
            RuntimeError: If the kernel is not started when tools are provided
        """
        if not self.kernel_id and tools:
            raise RuntimeError("Kernel not started")

        # tool_definition_code = get_tools_definition_code(tools)
        tool_definition_codes = set()
        packages_to_install = set()
        packages_to_install.add("smolagents")

        for tool in tools.values():
            # Encode the tool using pickle and base64 for secure transmission
            tool_code = tool.to_dict()["code"]
            tool_code += f"\n{tool.name} = {tool.__class__.__name__}()"
            tool_definition_codes.add(tool_code)
            for package in tool.to_dict()["requirements"]:
                packages_to_install.add(package)

        self._execute_code(
            f"!pip install {' '.join(packages_to_install)}\n{'\n'.join(tool_definition_codes)}"
        )
    
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
            if hasattr(self, 'port') and self.port:
                with DockerExecutor._port_lock:
                    if self.port in self._used_ports:
                        self._used_ports.remove(self.port)
                
            if self.logger:
                self.logger.log("DockerExecutor cleanup completed", level=1)
                
            # Save final metrics
            if hasattr(self, 'run_id') and self.run_id:
                self._save_metrics()
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """
        Ensure cleanup when the executor is destroyed.
        """
        self.cleanup()
