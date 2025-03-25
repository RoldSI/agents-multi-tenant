#!/usr/bin/env python
# coding=utf-8

"""
Docker utilities for the multi-tenant agent framework.
"""

import os
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any

def check_docker_available() -> Tuple[bool, str]:
    """
    Check if Docker is available and running.
    
    Returns:
        Tuple containing:
        - Boolean indicating if Docker is available
        - String with error message if not available, empty string otherwise
    """
    try:
        # Try to import docker package
        import docker
        
        # Try to initialize client
        client = docker.from_env()
        
        # Check if Docker daemon is running by making a ping
        client.ping()
        
        return True, ""
    except ImportError:
        return False, "Docker Python package not installed. Run: pip install docker"
    except Exception as e:
        return False, f"Docker error: {str(e)}"

def get_docker_config() -> Dict[str, Any]:
    """
    Get Docker configuration for smolagents.
    
    Returns:
        Dictionary with Docker configuration parameters
    """
    config = {
        # Default Docker configuration
        "image": os.environ.get("DOCKER_IMAGE", "python:3.9-slim"),
        "working_dir": "/workspace",
        "volumes": {
            os.getcwd(): {"bind": "/workspace", "mode": "rw"}
        },
        "detach": False,
        "remove": True,
    }
    
    # Add custom environment variables from host if specified
    env_vars_to_pass = os.environ.get("DOCKER_ENV_VARS", "").split(",")
    if env_vars_to_pass and env_vars_to_pass[0]:  # Check if not empty string
        env = {}
        for var in env_vars_to_pass:
            var = var.strip()
            if var and var in os.environ:
                env[var] = os.environ[var]
        
        if env:
            config["environment"] = env
    
    return config

def test_docker_setup() -> Tuple[bool, str]:
    """
    Test Docker setup by running a simple container.
    
    Returns:
        Tuple containing:
        - Boolean indicating if the test was successful
        - Output from the test or error message
    """
    docker_available, error = check_docker_available()
    if not docker_available:
        return False, error
    
    try:
        import docker
        client = docker.from_env()
        
        # Run a simple test container
        container = client.containers.run(
            "python:3.9-slim",
            command="python -c 'print(\"Docker test successful\")'",
            remove=True
        )
        
        output = container.decode('utf-8').strip()
        return True, output
    except Exception as e:
        return False, f"Docker test failed: {str(e)}" 