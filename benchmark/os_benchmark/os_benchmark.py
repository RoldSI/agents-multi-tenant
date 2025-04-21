#!/usr/bin/env python
# coding=utf-8

"""
Parallel testing utility for multi-tenant agent framework.
Executes multiple queries in parallel using a shared DockerExecutor.
"""

import argparse
import json
import os

from dotenv import load_dotenv
from smolagents.tools import tool
from multi_tenant.DockerExecutor import DockerExecutor
from benchmark.os_benchmark.os_tasks import Task1, Task2
from benchmark.ParallelRunner import ParallelAgentRunner

# Load environment variables from .env file
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

def main():
    """Main function to parse arguments and run the parallel agent test."""
    parser = argparse.ArgumentParser(description='Run parallel agent queries with shared Docker executor')
    parser.add_argument('--rate', type=float, default=1.0, 
                        help='Number of queries to execute per second (default: 1.0)')
    parser.add_argument('--rebuild-docker-image', action='store_true', default=False,
                        help='Force rebuild Docker image before execution (builds/prepares Docker image)')
    parser.add_argument('--task-set', type=int, default=1,
                        help='ID of the task set to run (default: 1)')
    parser.add_argument('--task', type=int, default=None,
                        help='Only run the n-th task from the task set (default: None, run all tasks)')
    
    args = parser.parse_args()
    
    if args.rebuild_docker_image:
        DockerExecutor.setup(force_rebuild=True)
    else:
        DockerExecutor.setup(force_rebuild=False)
    
    # load tasks
    tasks = []
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, f'files/data/{args.task_set}')
    scripts_dir = os.path.join(current_dir, f'files/scripts/{args.task_set}')

    # Load tasks from JSON files in data directory
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    if isinstance(task_data, list):
                        for i, task_obj in enumerate(task_data):
                            # Skip if task index doesn't match --task argument
                            if args.task is not None and i != args.task:
                                continue
                            # Choose task class based on task set
                            if args.task_set == 1:
                                task = Task1(json=task_obj, scripts_dir=scripts_dir)
                            elif args.task_set == 2:
                                task = Task2(json=task_obj, scripts_dir=scripts_dir)
                            # Add the task to the list
                            tasks.append(task)
    else:
        print(f"Warning: Data directory {data_dir} does not exist. Using default task.")
        tasks = [("What is your working directory?", [execute_command, execute_shell_script], DockerExecutor())]

    # Create and run the parallel agent
    runner = ParallelAgentRunner(rate=args.rate)
    results = runner.execute_batch(tasks)

    print(f"RESULTS:\n{results}")

if __name__ == "__main__":
    main()
