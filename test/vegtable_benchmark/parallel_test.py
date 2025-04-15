#!/usr/bin/env python
# coding=utf-8

"""
Parallel testing utility for multi-tenant agent framework.
Executes multiple queries in parallel using a shared DockerExecutor.
"""

import argparse
import json
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union
import multiprocessing
import queue
from datetime import datetime

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel
from smolagents.tools import Tool
from multi_tenant.DockerExecutor import DockerExecutor
from multi_tenant.base import function_to_tool_class

# Load environment variables from .env file
load_dotenv()

class ParallelAgentRunner:
    """
    Runs multiple agent queries in parallel at a specified rate using a shared DockerExecutor.
    """
    
    def __init__(self, rate: float, repeat_count: int = 1):
        """
        Initialize the parallel agent runner.
        
        Args:
            rate: Number of queries to execute per second
            repeat_count: Number of times the tasks were repeated
        """
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0
        self.results_queue = queue.Queue()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "run_id": self.run_id,
            "rate": rate,
            "repeat_count": repeat_count,
            "start_time": time.time(),
            "end_time": None,
            "total_duration": None,
            "tasks": [],
            "total_tasks": 0,
            "unique_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0
        }
        
        # if not self.api_key:
        #     raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create results directory if it doesn't exist
        os.makedirs("test/results", exist_ok=True)
        
        print(f"Initialized ParallelAgentRunner with rate: {rate} queries/second (Run ID: {self.run_id})")
    
    def execute_query(self, task_id: int, tools_code: str, query: str) -> Dict[str, Any]:
        """
        Execute a single query with the given tools.
        
        Args:
            task_id: Identifier for this task
            tools_code: String containing the tool code definitions
            query: The query to execute
            
        Returns:
            A dictionary containing the results
        """
        # Generate a unique ID for this task
        task_run_id = f"{self.run_id}_{task_id}"
        
        start_time = time.time()
        print(f"Task {task_id}: Starting execution of query: {query}")
        
        try:
            # Parse the tools code and create Tool instances
            tool_instances = []
            
            # Split the tools code into separate function definitions
            tool_codes = []
            current_function = []
            in_function = False
            
            # Process the input line by line to extract function definitions
            for line in tools_code.split('\n'):
                # Check for the start of a function definition
                if line.strip().startswith('def '):
                    if current_function:  # If we already have a function being processed
                        tool_codes.append('\n'.join(current_function))
                        current_function = []
                    in_function = True
                
                # Collect the line if we're in a function definition
                if in_function:
                    current_function.append(line)
                    
                    # Check if this might be the end of a function
                    # (a line with no indentation that's not a def, class, or empty line)
                    if (line.strip() and not line.startswith(' ') and not line.startswith('\t') and 
                        not line.startswith('def ') and not line.startswith('class ')):
                        in_function = False
            
            # Add the last function if there is one
            if current_function:
                tool_codes.append('\n'.join(current_function))
            
            # Create Tool instances from the extracted functions
            for code in tool_codes:
                if code.strip():  # Skip empty code
                    tool_instances.append(function_to_tool_class(code))
            
            # Create an OpenAI model for this query
            # model = OpenAIServerModel(
            #     model_id="gpt-4o-mini",
            #     api_key=self.api_key
            # )
            model = OpenAIServerModel(
                model_id="meta-llama/Llama-3.3-70B-Instruct",
                api_base="https://fmapi.swissai.cscs.ch",
                api_key="sk-rc-UQRkeJAH8zmt9Pm-QeEEfg"
            )
            
            # Create a DockerExecutor with the task's unique ID for tracking
            executor = DockerExecutor(run_id=task_run_id)
            
            # Create an agent with the executor
            agent = CodeAgent(
                tools=tool_instances,
                model=model
                # executor_class=executor
            )
            
            # Execute the query
            response = agent.run(query)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Prepare and return the result
            result = {
                "task_id": task_id,
                "run_id": task_run_id,
                "query": query,
                "response": response,
                "duration": duration,
                "success": True,
                "error": None
            }
            
            print(f"Task {task_id}: Completed in {duration:.2f}s")
            
            # Track task metrics
            self.metrics["tasks"].append({
                "task_id": task_id,
                "run_id": task_run_id,
                "duration": duration,
                "success": True,
                "query": query
            })
            self.metrics["successful_tasks"] += 1
            
            # Save updated metrics
            self._save_metrics()
            
            self.results_queue.put(result)
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            error_message = str(e)
            print(f"Task {task_id}: Failed with error: {error_message}")
            
            # Prepare and return the error result
            result = {
                "task_id": task_id,
                "run_id": task_run_id,
                "query": query,
                "response": None,
                "duration": duration,
                "success": False,
                "error": error_message
            }
            
            # Track task metrics
            self.metrics["tasks"].append({
                "task_id": task_id,
                "run_id": task_run_id,
                "duration": duration,
                "success": False,
                "query": query,
                "error": error_message
            })
            self.metrics["failed_tasks"] += 1
            
            # Save updated metrics
            self._save_metrics()
            
            self.results_queue.put(result)
            return result
    
    def _save_metrics(self):
        """Save the current metrics to the main.json file in the test/results directory."""
        # Update end_time and total_duration
        self.metrics["end_time"] = time.time()
        self.metrics["total_duration"] = self.metrics["end_time"] - self.metrics["start_time"]
        
        # Create results directory if it doesn't exist
        os.makedirs("test/results", exist_ok=True)
        
        # Save to file
        metrics_file = f"test/results/main.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def execute_batch(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks at the specified rate.
        
        Args:
            tasks: List of task dictionaries, each with 'tools' and 'query' keys
            
        Returns:
            List of result dictionaries
        """
        # Create a thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks to the thread pool
            futures = []
            for i, task in enumerate(tasks):
                future = executor.submit(
                    self.execute_query, 
                    i, 
                    task.get('tools', ''), 
                    task.get('query', '')
                )
                futures.append(future)
                
                # Sleep to maintain the specified rate
                if i < len(tasks) - 1 and self.interval > 0:
                    time.sleep(self.interval)
            
            # Collect and return all results
            results = []
            for future in futures:
                results.append(future.result())
                
            # Save final metrics
            self._save_metrics()
                
            return results
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results from the queue."""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results

def main():
    """Main function to parse arguments and run the parallel agent test."""
    parser = argparse.ArgumentParser(description='Run parallel agent queries with shared Docker executor')
    parser.add_argument('input_file', type=str, help='JSON file containing tasks to execute')
    parser.add_argument('--rate', type=float, default=1.0, 
                        help='Number of queries to execute per second (default: 1.0)')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file to write results (default: results.json)')
    parser.add_argument('--setup', action='store_true',
                        help='Run DockerExecutor.setup() before execution (builds/prepares Docker image)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat the tasks from the input file (default: 1)')
    
    args = parser.parse_args()
    
    # Run DockerExecutor setup if requested
    if args.setup:
        print("Setting up DockerExecutor...")
        DockerExecutor.setup(force_rebuild=False)
    
    # Load tasks from the input file
    try:
        with open(args.input_file, 'r') as f:
            original_tasks = json.load(f)
            
        # Duplicate tasks based on the repeat parameter
        tasks = []
        for _ in range(args.repeat):
            tasks.extend(original_tasks)
            
        unique_task_count = len(original_tasks)
        total_task_count = len(tasks)
        print(f"Loaded {unique_task_count} unique tasks, repeated {args.repeat} times for a total of {total_task_count} tasks")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Create and run the parallel agent
    runner = ParallelAgentRunner(args.rate, args.repeat)
    runner.metrics["unique_tasks"] = unique_task_count
    runner.metrics["total_tasks"] = total_task_count
    results = runner.execute_batch(tasks)
    
    # Write results to the output file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Execution complete. Processed {len(results)} tasks ({runner.metrics['unique_tasks']} unique, repeated {args.repeat} times).")
    print(f"Results written to {args.output}")
    print(f"Performance metrics written to test/results/main.json and individual task metrics in test/results/*.json")

if __name__ == "__main__":
    main()
