import os
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import time

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel
from multi_tenant.DockerExecutor import DockerExecutor

class Task:
    def __init__(self, json: object, scripts_dir: str):
        self.json = json
        self.scripts_dir = scripts_dir
        self.description = json.get('description', '')
        self.create = json.get('create', {})
        self.evaluation = json.get('evaluation', {})
        self.labels = json.get('labels', [])
        
        # Initialize DockerExecutor
        self.docker_executor = DockerExecutor()
        
        # Run initialization script if specified
        self._pre_run_init()
        
        # Initialize tools and agent
        self._init_agent()
    
    def _pre_run_init(self):
        """Pre-run initialization hook. Override in subclasses to implement specific initialization."""
        pass
    
    def _init_agent(self):
        """Initialize the agent with tools and model."""
        # Initialize tools
        self.tools = []
        
        # Initialize agent
        # self.model = OpenAIServerModel(
        #     model_id="meta-llama/Llama-3.3-70B-Instruct",
        #     api_base="https://fmapi.swissai.cscs.ch",
        #     api_key="sk-rc-UQRkeJAH8zmt9Pm-QeEEfg"
        # )
        self.model = OpenAIServerModel(
            model_id="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            executor_class=self.docker_executor
        )
    
    def _post_run_eval(self) -> Dict[str, Any]:
        """Post-run evaluation hook. Override in subclasses to implement specific evaluation.
            
        Returns:
            Dictionary containing evaluation results with keys:
            - success: bool indicating if evaluation was successful
            - ground_truth: the correct answer if available
            - is_correct: bool indicating if the response was correct
            - error: error message if any
        """
        return {
            "ground_truth": None,
            "is_correct": None,
            "error": None
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the task using the agent and evaluate the answer."""
        try:
            # Run the agent with the task description
            self.response = self.agent.run(self.description)
            
            # Get post-run evaluation results
            eval_results = self._post_run_eval()
            
            # Merge run results with evaluation results
            return {
                "success": True,
                "response": self.response,
                **eval_results
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "ground_truth": None,
                "is_correct": None,
                "error": str(e)
            }

class ParallelAgentRunner:
    """
    Runs multiple agent queries in parallel at a specified rate using a shared DockerExecutor.
    """
    
    def __init__(self, rate: float):
        """
        Initialize the parallel agent runner.
        
        Args:
            output_folder: Directory path where metrics and results should be saved
            rate: Number of queries to execute per second
        """
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0
        
        print(f"Initialized ParallelAgentRunner with rate: {rate} queries/second")
    
    def execute_batch(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks at the specified rate.
        
        Args:
            tasks: List of Task objects to execute
            
        Returns:
            List of result dictionaries
        """
        # Create a thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks to the thread pool
            futures = []
            for i, task in enumerate(tasks):
                future = executor.submit(task.run)
                futures.append(future)
                
                # Sleep to maintain the specified rate
                if i < len(tasks) - 1 and self.interval > 0:
                    time.sleep(self.interval)
            
            # Collect and return all results
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
                
            return results 