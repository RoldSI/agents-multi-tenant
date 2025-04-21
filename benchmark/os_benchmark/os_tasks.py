import os
from typing import Dict, Any
from smolagents.tools import tool
from benchmark.ParallelRunner import Task

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

class Task1(Task):
    def _pre_run_init(self):
        """Run the initialization script if specified in the task configuration."""
        # Validate create section
        if not hasattr(self, 'create') or 'init' not in self.create or 'file' not in self.create['init']:
            raise ValueError("Missing required init file in create section")
            
        init_file = os.path.join(self.scripts_dir, self.create['init']['file'])
        if not os.path.exists(init_file):
            raise ValueError(f"Init file not found: {init_file}")
            
        # Copy the init script to the container and execute it
        files = {
            "/tmp/init.sh": init_file
        }
        commands = [
            "chmod +x /tmp/init.sh",
            "/tmp/init.sh"
        ]
        self.docker_executor.execute_with_files(files, commands)
    
    def _post_run_eval(self) -> Dict[str, Any]:
        """Evaluate the agent's response by comparing it with ground truth.
        
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            ValueError: If evaluation configuration is invalid
        """
        # Validate evaluation section
        if not hasattr(self, 'evaluation'):
            raise ValueError("Missing evaluation section")
        if 'example' not in self.evaluation or 'code' not in self.evaluation['example']:
            raise ValueError("Missing example code in evaluation section")
        if 'check' not in self.evaluation or not isinstance(self.evaluation['check'], list):
            raise ValueError("Missing or invalid check section in evaluation")
            
        # Get ground truth by running example code
        # Wrap the command in sh -c to properly handle pipes
        example_result = self.docker_executor.execute_with_files(
            files={},
            commands=[self.evaluation['example']['code']]
        )
        ground_truth = example_result['0']['output'].strip()
        
        # Find the first valid check script
        check_script = None
        for check in self.evaluation['check']:
            if check and check.get('language') == 'python' and 'file' in check:
                check_script = check['file']
                break
                
        if not check_script:
            raise ValueError("No valid check script found in evaluation section")
            
        # Run the check script
        check_file = os.path.join(self.scripts_dir, check_script)
        if not os.path.exists(check_file):
            raise ValueError(f"Check file not found: {check_file}")
            
        # Copy the check file to the container and make it executable
        files = {
            "/tmp/check.py": check_file
        }
        commands = [
            "chmod +x /tmp/check.py",
            f"python /tmp/check.py '{self.response}' '{ground_truth}'"
        ]
        check_result = self.docker_executor.execute_with_files(files, commands)
        
        # The check script returns 0 for success (correct) and 1 for failure (incorrect)
        is_correct = check_result['1']['exit_code'] == 0
        
        return {
            "success": True,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "error": None
        }

class Task2(Task):
    def _pre_run_init(self):
        """Run environment setup commands specified in the task configuration."""
        # Only proceed if a create/init section is defined
        if not hasattr(self, 'create') or 'init' not in self.create:
            return
        init_items = self.create['init']
        # Normalize to a list
        if not isinstance(init_items, list):
            init_items = [init_items]
        # Iterate through each init step
        for item in init_items:
            if isinstance(item, dict) and 'file' in item:
                # A script file to run
                init_path = os.path.join(self.scripts_dir, item['file'])
                if not os.path.exists(init_path):
                    raise ValueError(f"Init file not found: {init_path}")
                files = {'/tmp/init.sh': init_path}
                commands = ['chmod +x /tmp/init.sh', '/tmp/init.sh']
            elif isinstance(item, str):
                # A shell command string
                files = {}
                commands = [item]
            else:
                # Unrecognized init item, skip
                continue
            self.docker_executor.execute_with_files(files, commands)

    def _post_run_eval(self) -> Dict[str, Any]:
        """Post-run evaluation for environment.json tasks."""
        # Validate evaluation section
        if not hasattr(self, 'evaluation') or 'example' not in self.evaluation or 'check' not in self.evaluation:
            raise ValueError("Missing evaluation or example/check in task config")
        # Extract example command (string or dict)
        example_cmd = (
            self.evaluation['example']
            if isinstance(self.evaluation['example'], str)
            else self.evaluation['example'].get('code')
        )
        # Run example to get ground truth
        example_result = self.docker_executor.execute_with_files(
            files={}, commands=[example_cmd]
        )
        ground_truth = example_result['0']['output'].strip()
        # Determine check item and type
        check_item = None
        check_type = None
        for item in self.evaluation['check']:
            if not item:
                continue
            if isinstance(item, str):
                check_item = item
                check_type = 'shell'
                break
            if isinstance(item, dict) and item.get('language') == 'python' and 'file' in item:
                check_item = item['file']
                check_type = 'python'
                break
        if check_item is None:
            raise ValueError("No valid check found in evaluation section")
        # Run check
        if check_type == 'shell':
            check_result = self.docker_executor.execute_with_files(
                files={}, commands=[check_item]
            )
            is_correct = check_result['0']['exit_code'] == 0
        else:
            check_file = os.path.join(self.scripts_dir, check_item)
            if not os.path.exists(check_file):
                raise ValueError(f"Check file not found: {check_file}")
            files = {'/tmp/check.py': check_file}
            commands = [
                "chmod +x /tmp/check.py",
                f"python /tmp/check.py '{self.response}' '{ground_truth}'"
            ]
            check_result = self.docker_executor.execute_with_files(
                files=files, commands=commands
            )
            is_correct = check_result['1']['exit_code'] == 0
        return {
            "success": True,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "error": None
        }
