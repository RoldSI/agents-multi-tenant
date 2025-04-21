#!/usr/bin/env python3
# coding=utf-8

"""
Script to generate performance plots from Docker execution metrics.
Takes a folder with JSON metrics files and generates plots in a subfolder.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_data(folder_path):
    """
    Load data from JSON files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing JSON files
        
    Returns:
        Tuple of (main_data, task_data) where task_data is a dictionary
        mapping run_ids to their task data
    """
    folder_path = Path(folder_path)
    
    # Load main.json file
    main_path = folder_path / "main.json"
    if not main_path.exists():
        print(f"Error: {main_path} not found")
        sys.exit(1)
        
    with open(main_path, 'r') as f:
        main_data = json.load(f)
    
    # Load individual task data files
    task_data = {}
    task_files = list(folder_path.glob("*.json"))
    
    for file_path in task_files:
        if file_path.name == "main.json":
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                run_id = file_path.stem  # Use filename without extension as run_id
                task_data[run_id] = data
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return main_data, task_data

def create_cumulative_plot(values, title, xlabel, ylabel, filename, folder):
    """
    Create a cumulative distribution plot.
    
    Args:
        values: List of values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        folder: Output folder
    """
    if len(values) == 0:
        print(f"Warning: No data for {title}, skipping plot")
        return
        
    # Convert to numpy array if not already
    values_array = np.array(values)
    
    # Sort values for cumulative plot
    sorted_values = np.sort(values_array)
    
    # Create percentiles for x-axis (0-100%)
    x_percentiles = np.linspace(0, 100, len(sorted_values))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_percentiles, sorted_values, marker='', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create quantile annotations
    quantiles = [0.25, 0.5, 0.75, 0.9]
    for q in quantiles:
        q_val = np.percentile(sorted_values, q * 100)
        plt.axhline(y=q_val, color='r', linestyle='--', alpha=0.3)
        plt.text(5, q_val * 1.02, f'{q*100}% ≤ {q_val:.4f}', color='r')
    
    plt.tight_layout()
    plt.savefig(Path(folder) / filename, dpi=300)
    plt.close()

def process_data(main_data, task_data, output_folder):
    """
    Process data and generate plots.
    
    Args:
        main_data: Data from main.json
        task_data: Dictionary of task data
        output_folder: Folder to save plots
    """
    # Extract task information
    tasks = []
    
    for run_id, data in task_data.items():
        # Skip if no code_executions data
        if "code_executions" not in data:
            continue
        
        # First find corresponding task in main_data to get total duration
        task_info = None
        for task in main_data.get("tasks", []):
            if task.get("run_id") == run_id:
                task_info = task
                break
        
        total_runtime = 0
        if task_info:
            total_runtime = task_info.get("duration", 0)
        else:
            # Skip if we can't find the task in main.json
            print(f"Warning: Could not find task {run_id} in main.json, skipping")
            continue
            
        # Get container startup time
        container_startup_time = data.get("container_startup_time", 0)
        
        # Extract execution times
        code_executions = data.get("code_executions", [])
        total_execution_time = data.get("total_execution_time", 0)
        
        # Find pip installation command (usually the 2nd execution)
        pip_execution_time = 0
        if len(code_executions) >= 2:
            for exec_info in code_executions:
                if "!pip install" in exec_info.get("code_snippet", ""):
                    pip_execution_time = exec_info.get("time", 0)
                    break
        
        # Calculate metrics using total_runtime from main.json as denominator
        docker_startup_ratio = 0
        if total_runtime > 0:
            docker_startup_ratio = container_startup_time / total_runtime
            
        docker_startup_pip_ratio = 0
        if total_runtime > 0:
            docker_startup_pip_ratio = (container_startup_time + pip_execution_time) / total_runtime
            
        docker_execution_ratio = 0
        if total_runtime > 0:
            docker_execution_ratio = total_execution_time / total_runtime
        
        exec_time_without_pip = 0
        docker_execution_without_pip_ratio = 0
        if total_runtime > 0:
            exec_time_without_pip = total_execution_time - pip_execution_time
            docker_execution_without_pip_ratio = exec_time_without_pip / total_runtime
        
        tasks.append({
            "run_id": run_id,
            "container_startup_time": container_startup_time,
            "pip_execution_time": pip_execution_time,
            "total_execution_time": total_execution_time,
            "execution_time_without_pip": exec_time_without_pip,
            "total_runtime": total_runtime,
            "docker_startup_ratio": docker_startup_ratio,
            "docker_startup_pip_ratio": docker_startup_pip_ratio,
            "docker_execution_ratio": docker_execution_ratio,
            "docker_execution_without_pip_ratio": docker_execution_without_pip_ratio
        })
    
    # Create pandas DataFrame for easier analysis
    df = pd.DataFrame(tasks)
    
    if len(df) == 0:
        print("Warning: No valid task data found. Skipping plot generation.")
        return
    
    # Get values as lists instead of numpy arrays to avoid truth value errors
    docker_startup_ratios = df["docker_startup_ratio"].tolist()
    docker_startup_pip_ratios = df["docker_startup_pip_ratio"].tolist()
    docker_execution_ratios = df["docker_execution_ratio"].tolist()
    docker_execution_without_pip_ratios = df["docker_execution_without_pip_ratio"].tolist()
    total_runtimes = df["total_runtime"].tolist()
    
    # Create plots
    create_cumulative_plot(
        docker_startup_ratios,
        "Cumulative Docker Startup Time / Total Task Duration",
        "Percentile (%)",
        "Docker Startup Time / Total Duration",
        "1_docker_startup_ratio.png",
        output_folder
    )
    
    create_cumulative_plot(
        docker_startup_pip_ratios,
        "Cumulative (Docker Startup + Pip Install) Time / Total Task Duration",
        "Percentile (%)",
        "(Docker Startup + Pip Install) Time / Total Duration",
        "2_docker_startup_pip_ratio.png",
        output_folder
    )
    
    create_cumulative_plot(
        docker_execution_ratios,
        "Cumulative Docker Execution Time / Total Task Duration",
        "Percentile (%)",
        "Docker Execution Time / Total Duration",
        "3_docker_execution_ratio.png",
        output_folder
    )
    
    create_cumulative_plot(
        docker_execution_without_pip_ratios,
        "Cumulative Docker Execution Time (without Pip Install) / Total Task Duration",
        "Percentile (%)",
        "Docker Execution Time (without Pip) / Total Duration",
        "4_docker_execution_without_pip_ratio.png",
        output_folder
    )
    
    create_cumulative_plot(
        total_runtimes,
        "Cumulative Runtime for Each Task",
        "Percentile (%)",
        "Task Runtime (seconds)",
        "5_total_runtime.png",
        output_folder
    )

def main():
    parser = argparse.ArgumentParser(description="Generate performance plots from Docker execution metrics")
    parser.add_argument("folder", type=str, help="Path to the folder containing JSON metrics files")
    args = parser.parse_args()
    
    # Validate the input folder
    input_folder = Path(args.folder)
    if not input_folder.exists():
        print(f"Error: Folder {input_folder} does not exist")
        sys.exit(1)
    
    # Create output folder
    output_folder = input_folder / "0-plots"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: {output_folder}")
    
    # Load data
    print(f"Loading data from {input_folder}...")
    main_data, task_data = load_data(input_folder)
    print(f"Loaded {len(task_data)} task data files")
    
    # Process data and generate plots
    print("Generating plots...")
    process_data(main_data, task_data, output_folder)
    
    print("Done! Plots have been saved to:", output_folder)

if __name__ == "__main__":
    main() 