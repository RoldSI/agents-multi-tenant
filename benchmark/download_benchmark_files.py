from huggingface_hub import snapshot_download, login
import os
import argparse

def download_agentperf(output_dir, token=None):
    """
    Download the entire xiaozheyao/agentperf dataset to the specified directory
    
    Args:
        output_dir: Directory where the dataset should be saved
        token: HuggingFace token for authentication (optional if logged in)
    """
    print(f"Downloading xiaozheyao/agentperf dataset to {output_dir}...")
    
    # Authenticate if token is provided
    if token:
        login(token=token, add_to_git_credential=True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the entire repository
    repo_path = snapshot_download(
        repo_id="xiaozheyao/agentperf",
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False  # Set to True to save disk space with symlinks
    )
    
    print(f"Download complete! Files saved to: {repo_path}")
    return repo_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AgentPerf dataset from HuggingFace")
    parser.add_argument("--output", "-o", type=str, default="./agentperf",
                        help="Directory where the dataset should be saved")
    parser.add_argument("--token", "-t", type=str, default=None, 
                        help="HuggingFace token for authentication")
    
    args = parser.parse_args()
    download_agentperf(args.output, args.token)