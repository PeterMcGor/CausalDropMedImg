import subprocess
import time
from pathlib import Path
import sys

def launch_jobs(yaml_dir="experiments_yamls/MSSEG2016_rater_experiments"):
    # Get all yaml files
    yaml_files = sorted(Path(yaml_dir).glob("*.yaml"))
    total_jobs = len(yaml_files)
    
    print(f"Found {total_jobs} jobs to launch")
    
    # Ask for confirmation
    response = input(f"Do you want to launch {total_jobs} jobs? [y/N]: ")
    if response.lower() != 'y':
        print("Aborting job launch")
        sys.exit(0)
    
    # Launch jobs with a small delay between each
    for i, yaml_file in enumerate(yaml_files, 1):
        print(f"\nLaunching job {i}/{total_jobs}: {yaml_file.name}")
        try:
            result = subprocess.run(['sc', 'apply', str(yaml_file), '--start'], 
                                 capture_output=True, 
                                 text=True)
            if result.returncode == 0:
                print(f"Successfully launched {yaml_file.name}")
            else:
                print(f"Failed to launch {yaml_file.name}")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Error launching {yaml_file.name}: {str(e)}")
        
        # Wait a few seconds between job submissions to avoid overloading the API
        time.sleep(2)
    
    print("\nJob launch complete!")

if __name__ == "__main__":
    launch_jobs()