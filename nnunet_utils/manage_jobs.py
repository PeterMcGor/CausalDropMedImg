import subprocess
import time
import argparse
from pathlib import Path

def parse_jobs_table(table_str):
    # Split into lines and remove empty lines
    lines = [line.strip() for line in table_str.split('\n') if line.strip()]
    
    # Skip header and separator lines
    data_lines = [line for line in lines[2:] if not all(c == '-' for c in line)]
    
    jobs = []
    for line in data_lines:
        # Split by whitespace and get the relevant fields
        parts = line.split()
        # The name is the second field, status is the fourth field
        jobs.append({
            'name': parts[1],
            'status': parts[3]
        })
    
    return jobs

def manage_jobs(status, action):
    """
    Manage jobs based on their status and desired action.
    
    Args:
        status (str): Status to filter jobs by (e.g., 'completed', 'stopped', 'running')
        action (str): Action to perform ('stop' or 'delete')
    """
    # Validate action
    if action not in ['stop', 'delete']:
        print(f"Invalid action: {action}. Must be 'stop' or 'delete'")
        return
    
    # Get list of all jobs
    print("Fetching job list...")
    result = subprocess.run(['sc', 'list', 'jobs'], 
                          capture_output=True, 
                          text=True)
    
    if result.returncode != 0:
        print(f"Error getting job list: {result.stderr}")
        return
    
    # Parse table output
    jobs = parse_jobs_table(result.stdout)
    
    # Filter jobs by status
    filtered_jobs = [job for job in jobs if job['status'] == status]
    print(f"\nFound {len(filtered_jobs)} jobs with status '{status}'")
    
    if not filtered_jobs:
        print(f"No {status} jobs to {action}")
        return
    
    # Show list of jobs to be processed
    print(f"\nJobs to be {action}d:")
    for job in filtered_jobs:
        print(f"- {job['name']}")
    
    # Ask for confirmation
    response = input(f"\nDo you want to {action} these {len(filtered_jobs)} {status} jobs? [y/N]: ")
    if response.lower() != 'y':
        print("Aborting")
        return
    
    # Process each job
    for i, job in enumerate(filtered_jobs, 1):
        job_name = job['name']
        print(f"\n{action.capitalize()}ing job {i}/{len(filtered_jobs)}: {job_name}")
        
        try:
            cmd_result = subprocess.run(['sc', action, 'job', job_name], 
                                      capture_output=True, 
                                      text=True)
            if cmd_result.returncode == 0:
                print(f"Successfully {action}d {job_name}")
            else:
                print(f"Failed to {action} {job_name}")
                print(f"Error: {cmd_result.stderr}")
        except Exception as e:
            print(f"Error {action}ing {job_name}: {str(e)}")
        
        # Small delay between actions
        time.sleep(1)
    
    print(f"\nJob {action} complete!")

def main():
    parser = argparse.ArgumentParser(description='Manage Saturn Cloud jobs based on status')
    parser.add_argument('--status', required=True, 
                       help='Status to filter jobs by (e.g., completed, stopped, running)')
    parser.add_argument('--action', required=True, choices=['stop', 'delete'],
                       help='Action to perform on filtered jobs')
    
    args = parser.parse_args()
    manage_jobs(args.status, args.action)

if __name__ == "__main__":
    main()
    # stop jobs
    # python manage_jobs.py --status completed --action stop

    # delete jobs
    # python manage_jobs.py --status stopped --action delete

    # stop all running jobs 
    # python manage_jobs.py --status running --action stop