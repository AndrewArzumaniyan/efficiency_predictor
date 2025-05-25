import subprocess
import os
import shutil
import logging
import logging

logger = logging.getLogger(__name__)

def run_command(command, check=False, output_file=None):
    """
    Run a command and return its output.
    If output_file is provided, log the output to this file.
    """
    logger.info(f"Executing: {command}")
    try:
        result = subprocess.run(
            command, 
            check=check, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(result.stdout)
        if result.stderr:
            logger.error(f"Error output: {result.stderr}")
        
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Output: \n{result.stdout}\n")
                if result.stderr:
                    f.write(f"Error output: \n{result.stderr}\n")
                f.write("-" * 50 + "\n")
                
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}: {command}\nError: {e.stderr}"
        logger.error(error_msg)
        
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"FAILED Command: {command}\n")
                f.write(f"Error (return code {e.returncode}): \n{e.stderr}\n")
                f.write("-" * 50 + "\n")
                
        return False, e