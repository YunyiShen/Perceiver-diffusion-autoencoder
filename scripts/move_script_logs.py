from pathlib import Path
import shutil

def move_script_logs():
    """
    Moves all files ending in .err or .out to a script_logs folder.
    
    Creates the script_logs directory if it doesn't exist and moves all matching
    files from the current directory and subdirectories into it.
    """
    # Get current directory using pathlib
    current_dir = Path.cwd()
    
    # Create script_logs directory if it doesn't exist
    logs_dir = current_dir / "script_logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Find all files ending with .err or .out recursively
    err_files = current_dir.rglob("*.err")
    out_files = current_dir.rglob("*.out")
    
    # Combine all matching files
    all_log_files = list(err_files) + list(out_files)
    
    # Move each file to script_logs directory
    for file_path in all_log_files:
        try:
            # Skip files that are already in script_logs directory
            if file_path.parent == logs_dir:
                continue
                
            # Create destination path
            dest_path = logs_dir / file_path.name
            
            # Handle filename conflicts by adding a number suffix
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                dest_path = logs_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Move the file
            shutil.move(str(file_path), str(dest_path))
            print(f"Moved: {file_path} -> {dest_path}")
            
        except Exception as e:
            print(f"Error moving {file_path}: {str(e)}")

move_script_logs()