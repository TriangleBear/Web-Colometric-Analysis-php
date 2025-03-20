import os
import hashlib

def hash_file(file_path):
    """Generate a hash for the given file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def remove_duplicate_files(folder):
    # Dictionary to track seen files (by their hash)
    seen_files = {}

    # List all files in the folder
    files_in_folder = os.listdir(folder)

    for file in files_in_folder:
        # Ignore files with "- Copy" in the name
        if "- Copy" in file:
            continue

        file_path = os.path.join(folder, file)

        # Only check files (skip directories)
        if os.path.isfile(file_path):
            # Generate hash of the file
            file_hash = hash_file(file_path)

            # Check if the hash is already seen
            if file_hash in seen_files:
                # Duplicate found, remove the file
                os.remove(file_path)
                print(f"Removed duplicate file: {file}")  # Print the name of the deleted file
            else:
                # Save this file hash for future comparison
                seen_files[file_hash] = file_path

# Example usage:
folder = r"D:\Programming\Urine_Test_Strips\Test strips\labelimg mask images"  # Path to the folder
remove_duplicate_files(folder)
