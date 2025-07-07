# ./batching_cli.py
from openai import OpenAI
import os
import glob
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# # Initialize OpenAI client with API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def upload_file(filepath: str):
    """Uploads a single file to OpenAI."""
    print(f"Uploading file: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return None
    try:
        with open(filepath, "rb") as f:
            file_object = client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"Successfully uploaded: {os.path.basename(filepath)}")
        print(f"  File ID: {file_object.id}")
        print(f"  File size: {file_object.bytes} bytes")
        return file_object
    except Exception as e:
        print(f"Failed to upload {os.path.basename(filepath)}: {str(e)}")
        return None

def upload_folder(folder_path: str):
    """Uploads all files in a given folder to OpenAI."""
    print(f"Starting batch upload from folder: {folder_path}")
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist or is not a directory.")
        return
    
    files = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
    
    if not files:
        print(f"No files found in folder: {folder_path}")
        return
    
    print(f"Found {len(files)} files to upload.")
    
    uploaded_files = []
    failed_files = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {os.path.basename(file_path)}")
        result = upload_file(file_path)
        
        if result:
            uploaded_files.append(result)
        else:
            failed_files.append(file_path)
    
    print(f"\n{'='*50}")
    print("UPLOAD SUMMARY:")
    print(f"Successfully uploaded: {len(uploaded_files)} files")
    print(f"Failed uploads: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {os.path.basename(failed_file)}")
    
    if uploaded_files:
        print("\nUploaded file IDs:")
        for file_obj in uploaded_files:
            print(f"  - {file_obj.id}")

def create_batches(file_ids: list[str]):
    """Creates batch jobs from a list of file IDs."""
    print(f"Attempting to create batches for {len(file_ids)} file(s).")
    
    created_batches = []
    failed_files = []

    for file_id in file_ids:
        print(f"\nProcessing file ID: {file_id}")
        try:
            # This will throw an error if file does not exist, handled by except block
            batch = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(f"Successfully created batch for file ID {file_id}")
            print(f"  Batch ID: {batch.id}")
            created_batches.append(batch)
        except Exception as e:
            print(f"Failed to create batch for file ID {file_id}: {str(e)}")
            failed_files.append(file_id)

    print(f"\n{'='*50}")
    print("BATCH CREATION SUMMARY:")
    print(f"Successfully created: {len(created_batches)} batches")
    print(f"Failed for: {len(failed_files)} file IDs")

    if failed_files:
        print("\nFailed file IDs:")
        for file_id in failed_files:
            print(f"  - {file_id}")

    if created_batches:
        print("\nCreated batch IDs:")
        for batch in created_batches:
            print(f"  - {batch.id}")

def check_batch_status(batch_id: str):
    """Checks the status of a given batch."""
    print(f"Checking status for batch ID: {batch_id}")
    try:
        batch = client.batches.retrieve(batch_id)
        print("Batch Status:")
        print(f"  ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Created at: {batch.created_at}")
        if batch.output_file_id:
            print(f"  Output File ID: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"  Error File ID: {batch.error_file_id}")
        if batch.completed_at:
            print(f"  Completed at: {batch.completed_at}")
        if batch.failed_at:
            print(f"  Failed at: {batch.failed_at}")
        if batch.errors:
            print("  Errors:")
            for error in batch.errors.data:
                print(f"    - Code: {error.code}, Message: {error.message}")
    except Exception as e:
        print(f"Failed to retrieve batch status: {str(e)}")

def download_file_content(file_id: str, download_location: str):
    """Downloads a file's content from OpenAI."""
    print(f"Downloading content for file ID: {file_id}")
    try:
        # Retrieve file metadata to get the original filename
        file_metadata = client.files.retrieve(file_id)
        filename = file_metadata.filename
        
        # Ensure download location exists
        os.makedirs(download_location, exist_ok=True)
        
        # Get file content
        content_response = client.files.content(file_id)
        
        # Write content to a new file
        save_path = os.path.join(download_location, os.path.basename(filename))
        with open(save_path, "wb") as f:
            f.write(content_response.content)
            
        print(f"Successfully downloaded and saved file to: {save_path}")
    except Exception as e:
        print(f"Failed to download file content: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A CLI utility for OpenAI Batch API operations.")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--upload-folder", type=str, help="Path to the folder containing files to upload.")
    group.add_argument("--upload-file", type=str, help="Path to a single file to upload.")
    group.add_argument("--check-batch-status", type=str, help="The ID of the batch to check.")
    group.add_argument("--download-file-content", nargs=2, metavar=('FILE_ID', 'DOWNLOAD_LOCATION'), help="The ID of the file to download and the location to save it.")
    group.add_argument("--create-batch", nargs='+', metavar='FILE_ID', help="One or more file IDs to create batches from.")
    
    args = parser.parse_args()
    
    if args.upload_folder:
        upload_folder(args.upload_folder)
    elif args.upload_file:
        upload_file(args.upload_file)
    elif args.check_batch_status:
        check_batch_status(args.check_batch_status)
    elif args.download_file_content:
        file_id, download_location = args.download_file_content
        download_file_content(file_id, download_location)
    elif args.create_batch:
        create_batches(args.create_batch)