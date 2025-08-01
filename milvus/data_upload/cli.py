import argparse
import os
import json
from .utils import prepare_bulk_writer, import_all_bulk_data, check_import_status, get_records_from_jsonl


def get_records_from_jsonl_files(input_folder: str, output_folder: str):
    """
    Extract records from all JSONL files in input folder and save to a single JSON file.

    Args:
        input_folder (str): Path to folder containing JSONL files
        output_folder (str): Path to folder where output JSON will be saved
    """
    all_files = os.listdir(input_folder)
    jsonl_files = [f for f in all_files if f.endswith(".jsonl")]

    for i, file_name in enumerate(jsonl_files):

        print(f"[{i+1}/{len(jsonl_files)}]: Processing file {file_name}...")
        file_path = os.path.join(input_folder, file_name)
        try:
            file_records = get_records_from_jsonl(file_path)
            print(f"  {len(file_records)} records extracted")
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            print("  Skipping this file...")
            continue

        # save records from this file into json file
        if file_records:
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"rows": file_records}, f, ensure_ascii=False, indent=4)
            print(f"Saved records from {file_name} to {output_file}\n")


def push_bulk_data(input_folder: str):
    """
    Process records and push them to MinIO using bulk writer.

    Args:
        input_folder (str): Path to folder containing JSON files with records
    """
    writer = prepare_bulk_writer()

    all_files = os.listdir(input_folder)
    json_files = [f for f in all_files if f.endswith(".json")]

    total_records = 0
    for file_name in json_files:
        print(f"Processing file {file_name}...")
        file_path = os.path.join(input_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            records = data.get("rows", [])

            for record in records:
                if len(record.get("chunk", "")) > 300:
                    print(f"  Skipping record with chunk length {len(record['chunk'])} > 300")
                    continue
                writer.append_row(record)
                total_records += 1

    writer.commit()
    batch_files = writer.batch_files

    print(f"Processed {total_records} records")
    print(f"Created {len(batch_files)} batch files:")
    for batch_file in batch_files:
        print(f"- {batch_file}")


def populate_milvus(batch_files: list = None):
    """
    Import all batch files from MinIO into Milvus collection.
    """

    if not batch_files:
        print("No batch files found in MinIO")
        return

    print(f"Found {len(batch_files)} batch files to import")
    resp = import_all_bulk_data(batch_files)

    if resp:
        print("Import response:")
        print(json.dumps(resp.json(), indent=4))

        job_id = resp.json()["data"]["jobId"]
        print(f"Import job started with ID: {job_id}")
        print("Checking import status...")
        status = check_import_status(job_id)
        print(status)
    else:
        print("No files to import")


def get_import_status_info(job_id: str):
    """
    Get and display import status for a specific job ID.

    Args:
        job_id (str): The import job ID to check
    """
    print(f"Checking status for job ID: {job_id}")
    status = check_import_status(job_id)
    print(status)


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(description="Milvus data import CLI utility")

    parser.add_argument(
        "--get-records-from-jsonl",
        nargs=2,
        metavar=("INPUT_FOLDER", "OUTPUT_FOLDER"),
        help="Extract records from JSONL files and save to JSON",
    )
    parser.add_argument("--push-bulk-to-minio", metavar="INPUT_FOLDER", help="Push bulk data to MinIO")
    parser.add_argument(
        "--populate-milvus",
        nargs="*",
        metavar="BATCH_FILE",
        help="Import batch files from MinIO to Milvus (specify files or leave empty for all)",
    )
    parser.add_argument("--get-import-status", metavar="JOB_ID", help="Get import status for job ID")

    args = parser.parse_args()

    if args.get_records_from_jsonl:
        input_folder, output_folder = args.get_records_from_jsonl
        get_records_from_jsonl_files(input_folder, output_folder)
    elif args.push_bulk_to_minio:
        push_bulk_data(args.push_bulk_to_minio)
    elif args.populate_milvus is not None:
        populate_milvus([args.populate_milvus])
    elif args.get_import_status:
        get_import_status_info(args.get_import_status)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
