# ./prepare_batch.py prepare batch requests from extracted jsons for cleanup using openai Batch API.
# Run this file from the main directory of the project
# Use: python -m data-pipeline.utils.transform.batch_cleanup.prepare_batch <input_folder_path> <output_folder_path>

from .. import text_cleanup, table_cleanup, image_cleanup
import time
import json
import os
import argparse
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


def prepare_page_text_request(text: str, base_req_id: str):
    """Prepare a single text request for the OpenAI Batch API from the raw text to be cleaned."""
    
    if not text:
        return None
    agent = text_cleanup.TextCleanup(text=text)
    system_prompt = agent.system_instruction
    user_prompt = agent.instruction
    output_schema = agent.output_schema

    return {
        "custom_id": f"req_{base_req_id}_text",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_schema", "json_schema": output_schema},
            "temperature": 0.50,
            "top_p": 1,
            "max_completion_tokens": 2024,
            "stream": False,
        },
    }


def prepare_page_table_requests(tables: list[dict], base_req_id: str):
    all_requests = []
    skipped = 0
    for t in tables:
        try:
            data = t.get("data", None)
            agent = table_cleanup.TableCleanup(table_data=data)
        except ValueError:
            skipped += 1
            continue
        system_prompt = agent.system_instruction
        output_schema = agent.output_schema
        user_prompt = agent.instruction
        req = {
            "custom_id": f"req_{base_req_id}_t{t['table']+1}_table",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": output_schema,
                },
                "temperature": 0.50,
                "top_p": 1,
                "max_completion_tokens": 2024,
                "stream": False,
            },
        }
        all_requests.append(req)

    return all_requests, skipped


def prepare_page_image_requests(images: list[dict], base_req_id: str):
    all_requests = []
    skipped = 0
    for i, img in enumerate(images):
        try:
            agent = image_cleanup.ImageCleanup(image_data=img)
        except ValueError:
            skipped += 1
            continue
        
        system_prompt = agent.system_instruction
        user_prompt = agent.instruction
        output_schema = agent.output_schema
        
        base64_image = agent.img
        extension = agent.ext
        
        req = {
            "custom_id": f"req_{base_req_id}_i{img['image_id']}_image",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{extension};base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": output_schema,
                },
                "temperature": 0.50,
                "top_p": 1,
                "max_completion_tokens": 2024,
                "stream": False,
            },
        }
        all_requests.append(req)

    return all_requests, skipped


def prepare_full_batches_for_cleanup(input_folder: str, output_folder: str):
    texts_batch_requests = []
    tables_batch_requests = []
    images_batch_requests = []

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    global_start_time = time.time()
    try:
        # Create a list to hold the paths of all JSON files
        json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    except Exception as e:
        print(f"Error: {e}")
        return False

    if json_files:
        fl = len(json_files)
        print("="*100, f"Found {fl} JSON files in {input_folder}.", sep="\n")
        print("Begin processing ...", "-"*100, sep="\n")

        for nf, f in enumerate(json_files):
            # Construct the full file path
            file_path = os.path.join(input_folder, f)
            file_id = f.split(".")[0]
            
            # for logging
            texts_processed = 0
            images_processed = 0
            tables_processed = 0
            texts_skipped = tables_skipped = images_skipped = 0
            file_processing_start_time = time.time()

            with open(file_path, "r", encoding="utf-8") as file:
                pages = json.load(file)
                for np, p in enumerate(pages):
                    page_id = f"{file_id}_p{np+1}"
                    if p["plain_text"]:
                        texts_batch_requests.append(
                            prepare_page_text_request(p["plain_text"], page_id)
                        )
                        texts_processed += 1
                    if p["tables"]:
                        requests, skipped = prepare_page_table_requests(p["tables"], page_id)
                        tables_batch_requests.extend(requests)
                        tables_skipped += skipped
                        tables_processed += len(p["tables"])
                    if p["images"]:
                        requests, skipped = prepare_page_image_requests(p["images"], page_id)
                        images_batch_requests.extend(requests)
                        images_skipped += skipped
                        images_processed += len(p["images"])
            file_processing_end_time = time.time()
            file_processing_time = file_processing_end_time - file_processing_start_time
            print(
                f"[{nf+1}/{fl}]: {f} processed in {file_processing_time:.3f}",
                f"\t- Total data: {texts_processed} texts  {tables_processed} tables  {images_processed} images",
                f"\t- processed: {texts_processed - texts_skipped} texts  {tables_processed - tables_skipped} tables  {images_processed - images_skipped} images",
                f"\t- skipped: {texts_skipped} texts  {tables_skipped} tables  {images_skipped} images",
                sep="\n", end="\n\n"
            )

    # save each batch to a jsonl file
    if texts_batch_requests:
        text_output_file = os.path.join(output_folder, "texts_batch.jsonl")
        with open(text_output_file, "w", encoding="utf-8") as f:
            for req in texts_batch_requests:
                f.write(json.dumps(req) + "\n")

    if tables_batch_requests:
        tables_output_file = os.path.join(output_folder, "tables_batch.jsonl")
        with open(tables_output_file, "w", encoding="utf-8") as f:
            for req in tables_batch_requests:
                f.write(json.dumps(req) + "\n")

    if images_batch_requests:
        images_output_file = os.path.join(output_folder, "images_batch.jsonl")
        with open(images_output_file, "w", encoding="utf-8") as f:
            for req in images_batch_requests:
                f.write(json.dumps(req) + "\n")

    global_end_time = time.time()
    global_processing_time = global_end_time - global_start_time
    # print summary
    print("="*100, "Processing completed.", "="*100, sep="\n")
    print(f"Total processing time: {global_processing_time:.3f}s")
    print("Statistics:")
    print(f"\t- Text requests: {len(texts_batch_requests)}")
    print(f"\t- Table requests: {len(tables_batch_requests)}")
    print(f"\t- Image requests: {len(images_batch_requests)}")
    print(
        f"\t- Total requests: {len(texts_batch_requests) + len(tables_batch_requests) + len(images_batch_requests)}"
    )
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch requests for cleanup processing"
    )
    parser.add_argument(
        "input_folder", help="Path to input folder containing JSON files"
    )
    parser.add_argument(
        "output_folder", help="Path to output folder for batch JSONL files"
    )
    args = parser.parse_args()

    prepare_full_batches_for_cleanup(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
