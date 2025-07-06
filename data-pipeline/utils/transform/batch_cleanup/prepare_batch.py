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
    agent = text_cleanup.TextCleanup(text=text)
    system_prompt = agent.SYSTEM_INSTRUCTION
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
    system_prompt = table_cleanup.TableCleanup().SYSTEM_INSTRUCTION
    output_schema = table_cleanup.TableCleanup().output_schema

    for t in tables:
        if t["data"]:
            agent = table_cleanup.TableCleanup(table_data=t)
            user_prompt = agent.instruction
            req = {
                "custom_id": f"req_{base_req_id}_t{t['table']}_table",
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

    return all_requests


def prepare_page_image_requests(images: list[dict], base_req_id: str):
    all_requests = []
    system_prompt = image_cleanup.ImageCleanup().SYSTEM_INSTRUCTION
    output_schema = image_cleanup.ImageCleanup().output_schema

    for i, img in enumerate(images):
        agent = image_cleanup.ImageCleanup(image_data=img)
        user_prompt = agent.instruction
        base64_image = agent.img
        extension = agent.ext
        req = {
            "custom_id": f"req_{base_req_id}_i{i}_image",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini",
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
        all_requests.append(req)  # This was incorrectly indented

    return all_requests


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
        print(f"Found {fl} JSON files in {input_folder}.")
        print("Begin processing ...")

        for nf, f in enumerate(json_files):
            file_path = os.path.join(input_folder, f)
            file_id = f.split(".")[0]
            ntxt = nimg = ntbl = 0
            file_processing_start_time = time.time()

            with open(file_path, "r", encoding="utf-8") as file:
                pages = json.load(file)
                for np, p in enumerate(pages):
                    page_id = f"{file_id}_p{np}"
                    if p["plain_text"]:
                        texts_batch_requests.append(
                            prepare_page_text_request(p["plain_text"], page_id)
                        )
                        ntxt += 1
                    if p["tables"]:
                        tables_batch_requests.extend(
                            prepare_page_table_requests(p["tables"], page_id)
                        )
                        ntbl += len(p["tables"])
                    if p["images"]:
                        images_batch_requests.extend(
                            prepare_page_image_requests(p["images"], page_id)
                        )
                        nimg += len(p["images"])
            file_processing_end_time = time.time()
            file_processing_time = file_processing_end_time - file_processing_start_time
            print(
                f"[{nf+1}/{fl}]: {f} processed in {file_processing_time:.3f}s: ({ntxt} texts  {ntbl} tables  {nimg} images)"
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
    print("Processing completed.")
    print(f"Total processing time: {global_processing_time:.3f}s")
    print("Statistics:")
    print(f"\t- Text requests: {len(texts_batch_requests)}")
    print(f"\t- Table requests: {len(tables_batch_requests)}")
    print(f"\t- Image requests: {len(images_batch_requests)}")
    print(
        f"\t- Total requests: {len(texts_batch_requests) + len(tables_batch_requests) + len(images_batch_requests)}"
    )


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
