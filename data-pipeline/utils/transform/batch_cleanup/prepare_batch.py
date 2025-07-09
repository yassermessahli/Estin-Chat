"""
Prepare batch requests for cleanup processing

Usage: prepare_batch.py <input_folder> <output_folder>

Positional arguments:
  input_folder   Path to input folder containing JSON files extracted from raw PDFs
  output_folder  Path to output folder to store batch JSONL files for each data type

Options:
  -h, --help     show this help message and exit
  
NOTE:
You should run this script as a module from the root directory of the repository as follows:
python -m data_pipeline.utils.transform.batch_cleanup.prepare_batch <input_folder> <output_folder>
"""

from .. import text_cleanup, table_cleanup, image_cleanup
import time
import json
import os
import argparse
import dotenv
import tiktoken

# Load environment variables from .env file
dotenv.load_dotenv()


def count_tokens(text: str, model: str = None) -> int:
    """tokens counter from a text string based on a specific model tokenizer."""
    if not text:
        return 0
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def prepare_page_text_request(text: str, base_req_id: str):
    """Prepare a single text request for the OpenAI Batch API from the raw text to be cleaned."""
    
    if not text:
        return None
    agent = text_cleanup.TextCleanup(text=text, context=base_req_id)
    system_prompt = agent.system_instruction
    user_prompt = agent.instruction
    output_schema = agent.output_schema

    request = {
        "custom_id": f"req_{base_req_id}_text",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": os.getenv("TEXTS_CLEANUP_MODEL", "gpt-4.1-mini"),
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
    # count the total input tokens
    input_tokens = count_tokens(text=system_prompt + user_prompt, model="gpt-4")
    return request, input_tokens


def prepare_page_table_requests(tables: list[dict], base_req_id: str):
    all_requests = []
    skipped = 0
    input_tokens = 0
    for tbl in tables:
        try:
            data = tbl.get("data", None)
            agent = table_cleanup.TableCleanup(table_data=data, context=base_req_id)
        except ValueError:
            skipped += 1
            continue
        
        system_prompt = agent.system_instruction
        user_prompt = agent.instruction
        output_schema = agent.output_schema
        req = {
            "custom_id": f"req_{base_req_id}_t{tbl['table']+1}_table",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": os.getenv("TABLES_CLEANUP_MODEL", "gpt-4.1-mini"),
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
        input_tokens += count_tokens(text=system_prompt + user_prompt, model="gpt-4")


    return all_requests, skipped, input_tokens


def prepare_page_image_requests(images: list[dict], base_req_id: str):
    all_requests = []
    skipped = 0
    input_tokens = 0
    for img in images:
        try:
            agent = image_cleanup.ImageCleanup(image_data=img, context=base_req_id)
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
                "model": os.getenv("IMAGES_CLEANUP_MODEL", "gpt-4o-mini"),
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
        input_tokens += count_tokens(text=system_prompt + user_prompt, model="gpt-4o")

    return all_requests, skipped, input_tokens


def prepare_full_batches_for_cleanup(input_folder: str, output_folder: str):
    texts_batch_requests = []
    tables_batch_requests = []
    images_batch_requests = []
    
    texts_input_tokens = 0
    tables_input_tokens = 0
    images_input_tokens = 0

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
            texts_got = 0
            images_got = 0
            tables_got = 0
            texts_skipped = 0  # Texts are not skipped, so this will be 0
            tables_skipped = 0
            images_skipped = 0
            file_texts_tokens = 0
            file_tables_tokens = 0
            file_images_tokens = 0
            file_processing_start_time = time.time()

            with open(file_path, "r", encoding="utf-8") as file:
                pages = json.load(file)
                for np, p in enumerate(pages):
                    page_id = f"{file_id}_p{np+1}"
                    if p["plain_text"]:
                        request, input_tokens = (prepare_page_text_request(p["plain_text"], page_id))
                        if request:
                            texts_batch_requests.append(request)
                            texts_got += 1
                            texts_input_tokens += input_tokens
                            file_texts_tokens += input_tokens
                        else:
                            texts_skipped += 1
                    if p["tables"]:
                        requests, skipped, input_tokens = prepare_page_table_requests(p["tables"], page_id)
                        tables_batch_requests.extend(requests)
                        tables_skipped += skipped
                        tables_got += len(p["tables"])
                        tables_input_tokens += input_tokens
                        file_tables_tokens += input_tokens
                    if p["images"]:
                        requests, skipped, input_tokens = prepare_page_image_requests(p["images"], page_id)
                        images_batch_requests.extend(requests)
                        images_skipped += skipped
                        images_got += len(p["images"])
                        images_input_tokens += input_tokens
                        file_images_tokens += input_tokens
            file_processing_end_time = time.time()
            file_processing_time = file_processing_end_time - file_processing_start_time
            
            # ----------------------- Start of new table printing logic -----------------------------
            print(f"[{nf+1}/{fl}]: {f} processed in {file_processing_time:.3f}s")
            
            header = f"| {'Type':<8} | {'Got':>10} | {'Processed':>10} | {'Skipped':>10} | {'Input Tokens':>15} |"
            separator = "-" * len(header)
            
            print(separator)
            print(header)
            print(separator)
            
            texts_processed = texts_got - texts_skipped
            tables_processed = tables_got - tables_skipped
            images_processed = images_got - images_skipped
            
            # Data rows
            print(f"| {'Texts':<8} | {texts_got:>10} | {texts_processed:>10} | {texts_skipped:>10} | {file_texts_tokens:>15} |")
            print(f"| {'Tables':<8} | {tables_got:>10} | {tables_processed:>10} | {tables_skipped:>10} | {file_tables_tokens:>15} |")
            print(f"| {'Images':<8} | {images_got:>10} | {images_processed:>10} | {images_skipped:>10} | {file_images_tokens:>15} |")

            print(separator)
            
            # Totals row
            total_got = texts_got + tables_got + images_got
            total_skipped = texts_skipped + tables_skipped + images_skipped
            total_processed = total_got - total_skipped
            total_tokens = file_texts_tokens + file_tables_tokens + file_images_tokens
            print(f"| {'Total':<8} | {total_got:>10} | {total_processed:>10} | {total_skipped:>10} | {total_tokens:>15} |")
            
            print(separator)
            print() # for a blank line after the table
            # ------------------------- End of new table printing logic ----------------------------

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
    nb_texts_requests = len(texts_batch_requests)
    nb_tables_requests = len(tables_batch_requests)
    nb_images_requests = len(images_batch_requests)
    # print summary
    print("="*100, "Processing completed.", "="*100, sep="\n")
    print("Statistics:")
    print(f"\tTotal files processed: {len(json_files)}")
    print(f"\tTotal processing time: {global_processing_time:.3f}s")
    print(f"\tTexts: {nb_texts_requests} requests | {texts_input_tokens} input tokens")
    print(f"\tTables: {nb_tables_requests} requests | {tables_input_tokens} input tokens")
    print(f"\tImages: {nb_images_requests} requests | {images_input_tokens} text input tokens")
    print(
        f"\tTotal: {nb_texts_requests + nb_tables_requests + nb_images_requests} requests | ",
        f"{texts_input_tokens + tables_input_tokens + images_input_tokens} input tokens",
        sep=""
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
