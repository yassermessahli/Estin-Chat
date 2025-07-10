from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType, bulk_import, get_import_progress
from huggingface_hub import InferenceClient
from ..setup.schema import schema

import json
import re
import os
import dotenv


# Load environment variables

dotenv.load_dotenv("../.env")

EMBEDDING_ENDPOINT = os.getenv("TEI_ENDPOINT", None)
EMBEDDING_API_KEY = os.getenv("TEI_API_KEY", None)

MINIO_ENDPOINT = os.getenv("MINIO_HOST", None)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", None)
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", None)
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", None)

MILVUS_ENDPOINT = os.getenv("MILVUS_HOST", None)
DATABASE_NAME = os.getenv("DATABASE_NAME", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", None)


# Instantiate the tei inference client
tei_inference_client = InferenceClient(
    base_url=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
)

def prepare_bulk_writer():
    """Prepare a RemoteBulkWriter instance for MinIO data import."""
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET_NAME,
        secure=False
    )

    writer = RemoteBulkWriter(
        schema=schema,
        remote_path="/",
        connect_param=conn,
        file_type=BulkFileType.PARQUET,
    )

    print('bulk writer created...')
    return writer

def import_all_bulk_data(batch_files: list):
    """
    Perform the bulk import of data into Milvus.
    Args:
        batch_files (list): List of parquet files to be imported.
    """
    if batch_files:
        return bulk_import(
            collection_name=COLLECTION_NAME,
            db_name=DATABASE_NAME,
            url=MILVUS_ENDPOINT,
            files=batch_files
        )
    return None


def check_import_status(job_id: str):
    """
    Check the status of a bulk import job in Milvus.
    Args:
        job_id (str): The ID of the import job to check.
    """
    
    resp = get_import_progress(
        url=MILVUS_ENDPOINT,
        job_id=job_id,
    )
    return json.dumps(resp.json(), indent=4)


def get_records_from_jsonl(file_path: dict):
    """
    Extract records from a JSONL file (Output file from OpenAI Batch API).
    Args:
        file_path (str): Path to the JSONL file.
    Returns:
        list: A list of records (Dictionaries) to be imported into Milvus.
    """
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        
        lines = file.readlines()
        print(f" Found {len(lines)} lines")
        progress = 0
        for i, line in enumerate(lines):
            if i % (len(lines)//100) == 0:
                progress += 1
                print(f"Progress: {progress}%")
            try:
                line = json.loads(line.strip())
                request_id = line.get("custom_id", "")
                record = parse_metadata(request_id)
                
                chunks = json.loads(line["response"]["body"]["choices"][0]["message"]["content"])["paragraphs"]
                for chunk in chunks:
                    chunk_content = chunk["content"].strip()
                    
                    # Split chunk if it exceeds 300 characters
                    if len(chunk_content) <= 300:
                        record["chunk"] = chunk_content
                        record["vector"] = tei_inference_client.feature_extraction(chunk_content)[0]
                        records.append(record.copy())
                    else:
                        # Split into smaller chunks of max 300 characters
                        for i in range(0, len(chunk_content), 300):
                            tiny_chunk = chunk_content[i:i+300]
                            record["chunk"] = tiny_chunk
                            record["vector"] = tei_inference_client.feature_extraction(tiny_chunk)[0]
                            records.append(record.copy())
            except json.JSONDecodeError:
                continue

    return records


def parse_metadata(id: str):
    """
    Parse the metadata from a given ID string.
    Args:
        id (str): The ID string to parse.
    Returns:
        dict: A dictionary containing parsed metadata.
    This function extracts various components from the ID string, such as:
        - year_of_study
        - level (e.g., 1CP, 2CP)
        - data_type (e.g., image, text, table)
        - semester (e.g., S1, S2)
        - page (e.g., p1, p2)
        - document_type (e.g., COURS, TD, TP, EXAM, INTERRO, OTHER)
        - subject_code (the first part of the ID)
        - title (the rest of the ID after subject_code)
    """

    # get all parts
    parts = id.split("_")
    
    # Patterns and lookups
    level_set = {"1CP", "2CP", "1CS", "2CS", "3CS"}
    data_type_set = {"image", "text", "table"}
    semester_set = {"S1", "S2"}
    doc_type_set = {"COURS", "TD", "TP", "EXAM", "INTERRO", "OTHER"}
    year_pattern = re.compile(r"^(19|20)\d{2}$")
    page_pattern = re.compile(r"^p\d+$")
    others_pattern = re.compile(r"^(req|i\d+|t\d+)$", re.IGNORECASE)

    meta = {
        # with default values
        "year_of_study": 2019,
        "level": "",
        "data_type": "",
        "semester": "",
        "page": -1,
        "document_type": "",
        "subject_code": "",
        "title": "",
    }

    filtered = []
    for part in parts:
        if others_pattern.match(part):
            continue
        if meta["year_of_study"] == 2019 and year_pattern.match(part):
            meta["year_of_study"] = int(part)
            continue
        if meta["level"] == "" and part in level_set:
            meta["level"] = part
            continue
        if meta["data_type"] == "" and part in data_type_set:
            meta["data_type"] = part
            continue
        if meta["semester"] == "" and part in semester_set:
            meta["semester"] = part
            continue
        if meta["page"] == -1 and page_pattern.match(part):
            meta["page"] = int(part.removeprefix("p"))
            continue
        if meta["document_type"] == "" and part in doc_type_set:
            meta["document_type"] = part
            continue
        filtered.append(part)

    # Assign subject_code and title
    if filtered:
        meta["subject_code"] = filtered[0]
        if len(filtered) > 1:
            meta["title"] = "_".join(filtered[1:])
        else:
            meta["title"] = ""
    else:
        meta["subject_code"] = ""
        meta["title"] = ""

    return meta
