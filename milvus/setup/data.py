import json
import re
import os
import dotenv
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType, bulk_import, get_import_progress
from schema import schema

dotenv.load_dotenv("../.env")

def prepare_bulk_writer():
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=os.getenv("MINIO_HOST"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket_name=os.getenv("MINIO_BUCKET_NAME"),
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

def import_all_bulk_data(writer: RemoteBulkWriter, batch_files: list):
    if batch_files:
        return bulk_import(
            collection_name="estin_docs",
            db_name="core_db",
            url=os.getenv("MILVUS_HOST"),
            files=batch_files
        )
    return None


def check_import_status(job_id: str):
    resp = get_import_progress(
        url=os.getenv("MILVUS_HOST"),
        job_id=job_id,
    )

    print(json.dumps(resp.json(), indent=4))

def prepare_records_from_jsonl(file_path: dict):
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        print(f" Found {len(lines)} lines in file")
        for i, line in enumerate(lines):
            if i % (len(lines)//10) == 0:
                print(f"  [{i}/{len(lines)}]")
            try:
                line = json.loads(line.strip())
                
                request_id = line.get("custom_id", "")
                record = parse_metadata(request_id)
                
                chunks = json.loads(line["response"]["body"]["choices"][0]["message"]["content"])["paragraphs"]
                for chunk in chunks:
                    chunk_content = chunk["content"].strip()
                    record["chunk"] = chunk_content
                    records.append(record.copy())
            except json.JSONDecodeError:
                continue

    return records


def parse_metadata(id: str):
    
    # get all parts
    parts = id.split("_")
    
    # Patterns and lookups
    level_set = {"1CP", "2CP", "1CS", "2CS", "3CS"}
    data_type_set = {"image", "text", "table"}
    semester_set = {"S1", "S2"}
    doc_type_set = {"COURS", "TD", "TP", "EXAM", "INTERRO", "OTHER"}
    year_pattern = re.compile(r"^(19|20)\d{2}$")
    page_pattern = re.compile(r"^p\d+$")
    useless_pattern = re.compile(r"^(req|i\d+|t\d+)$", re.IGNORECASE)

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
        if useless_pattern.match(part):
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

def process(input_folder: str, output_folder: str):

    writer = prepare_bulk_writer()

    records = []
    all_files = os.listdir(input_folder)
    
    for i, file_name in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}]: Processing file {file_name}...")
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(input_folder, file_name)
            file_records = prepare_records_from_jsonl(file_path)
            records.extend(file_records)
            for record in file_records:
                writer.append_row(record)
                writer.commit()
    
    batch_files = writer.batch_files

    print(f"Batch files created:")
    for batch_file in batch_files:
        print(f"- {batch_file}")

    resp = import_all_bulk_data(writer, batch_files)
    if resp:
        print(f"Import response")
        print(json.dumps(resp.json(), indent=4))
        
        print("checking import status...")
        check_import_status(resp.json()['data']['jobId'])
        
    else:
        print("No files to import.")
    
    
    # save in JSON format
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "1CP_2CP_Records.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"rows": records}, f, ensure_ascii=False, indent=4)
        print(f"Source data saved to {output_file}")
        


# Main execution
input_folder = "/home/estin/batches/output_files"
save_path = "/home/estin/batches/source_data_for_import"
process(input_folder, save_path)
