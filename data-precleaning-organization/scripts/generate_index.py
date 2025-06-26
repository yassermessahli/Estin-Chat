import os
import csv
import datetime

# === Config ===
BASE_DIR = "/home/estin/rag_data_pipeline/data_raw/"
OUTPUT_FILE ="/home/estin/rag_data_pipeline/index.csv"


# === Predefined columns ===
columns = [
    "file_name",         # name only
    "path",              # full original path
    "size_kb",           # size in KB
    "type",              # file extension
    "last_modified",     # last modified date
    "status",            # "included" or "ignored"
    "level",             # 1CP, 2CP, etc.
    "module",            # ALG1, BW...
    "doc_type",          # TD, TP, COURS, etc.
    "year",              # extracted year
    "renamed_path",

    "organized_path",    # where it was moved in data_organized
    "cleaned",           # yes / no
    "chunked",           # yes / no
    "embedded",          # yes / no
    "notes"              # explanation if ignored
]

index_data = []

# === Walk all files ===
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file == "index.csv":
            continue  # skip index file itself

        file_path = os.path.join(root, file)
        file_name = file
        file_type = os.path.splitext(file)[1][1:].lower()
        size_kb = round(os.path.getsize(file_path) / 1024, 2)
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

        index_data.append({
            "file_name": file_name,
            "path": file_path,
            "size_kb": size_kb,
            "type": file_type,
            "last_modified": last_modified,
            "status": "included",       # default is included
            "level":"",
            "module": "",
            "doc_type": "",
            "year": "",
            "renamed_path": "",
            "organized_path": "",
            "cleaned": "no",
            "chunked": "no",
            "embedded": "no",
            "notes": ""
        })

# === Write index.csv ===
with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(index_data)

print(f"✅ Index generated with {len(index_data)} files → {OUTPUT_FILE}")
