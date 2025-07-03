import os
import shutil
import csv

RAW_DIR = "/home/estin/rag_data_pipeline/data_raw/"
CLEAN_DIR = "/home/estin/rag_data_pipeline/data_clean/"
INDEX_FILE = "/home/estin/rag_data_pipeline/index.csv"

irrelevant_keywords = ['draft', 'old', 'tmp', 'DS_Store', 'backup']
irrelevant_prefixes = ['~$', '.', '__MACOSX']
 
# Load index
with open(INDEX_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

updated_rows = []
copied_count = 0
ignored_count = 0

for row in rows:
    file_path = row["path"]
    file_name = row["file_name"]
    file_lower = file_name.lower()
    reason = None

    if not os.path.exists(file_path):
        row["status"] = "missing"
        row["notes"] = "file not found"
        row["cleaned_path"] = ""
        updated_rows.append(row)
        continue

    # Check for irrelevant
    if os.path.getsize(file_path) == 0:
        reason = "empty"
    elif any(k in file_lower for k in irrelevant_keywords):
        reason = "keyword"
    elif any(file_lower.startswith(p) for p in irrelevant_prefixes):
        reason = "prefix"

    if reason:
        row["status"] = "ignored"
        row["notes"] = reason
        row["cleaned_path"] = ""
        ignored_count += 1

        # Déplacer le fichier ignoré dans data_ignored/
        IGNORED_DIR = "/home/estin/rag_data_pipeline/data_ignored/"
        relative_path = os.path.relpath(file_path, RAW_DIR)
        ignored_path = os.path.join(IGNORED_DIR, relative_path)
        os.makedirs(os.path.dirname(ignored_path), exist_ok=True)
        shutil.move(file_path, ignored_path)
        row["ignored_path"] = ignored_path
        # Supprimer les dossiers vides dans data_ignored/
        for dirpath, dirnames, filenames in os.walk(IGNORED_DIR, topdown=False):
              if not dirnames and not filenames:
              os.rmdir(dirpath)
    else:
        # Build equivalent path in clean dir
        relative_path = os.path.relpath(file_path, RAW_DIR)
        dest_path = os.path.join(CLEAN_DIR, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file_path, dest_path)

        row["status"] = "included"
        row["notes"] = ""
        row["cleaned_path"] = dest_path
        copied_count += 1

    updated_rows.append(row)

# Save updated index
with open(INDEX_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"✅ {copied_count} relevant files copied to data_clean with full structure.")
print(f"🚫 {ignored_count} files ignored and tracked in index.csv.")
