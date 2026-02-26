import csv
import os
import shutil


BASE_DIR = "/home/estin/rag_data_pipeline/data_raw/1CP"
LOG_FILE = os.path.join(BASE_DIR, "ignored_files.csv")

restored = []

# Read the log file
with open(LOG_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_name = row["file_name"]
        original_path = row["original_path"]
        moved_to = row["moved_to"]

        # Ensure original directory exists
        os.makedirs(os.path.dirname(original_path), exist_ok=True)

        try:
            shutil.move(moved_to, original_path)
            restored.append(original_path)
        except FileNotFoundError:
            print(f" File not found: {moved_to}")
        except Exception as e:
            print(f" Error restoring {moved_to} → {original_path}: {e}")

print(f" Restored {len(restored)} files to their original paths.")

# Optional: Remove ignored folder if empty
ignored_dir = os.path.join(BASE_DIR, "ignored")
if os.path.isdir(ignored_dir) and not os.listdir(ignored_dir):
    os.rmdir(ignored_dir)
    print(" Cleaned up empty 'ignored/' folder.")
