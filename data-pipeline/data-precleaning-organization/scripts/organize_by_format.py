import os
import shutil
from pathlib import Path

# === Paths ===
SOURCE_DIR = "/home/estin/rag_data_pipeline/data_renamed/1CP"
DEST_DIR = "/home/estin/rag_data_pipeline/data_organized"

# === Extension-to-folder mapping ===
format_map = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".png": "images/png",
    ".jpg": "images/jpg",
    ".jpeg": "images/jpg",
    ".py": "code/py",
    ".c": "code/c",
    ".asm": "code/asm",
    ".pptx": "pptx",
    ".xlsx": "excel",
    ".csv": "csv",
    ".zip": "archives/zip",
    ".rar": "archives/rar"
}

copied_files = []
others_count = 0

# === Walk through source folder ===
for root, _, files in os.walk(SOURCE_DIR):
    for file in files:
        if "index.csv" in file or "/ignored/" in root:
            continue

        ext = Path(file).suffix.lower()
        subfolder = format_map.get(ext, "others")

        src_path = os.path.join(root, file)
        dest_folder = os.path.join(DEST_DIR, subfolder)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, file)

        # Prevent overwriting
        counter = 1
        while os.path.exists(dest_path):
            file_stem = Path(file).stem
            dest_path = os.path.join(dest_folder, f"{file_stem}_{counter}{ext}")
            counter += 1

        shutil.copy2(src_path, dest_path)
        copied_files.append(dest_path)

print(f"Total copied files: {len(copied_files)}")
print(f"Files in 'others': {sum('others' in f for f in copied_files)}")
