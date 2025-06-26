import os
import re
import csv
import unicodedata
import shutil
from pathlib import Path

# === Configuration ===
CLEAN_DIR = "/home/estin/rag_data_pipeline/data_clean/1CP"
RENAMED_DIR = "/home/estin/rag_data_pipeline/data_renamed/1CP"
INDEX_FILE = "/home/estin/rag_data_pipeline/index.csv"

module_map = {
    "Algèbre 1": "ALG1",
    "Algorithmique et Structures de Données 1": "ASDS1",
    "Analyse mathématique 1": "ANA1",
    "Architecture des ordinateurs 1": "ARCHI1",
    "Bureautique et Web": "BW",
    "Electricité 1": "ELEC1",
    "English 01": "ENG1",
    "Introduction au système d'exploitation": "SE",
    "Algèbre 2": "ALG2",
    "Algorithmique et structures de données 2": "ASDS2",
    "Analyse mathématique 2": "ANA2",
    "Electronique fondamentale 1": "ELECTRO1",
    "English 2": "ENG2",
    "Mécanique du point": "MEC",
    "Techniques d'expression": "TECX",
}

# === Helpers ===

def normalize_text(text):
    """Remove accents and symbols"""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode()
    return re.sub(r"[^\w]", "", text).lower()

def fuzzy_find_module(path_parts):
    """Search module name from the deepest folder upwards"""
    normalized_map = {normalize_text(k): v for k, v in module_map.items()}
    for part in reversed(path_parts):
        part_norm = normalize_text(part)
        for mod_norm, abbr in normalized_map.items():
            if mod_norm == part_norm:
                return abbr
            if mod_norm in part_norm or part_norm in mod_norm:
                return abbr
    return "UNKNOWNMOD"

def normalize_type(path):
    path = path.lower()
    if any(k in path for k in ["cours", "course"]):
        return "COURS"
    elif any(k in path for k in ["td", "tds"]):
        return "TD"
    elif any(k in path for k in ["tp", "lab"]):
        return "TP"
    elif any(k in path for k in ["exam", "examen"]):
        return "EXAM"
    elif any(k in path for k in ["interro", "test", "controle", "eval"]):
        return "INTERRO"
    return "OTHER"

def extract_year(path):
    match = re.search(r"20\d{2}", path)
    return match.group(0) if match else None

def clean_filename(name):
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^\w\-]", "", name)
    return name.strip("-_")
    
# === Load existing index
with open(INDEX_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    index_rows = list(reader)

updated_rows = []
renamed_count = 0

# === Process only files with cleaned_path
for row in index_rows:
    cleaned_path = row.get("cleaned_path", "")
    if row["status"] not in ["included", "renamed"] or not cleaned_path or not os.path.exists(cleaned_path):
        updated_rows.append(row)
        continue

    ext = os.path.splitext(cleaned_path)[-1]
    title = os.path.splitext(os.path.basename(cleaned_path))[0]
    path_parts = Path(cleaned_path).parts

    level = next((p for p in path_parts if "1cp" in p.lower()), "1CP")
    semester = next((p for p in path_parts if re.fullmatch(r"s[12]", p.lower())), "S?")
    module_abbr = fuzzy_find_module(path_parts)
    doc_type = normalize_type("/".join(path_parts))
    year = extract_year("/".join(path_parts))
    cleaned_title = clean_filename(title)

    if year:
        new_filename = f"{level}_{semester}_{module_abbr}_{doc_type}_{year}_{cleaned_title}{ext}"
    else:
        new_filename = f"{level}_{semester}_{module_abbr}_{doc_type}_{cleaned_title}{ext}"

    # Rebuild relative structure
    relative_path = os.path.relpath(cleaned_path, CLEAN_DIR)
    renamed_folder = os.path.join(RENAMED_DIR, os.path.dirname(relative_path))
    os.makedirs(renamed_folder, exist_ok=True)

    renamed_path = os.path.join(renamed_folder, new_filename)
    shutil.copy2(cleaned_path, renamed_path)

    # Update index fields
    row["file_name"] = new_filename
    row["renamed_path"] = renamed_path
    row["module"] = module_abbr
    row["doc_type"] = doc_type
    row["year"] = year
    row["status"] = "renamed"

    renamed_count += 1
    updated_rows.append(row)

# === Write updated index
with open(INDEX_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"✅ {renamed_count} files copied into data_renamed/1CP with structured names and index updated.")
