import os
import re
import csv
import unicodedata
import shutil
from pathlib import Path

# === Configuration ===
CLEAN_DIR = "/home/estin/rag_data_pipeline/data_clean"
RENAMED_DIR = "/home/estin/rag_data_pipeline/data_renamed"
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
    # 2CP – S1
    "Algebra 3": "ALG3",
    "Analyse Mathématique 3": "ANA3",
    "Architecture des Ordinateurs 2": "ARCHI2",
    "Économie": "ECON",
    "Electronique Fondamentale 2": "ELECTRO2",
    "Probabilités et Statistiques 1": "PROBA1",
    "Structure Fichiers et Structure de Données": "SFSD",

    # 2CP – S2
    "Analyse Mathématique 4": "ANA4",
    "Introduction aux systèmes d'information": "SI",
    "Logique Mathématique": "LOG",
    "Optique et Ondes électromagnétiques": "OOE",
    "Probabilités et Statistiques 2": "PROBA2",
    "Programmation Orientée Objet (POO, OOP)": "POO",
    "Projet Pluridisciplinaire": "PRJP",
    # 1CS – S1
    "ANG": "ANG",
    "BDD": "BDD",
    "GL": "GL",
    "PAFA": "PAFA",
    "RO1": "RO1",
    "RX1": "RX1",
    "SE": "SE",
    "ThL": "ThL",

    # 1CS – S2
    "ADCI": "ADCI",
    "ANUM": "ANUM",
    "Entreprenariat": "ENT",
    "IA": "IA",
    "MF": "MF",
    "RO2": "RO2",
    "RX2": "RX2",
    "SEC": "SEC",
    # 2CS – S1
    "ANAD": "ANAD",
    "BDDA": "BDDA",
    "Cloud": "CLOUD",
    "Complexité": "CMPLX",
    "DS": "DS",
    "EN": "EN",
    "GL": "GL",
    "Projet": "PRJT",

    # 2CS – S2 (CS)
    "Administration Systèmes et réseaux": "ADSR",
    "Audit de la sécurité des système d_information": "AUDSEC",
    "Biométrie": "BIOM",
    "Cryptographie avancée": "CRYPTO",
    "Méthodes formelles pour la sécurité": "MF",
    "ML": "ML",
    "Sécurités des réseaux": "SR",
    "Sécurités des systèmes d'exploitation": "SECSE",

    # 2CS – S2 (IA)
    "BDD": "BDD",
    "BIGDATA": "BIGDATA",
    "BTI": "BTI",
    "IGC": "IGC",
    "ML": "ML",
    "SA": "SA",
    "SC": "SC",
    "TNO": "TNO",
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

def extract_specialty(path_parts):
    """Extract specialty from semester notation like S2(AI), S2(CS)"""
    for part in path_parts:
        # Look for patterns like S1(AI), S2(CS), etc.
        match = re.search(r"s[12]\(([A-Z]+)\)", part.lower())
        if match:
            return match.group(1).upper()
    return None

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

    level = next((p for p in path_parts if re.fullmatch(r"\d+(cp|cs)", p.lower())), "UNKNOWN")
    semester = next((p for p in path_parts if re.fullmatch(r"s[12]", p.lower())), "S?")
    specialty = extract_specialty(path_parts)
    module_abbr = fuzzy_find_module(path_parts)
    doc_type = normalize_type("/".join(path_parts))
    year = extract_year("/".join(path_parts))
    cleaned_title = clean_filename(title)

    # Build filename with specialty if present
    if specialty:
        level_semester = f"{level}_{semester}({specialty})"
    else:
        level_semester = f"{level}_{semester}"

    if year:
        new_filename = f"{level_semester}_{module_abbr}_{doc_type}_{year}_{cleaned_title}{ext}"
    else:
        new_filename = f"{level_semester}_{module_abbr}_{doc_type}_{cleaned_title}{ext}"

    # Rebuild relative structure
    relative_path = os.path.relpath(cleaned_path, CLEAN_DIR)
    renamed_folder = os.path.join(RENAMED_DIR, os.path.dirname(relative_path))
    os.makedirs(renamed_folder, exist_ok=True)

    renamed_path = os.path.join(renamed_folder, new_filename)
    shutil.copy2(cleaned_path, renamed_path)

    # Update index fields
    row["file_name"] = new_filename
    row["renamed_path"] = renamed_path
    row["level"] = level
    row["semester"] = semester
    row["specialty"] = specialty if specialty else ""
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

print(f"✅ {renamed_count} files copied into data_renamed with structured names and index updated.")
