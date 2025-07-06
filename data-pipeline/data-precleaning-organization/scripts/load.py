import os
import csv

RAW_DATA_DIR = "data_raw"  

os.makedirs(RAW_DATA_DIR, exist_ok=True)

with open("module_drive_links.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        module = row["Module"].strip().replace(" ", "_")
        drive_link = row["Drive Link"].strip()
        save_path = os.path.join(RAW_DATA_DIR, module)

        # Create folder for this module
        os.makedirs(save_path, exist_ok=True)

        print(f"Downloading {module} into {save_path}")
        os.system(f'gdown --folder "{drive_link}" -O "{save_path}"')
