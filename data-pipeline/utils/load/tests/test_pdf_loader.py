"""
Extract content from all PDF files as JSON files into a folder.

Usage: test_pdf_loader.py <input_folder> <output_folder>

Positional arguments:
  input_folder   Path to the folder containing PDF files
  output_folder  Path to the folder to save JSON files

Options:
  -h, --help     show this help message and exit
  
NOTE:
You should run this script as a module from the root directory of the repository as follows:
python -m data_pipeline.utils.load.tests.test_pdf_loader <input_folder> <output_folder>
"""

from ..pdf_loader import PDFLoader
import os
import sys
import json
import time
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract content from all PDF files in a folder."
    )
    parser.add_argument("input_folder", help="Path to the folder containing PDF files")
    parser.add_argument("output_folder", help="Path to the folder to save JSON files")
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder not found at {args.input_folder}")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    total_start_time = time.time()
    pdf_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(".pdf")]
    nb_pdfs = len(pdf_files)
    if not pdf_files:
        print(f"No PDF files found in {args.input_folder}")
        return

    for n, filename in enumerate(pdf_files):
        file_path = os.path.join(args.input_folder, filename)
        print(f"[{n}/{nb_pdfs}]: Processing {filename}...")
        start_time = time.time()

        pdf_loader = PDFLoader(file_path)
        result = pdf_loader.analyse()

        # Generate output filename
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_file = os.path.join(args.output_folder, output_filename)

        # Save as json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        end_time = time.time()
        print(f"Finished in: {end_time - start_time:.2f} seconds")

    total_end_time = time.time()
    print(
        f"Total execution time for {len(pdf_files)} files: {total_end_time - total_start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
