import os
import sys
import json
import time
import argparse


# Find the pdf_loader.py file relative to this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from pdf_loader import PDFLoader


def main():
    parser = argparse.ArgumentParser(description="Extract content from PDF file")
    parser.add_argument("file_path", help="Path to the PDF file to process")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: file not found")
        sys.exit(1)

    start_time = time.time()

    pdf_loader = PDFLoader(args.file_path)
    result = pdf_loader.analyse()

    # Create results folder if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(results_dir, exist_ok=True)

    # Generate output filename
    input_filename = os.path.basename(args.file_path)
    output_filename = os.path.splitext(input_filename)[0] + "_extracted.json"
    output_file = os.path.join(results_dir, output_filename)

    # Save as json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Executed in: {end_time - start_time:.5f} seconds")
    print(f"Results saved to: `outputs/` folder")


if __name__ == "__main__":
    main()
