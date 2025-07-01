import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any

# Add utils to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(os.path.dirname(current_dir), "utils")
sys.path.insert(0, utils_dir)

# import the components
from load.pdf_loader import PDFLoader
from split.hierarchical_splitter import HierarchicalSplitter
from transform.image_cleanup import ImageCleanup
from transform.model import Model, ModelParams
from transform.table_cleanup import TableCleanup
from transform.text_cleanup import TextCleanup


class DataPipelineOrchestrator:
    """
    Main orchestrator that coordinates the complete data pipeline:
    Load (PDF) , Transform (Clean) , Split (Chunk) , Output
    """
    
    def __init__(self, 
                 text_model: str = "qwen3:8b",           
                 table_model: str = "qwen3:8b",           
                 image_model: str = "llama3.2-vision:11b", 
                 input_folder: str = None, 
                 output_folder: str = None):
        
        # Initialize specialized models
        self.text_model_params = ModelParams(
            model=text_model,
            temperature=0.3, 
            num_ctx=4096
        )
        self.text_model = Model(self.text_model_params)
        
        self.table_model_params = ModelParams(
            model=table_model,
            temperature=0.2,  
            num_ctx=2048
        )
        self.table_model = Model(self.table_model_params)
        
        self.image_model_params = ModelParams(
            model=image_model,
            temperature=0.4,  
            num_ctx=2048,
            think=False
        )
        self.image_model = Model(self.image_model_params)
        
        # Initialize components
        self.splitter = HierarchicalSplitter()
        
        # Setup folders
        self.input_folder = input_folder or "/home/melissa-ghemari/estin-chatbot/data-pipeline/sample-data"
        self.output_folder = output_folder or "outputs"
        self._ensure_output_folder()
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_time": 0,
            "models_used": {
                "text": text_model,
                "table": table_model,
                "image": image_model
            },
            "errors": []
        }
    
    def _ensure_output_folder(self):
        """Create output folder structure if it doesn't exist"""
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_folder, "raw_extracted")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_folder, "cleaned")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_folder, "chunked")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_folder, "final")).mkdir(exist_ok=True)
    
    def _load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Load and extract content from PDF"""
        print(f"Loading PDF: {os.path.basename(pdf_path)}")
        loader = PDFLoader(pdf_path)
        return loader.analyse()
    
    def _transform_content(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        """Transform and clean extracted content"""
        print("Transforming content...")

        transformed_pages = []
        
        for page_data in pages_data:
            page_num = page_data["page"]
            print(f"Processing page {page_num}...")
            
            transformed_page = {
                "page": page_num,
                "cleaned_text": "",
                "cleaned_tables": [],
                "cleaned_images": []
            }
            
            # Clean text with specialized text model
            if page_data["plain_text"]:
                try:
                    text_cleaner = TextCleanup(page_data["plain_text"], self.text_model)
                    result = text_cleaner.process()
                    transformed_page["cleaned_text"] = result.message.content
                    print(f"✓ Text cleaned with {self.text_model_params.model} ({len(transformed_page['cleaned_text'])} chars)")
                except Exception as e:
                    print(f" ⚠️ Text cleaning failed: {e}")
                    transformed_page["cleaned_text"] = page_data["plain_text"]  # Fallback to original
            
            # Clean tables with specialized table model
            if page_data["tables"]:
                try:
                    for i, table in enumerate(page_data["tables"]):
                        # TableCleanup needs: table_data, context, model
                        table_cleaner = TableCleanup(
                            table_data=table.get("data", table),  # The actual table data
                            context=page_data.get("plain_text", ""),  # Page text as context
                            model=self.table_model  # The model instance
                        )
                        cleaned_table = table_cleaner.process()
                        transformed_page["cleaned_tables"].append({
                            "table_id": table.get("table", i + 1),
                            "cleaned_data": cleaned_table.message.content
                        })
                    print(f"✓ {len(page_data['tables'])} tables cleaned with {self.table_model_params.model}")
                except Exception as e:
                    print(f" ⚠️ Table cleaning failed: {e}")
                    # Fallback with safe structure
                    for i, table in enumerate(page_data["tables"]):
                        transformed_page["cleaned_tables"].append({
                            "table_id": table.get("table", i + 1),
                            "cleaned_data": str(table.get("data", "Table data not available"))
                        })
            
            # Clean images with specialized vision model
            if page_data["images"]:
                try:
                    for image in page_data["images"]:
                        # ImageCleanup needs: image_data, context, model
                        image_cleaner = ImageCleanup(
                            image_data=image,  # The complete image data (contains ext, base64, etc.)
                            context=page_data.get("plain_text", ""),  # Page text as context
                            model=self.image_model  # The model instance
                        )
                        cleaned_image = image_cleaner.process()
                        transformed_page["cleaned_images"].append({
                            "image_id": image["image_id"],
                            "description": cleaned_image.message.content,
                            "original_ext": image["ext"]
                        })
                    print(f"✓ {len(page_data['images'])} images described with {self.image_model_params.model}")
                except Exception as e:
                    print(f" ⚠️ Image processing failed: {e}")
                    # Fallback without base64 data
                    for image in page_data["images"]:
                        transformed_page["cleaned_images"].append({
                            "image_id": image["image_id"],
                            "description": f"Image {image['image_id']} (processing failed)",
                            "original_ext": image.get("ext", "unknown")
                        })
            
            transformed_pages.append(transformed_page)
    
        return transformed_pages

    def _save_outputs(self, filename: str, raw_data: List[Dict], cleaned_data: List[Dict], chunks: List[Dict]):
        """Save outputs to different folders"""
        base_name = os.path.splitext(filename)[0]
        
        # Save raw extracted data
        raw_path = os.path.join(self.output_folder, "raw_extracted", f"{base_name}_raw.json")
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        # Save cleaned data
        cleaned_path = os.path.join(self.output_folder, "cleaned", f"{base_name}_cleaned.json")
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Save chunked data
        chunked_path = os.path.join(self.output_folder, "chunked", f"{base_name}_chunks.json")
        with open(chunked_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Save final enriched data
        final_data = {
            "source_file": filename,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": len(cleaned_data),
            "total_chunks": len(chunks),
            "chunk_types": {
                "text": len([c for c in chunks if c["type"] == "text"]),
                "table": len([c for c in chunks if c["type"] == "table"]),
                "image": len([c for c in chunks if c["type"] == "image"])
            },
            "chunks": chunks
        }
        
        final_path = os.path.join(self.output_folder, "final", f"{base_name}_final.json")
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"Outputs saved to: {self.output_folder}")

    def process_file(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file through the complete pipeline"""
        filename = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Step 1: Load
            raw_data = self._load_pdf(pdf_path)
            self.stats["total_pages"] += len(raw_data)
            
            # Step 2: Transform
            cleaned_data = self._transform_content(raw_data)
            
            # Step 3: Split (now with filename metadata)
            chunks = self._split_content(cleaned_data, filename)
            self.stats["total_chunks"] += len(chunks)
            
            # Step 4: Save outputs
            self._save_outputs(filename, raw_data, cleaned_data, chunks)
            
            processing_time = time.time() - start_time
            self.stats["total_time"] += processing_time
            self.stats["files_processed"] += 1
            
            result = {
                "success": True,
                "filename": filename,
                "pages": len(raw_data),
                "chunks": len(chunks),
                "processing_time": processing_time
            }
            
            print(f"✅ Completed: {filename}")
            print(f" {len(raw_data)} pages → {len(chunks)} chunks")
            print(f" Processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            self.stats["errors"].append(error_msg)
            return {
                "success": False,
                "filename": filename,
                "error": str(e)
            }
    
    
    def process_folder(self) -> Dict[str, Any]:
        """Process all PDF files in the input folder"""
        print(f"\nScanning folder: {self.input_folder}")
        
        # Find all PDF files
        pdf_files = []
        for file in os.listdir(self.input_folder):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.input_folder, file))
        
        if not pdf_files:
            print("⚠️  No PDF files found in the input folder!")
            return {"success": False, "message": "No PDF files found"}
        
        print(f"Found {len(pdf_files)} PDF files")
        print(f"Models used:")
        print(f"Text: {self.text_model_params.model}")
        print(f"Tables: {self.table_model_params.model}")
        print(f"Images: {self.image_model_params.model}")
        print(f"Output folder: {self.output_folder}")
        
        overall_start = time.time()
        results = []
        
        # Process each file
        for pdf_path in pdf_files:
            result = self.process_file(pdf_path)
            results.append(result)
        
        # Final summary
        total_time = time.time() - overall_start
        self._print_final_summary(results, total_time)
        
        return {
            "success": True,
            "total_files": len(pdf_files),
            "successful": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "results": results,
            "stats": self.stats
        }
    
    def _print_final_summary(self, results: List[Dict], total_time: float):
        """Print final processing summary"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"\n{'='*60}")
        print(f"PIPELINE PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"Total pages processed: {self.stats['total_pages']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per file: {total_time/len(results):.2f}s")
        
        if successful:
            avg_chunks = sum(r["chunks"] for r in successful) / len(successful)
            print(f"Average chunks per file: {avg_chunks:.1f}")
        
        if failed:
            print(f"\nFailed files:")
            for result in failed:
                print(f"   - {result['filename']}: {result['error']}")
        
        print(f"\nOutput saved to: {self.output_folder}")
        print(f"{'='*60}")


def main():
    """Main function to run the complete pipeline with specialized models"""
    orchestrator = DataPipelineOrchestrator(
        text_model="qwen3:8b",                    
        table_model="qwen3:8b",                   
        image_model="llama3.2-vision:11b",       
        input_folder="/home/melissa-ghemari/estin-chatbot/data-pipeline/sample-data",
        output_folder="specialized_pipeline_outputs"
    )
    
    # Run the complete pipeline
    results = orchestrator.process_folder()
    
    return results


if __name__ == "__main__":
    main()