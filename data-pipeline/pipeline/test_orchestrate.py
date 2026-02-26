import os
import sys
import time

# Add the current directory to sys.path to import orchestrate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrate import DataPipelineOrchestrator


def test_orchestration():
    """Test the complete data pipeline orchestration"""
    print("Testing Data Pipeline Orchestration")
    print("="*50)
    
    # Initialize orchestrator with test parameters
    orchestrator = DataPipelineOrchestrator(
        input_folder="/home/melissa-ghemari/estin-chatbot/data-pipeline/sample-data",
        output_folder="test_outputs"
    )
    
    # Run the pipeline
    start_time = time.time()
    results = orchestrator.process_folder()
    end_time = time.time()
    
    # Display results
    if results["success"]:
        print(f"\nPipeline test completed successfully!")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Check outputs in: {orchestrator.output_folder}")
    else:
        print(f"\n❌ Pipeline test failed: {results.get('message', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    test_orchestration()