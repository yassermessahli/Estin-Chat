from client import client

def get_collection_record_count(collection_name="estin_docs"):
    """
    Get the number of records in a Milvus collection.
    
    Args:
        collection_name (str): Name of the collection to check
        
    Returns:
        int: Number of records in the collection
    """
    try:
        # Check if collection exists
        if not client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return 0
        
        # Get collection statistics
        collection_stats = client.get_collection_stats(collection_name=collection_name)
        
        # Extract row count from stats
        row_count = collection_stats.get('row_count', 0)
        
        print(f"Collection '{collection_name}' contains {row_count:,} records")
        
        return row_count
        
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return 0

def print_detailed_collection_stats(collection_name="estin_docs"):
    """
    Print detailed statistics about a Milvus collection.
    
    Args:
        collection_name (str): Name of the collection to check
    """
    try:
        # Check if collection exists
        if not client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return
        
        # Get collection statistics
        collection_stats = client.get_collection_stats(collection_name=collection_name)
        
        print(f"\n=== Collection '{collection_name}' Statistics ===")
        print(f"Total Records: {collection_stats.get('row_count', 0):,}")
        
        # Print all available stats
        for key, value in collection_stats.items():
            if key != 'row_count':  # Already printed above
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error getting detailed collection stats: {e}")

def main():
    """Main function to demonstrate usage"""
    collection_name = "estin_docs"
    
    print("Checking collection record count...")
    record_count = get_collection_record_count(collection_name)
    
    print("\nDetailed collection statistics:")
    print_detailed_collection_stats(collection_name)

if __name__ == "__main__":
    main()