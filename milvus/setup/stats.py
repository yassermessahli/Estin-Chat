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

def print_random_records(collection_name="estin_docs", limit=1):
    """
    Print random records from a Milvus collection.
    
    Args:
        collection_name (str): Name of the collection to query
        limit (int): Number of random records to display
    """
    try:
        # Check if collection exists
        if not client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return
        
        # Query random records
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["*"],
            limit=limit
        )
        
        if not results:
            print(f"No records found in collection '{collection_name}'")
            return
                
        for i, record in enumerate(results, 1):
            print(f"\nRecord {i}:")
            for field, value in record.items():
                # Truncate long text fields for readability
                if field == "vector":
                    print(f"  {field}: {type(value).__name__} with {len(value)} dimensions")
                    continue
                elif isinstance(value, str) and len(value) > 30:
                    value = value[:30] + "..."
                print(f"  {field}: {value}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error querying random records: {e}")

def main():
    """Main function to demonstrate usage"""
    collection_name = "estin_docs"
    
    print("="*50)
    print("Checking collection record count...")
    record_count = get_collection_record_count(collection_name)
    
    print("\n" + "="*50)
    print("Displaying random records:")
    print_random_records(collection_name)

if __name__ == "__main__":
    main()