import os
from pymilvus import MilvusClient
import ollama
import json
import re

# -------------------------------
# Milvus Setup (with auto-embedding)
# -------------------------------
MILVUS_HOST = os.getenv("MILVUS_HOST", "http://localhost:19530")
COLLECTION_NAME = "estin_docs"
client = MilvusClient(uri=MILVUS_HOST)

# -------------------------------
# Candidate Filter Fields
# -------------------------------
FILTER_FIELDS = {
    "level": ["1CP"],
    "semester": ["S1"],
    "subject_code": ["ELEC"]
}

# -------------------------------
# Step 1: Use LLM to classify filter fields (Improved)
# -------------------------------
def classify_query_filters(query: str) -> dict:
    """Let the LLM intelligently infer which filter values are relevant to the user query."""
    system_msg = (
        "You are an intelligent academic classifier for an engineering education system. "
        "Your task is to analyze student queries and intelligently determine which course filters apply.\n\n"
        "Available course structure:\n"
        f" - level: {FILTER_FIELDS['level']} (1st year engineering program)\n"
        f" - semester: {FILTER_FIELDS['semester']} (first semester courses)\n"
        f" - subject_code: {FILTER_FIELDS['subject_code']} (electrical engineering and electronics)\n\n"
        "INTELLIGENT INFERENCE GUIDELINES:\n"
        "1. SUBJECT ANALYSIS: Understand the topic domain of the query\n"
        "   - Electrical/Electronics concepts → 'ELEC'\n"
        "   - Consider the scientific and engineering context\n"
        "   - Think about what field of study the question belongs to\n\n"
        "2. ACADEMIC LEVEL: Consider the complexity and context\n"
        "   - Introductory/foundational questions → '1CP' (1st year)\n"
        "   - Basic engineering concepts → '1CP'\n\n"
        "3. CURRICULUM TIMING: Think about when topics are typically taught\n"
        "   - Fundamental concepts → 'S1' (first semester)\n"
        "   - Basic theory and principles → 'S1'\n\n"
        "Respond with ONLY a valid JSON object, no explanations or extra text."
    )

    try:
        response = ollama.chat(model="qwen3:8b", messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this query: '{query}'"}
        ])

        content = response["message"]["content"].strip()
        
        # Remove any thinking tags or extra content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        # Extract JSON from code blocks if present
        if "```json" in content:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        elif "```" in content:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        
        # Try to find JSON pattern if other methods fail
        if not content.startswith('{'):
            json_match = re.search(r'\{[^}]*\}', content)
            if json_match:
                content = json_match.group(0)
        
        # Parse JSON
        result = json.loads(content)
        
        # Validate and filter results
        validated_result = {}
        for k, v in result.items():
            if k in FILTER_FIELDS and v in FILTER_FIELDS[k]:
                validated_result[k] = v
        
        print(f"✅ Successfully parsed filters: {validated_result}")
        return validated_result
        
    except Exception as e:
        print(f"[Warning] Failed to parse classification: {e}")
        print(f"Raw content: {response['message']['content'] if 'response' in locals() else 'No response'}")
        
        # Fallback: try to extract filters using keyword matching
        fallback_filters = extract_filters_fallback(query)
        print(f"🔄 Using fallback extraction: {fallback_filters}")
        return fallback_filters

def extract_filters_fallback(query: str) -> dict:
    """Enhanced LLM-based fallback for filter extraction when primary classification fails."""
    fallback_prompt = (
        "You are a backup academic classifier. The primary classifier failed to process this query. "
        "Please analyze this student question and determine the most appropriate academic filters.\n\n"
        "Query to analyze: \"{}\"\n\n"
        "Available options:\n"
        f"- level: {FILTER_FIELDS['level']}\n"
        f"- semester: {FILTER_FIELDS['semester']}\n"
        f"- subject_code: {FILTER_FIELDS['subject_code']}\n\n"
        "Think about:\n"
        "1. What academic field does this question belong to?\n"
        "2. What level of student would ask this question?\n"
        "3. When in the curriculum would this topic be taught?\n\n"
        "Provide only a JSON object with your analysis."
    ).format(query)
    
    try:
        response = ollama.chat(model="qwen3:8b", messages=[
            {"role": "system", "content": "You are an academic classifier. Respond only with JSON."},
            {"role": "user", "content": fallback_prompt}
        ])
        
        content = response["message"]["content"].strip()
        
        # Clean up the response
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        if "```json" in content:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        elif "```" in content:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        
        if not content.startswith('{'):
            json_match = re.search(r'\{[^}]*\}', content)
            if json_match:
                content = json_match.group(0)
        
        result = json.loads(content)
        
        # Validate results
        validated_result = {}
        for k, v in result.items():
            if k in FILTER_FIELDS and v in FILTER_FIELDS[k]:
                validated_result[k] = v
        
        print(f"🔄 Fallback LLM classification: {validated_result}")
        return validated_result
        
    except Exception as e:
        print(f"❌ Fallback LLM classification also failed: {e}")
        # Last resort: return reasonable defaults for engineering queries
        return {"subject_code": "ELEC", "level": "1CP", "semester": "S1"}

# -------------------------------
# Step 2: Perform semantic search in Milvus (Improved)
# -------------------------------
def build_milvus_filter(filters: dict) -> str:
    """Build a proper Milvus filter expression."""
    if not filters:
        return None
    
    conditions = []
    for field, value in filters.items():
        # Use proper Milvus filter syntax
        conditions.append(f'{field} == "{value}"')
    
    return " and ".join(conditions)

def retrieve_documents(query: str, filters: dict, top_k=5) -> list:
    """Search Milvus using raw query text and filters."""
    try:
        # Build proper filter expression
        filter_expr = build_milvus_filter(filters)
        print(f"🔍 Milvus filter expression: {filter_expr}")
        
        # Try search with filters first
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query],
                filter=filter_expr,
                output_fields=["chunk", "title"],
                limit=top_k
            )
        except Exception as filter_error:
            print(f"⚠️ Search with filters failed: {filter_error}")
            print("🔄 Trying search without filters...")
            
            # If partition key error, try without any filters
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query],
                output_fields=["chunk", "title"],
                limit=top_k
            )
        
        if results and len(results) > 0 and len(results[0]) > 0:
            chunks = []
            for hit in results[0]:
                # Handle different response formats
                if hasattr(hit, 'entity'):
                    chunk_data = hit.entity
                else:
                    chunk_data = hit
                
                chunk_text = chunk_data.get("chunk", "") if isinstance(chunk_data, dict) else getattr(chunk_data, "chunk", "")
                title = chunk_data.get("title", "Unknown") if isinstance(chunk_data, dict) else getattr(chunk_data, "title", "Unknown")
                
                if chunk_text:
                    chunks.append(f"[{title}] {chunk_text}")
            
            print(f"✅ Found {len(chunks)} relevant documents")
            return chunks
        else:
            print("⚠️ No search results returned")
            return []
            
    except Exception as e:
        print(f"[Error] Milvus search failed: {e}")
        print(f"Filter used: {filters}")
        
        # Last resort: try a basic search without any parameters
        try:
            print("🔄 Attempting basic search as last resort...")
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query],
                limit=top_k
            )
            
            if results and len(results) > 0:
                chunks = []
                for hit in results[0]:
                    # Very basic chunk extraction
                    if hasattr(hit, 'entity') and hasattr(hit.entity, 'chunk'):
                        chunks.append(str(hit.entity.chunk))
                    elif isinstance(hit, dict) and 'chunk' in hit:
                        chunks.append(str(hit['chunk']))
                
                if chunks:
                    print(f"✅ Basic search found {len(chunks)} documents")
                    return chunks
        except Exception as basic_error:
            print(f"❌ Even basic search failed: {basic_error}")
        
        return []

# -------------------------------
# Step 3: Send final context to LLM (Improved)
# -------------------------------
def generate_answer(query: str, context_chunks: list) -> str:
    """Use LLM (Qwen) to generate the final answer."""
    if not context_chunks:
        return "I couldn't find any relevant documents to answer your question."
    
    context_text = "\n\n".join(context_chunks[:3])  # Limit to top 3 chunks
    
    prompt = (
        "You are a helpful study assistant for ESTIN engineering students. "
        "Answer the question based ONLY on the following course documents. "
        "If the documents don't contain enough information to answer the question completely, "
        "say so and provide what information is available.\n\n"
        "COURSE DOCUMENTS:\n"
        f"{context_text}\n\n"
        f"STUDENT QUESTION: {query}\n\n"
        "Please provide a clear, structured answer based on the course material above."
    )

    
    response = ollama.chat(model="qwen3:8b", messages=[
        {"role": "system", "content": "You are a helpful academic assistant. Answer clearly and accurately based only on the provided context."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

# -------------------------------
# Step 4: Full RAG Chain (Improved)
# -------------------------------
def rag_respond(query: str):
    print(f"\n🔍 User Query: {query}")
    print("=" * 50)

    # Step 1: Primary LLM-based filter classification
    print("Step 1: Intelligent filter inference using LLM...")
    filters = classify_query_filters(query)
    print(f"🧠 Primary LLM Classification: {filters}")
    
    # Step 2: If primary classification is insufficient, use LLM fallback
    if not filters or len(filters) == 0:
        print("🔄 Primary classification empty, using enhanced LLM fallback...")
        filters = extract_filters_fallback(query)
        print(f"🎯 Fallback LLM Classification: {filters}")
    
    # Step 3: Search documents
    print(f"\nStep 2: Searching documents with intelligently inferred filters: {filters}")
    chunks = retrieve_documents(query, filters)
    
    if not chunks:
        return "I couldn't find any relevant documents to answer your question. This might be due to collection configuration issues or the documents may not be indexed properly."

    # Step 4: Generate answer
    print(f"\nStep 3: Generating answer using {len(chunks)} document(s)...")
    answer = generate_answer(query, chunks)
    return answer

# -------------------------------
# Collection Info Helper
# -------------------------------
def check_collection_info():
    """Check if collection exists and get basic info."""
    try:
        collections = client.list_collections()
        print(f"Available collections: {collections}")
        
        if COLLECTION_NAME in collections:
            # Get collection stats
            stats = client.get_collection_stats(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' stats: {stats}")
            return True
        else:
            print(f"Collection '{COLLECTION_NAME}' not found!")
            return False
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False

# -------------------------------
# CLI (Improved)
# -------------------------------
if __name__ == "__main__":
    print("RAG Chatbot (Milvus + Qwen) - Improved Version")
    print("=" * 50)
    
    # Check collection status
    print("Checking collection status...")
    if not check_collection_info():
        print("⚠️ Collection issues detected. Continuing anyway...")
    
    print("\nType 'exit' to quit, 'help' for commands\n")
    
    while True:
        query = input("🧑‍🎓 You: ")
        
        if query.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        elif query.lower() == "help":
            print("Available commands:")
            print("- Ask any question about your ELEC course")
            print("- 'exit' or 'quit' to stop")
            print("- 'help' to see this message")
            continue
        elif not query.strip():
            continue
            
        try:
            response = rag_respond(query)
            print(f"\n🤖 Assistant: {response}")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("=" * 50)