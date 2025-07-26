import os
import requests
import json
import re
from pymilvus import MilvusClient
import time

# -------------------------------
# vLLM API Configuration
# -------------------------------
VLLM_BASE_URL = "http://localhost:8000"  # Your vLLM server URL
VLLM_MODEL_NAME = "/home/anis-bensmail/estin-chatbot/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf" 

def call_vllm_api(messages, max_tokens=1000, temperature=0.1, timeout=180):
    """Call vLLM API for chat completions with improved timeout and retry logic."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": VLLM_MODEL_NAME,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            print(f"🔄 API Call attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)")
            
            response = requests.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout  # Increased from 60 to 180 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ vLLM API Error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                    continue
                return None
                
        except requests.exceptions.Timeout as e:
            print(f"⏰ Request timed out (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying with longer timeout...")
                timeout *= 1.5  # Increase timeout for next attempt
                import time
                time.sleep(5)
                continue
            else:
                print("❌ All retry attempts exhausted due to timeout")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in 5 seconds...")
                import time
                time.sleep(5)
                continue
            return None
            
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return None
    
    return None

# -------------------------------
# Milvus Setup (with auto-embedding)
# -------------------------------
COLLECTION_NAME = "estin_docs"

MILVUS_HOST = os.getenv("MILVUS_HOST", "http://localhost:19530")

# Connect to Milvus
client = MilvusClient(uri=MILVUS_HOST, token="root:Milvus")
client.use_database("core_db")

# -------------------------------
# Candidate Filter Fields
# -------------------------------
FILTER_FIELDS = {
    "subject_code": [
        "ALG1", "ALG2", "ANA2", "ARCHI1", "ARCHI2", 
        "ASDS1", "ASDS2", "BW", "ELEC1", "ELECTRO1", 
        "ELECTRO2", "MEC", "SFSD"
    ],
    "document_type": ["COURS", "EXAM", "INTERRO", "OTHER", "TD", "TP"],
    "semester": ["S1", "S2"]
}

# -------------------------------
# Step 1: Use LLM to classify filter fields (Improved)
# -------------------------------
def classify_query_filters(query: str) -> dict:
    """Let the LLM intelligently infer which filter values are relevant to the user query."""
    system_msg = (
        "You are an intelligent academic classifier for an engineering education system.\n"
        "Your task is to analyze student queries and intelligently determine which course filters apply.\n\n"
        "If the question is in French, translate the answer in French too.\n\n"
        "Available course structure:\n"
        f" - subject_code: {FILTER_FIELDS['subject_code']} (various engineering subjects)\n"
        f" - document_type: {FILTER_FIELDS['document_type']} (different types of academic materials)\n\n"
        "INTELLIGENT INFERENCE GUIDELINES:\n"
        "1. SUBJECT ANALYSIS: Understand the topic domain of the query\n"
        "   - ALG1/ALG2: Algebra, linear algebra, mathematical foundations\n"
        "   - ANA2: Mathematical analysis, calculus, differential equations\n"
        "   - ARCHI1/ARCHI2: Computer architecture, digital systems, processors\n"
        "   - ASDS1/ASDS2: Data structures, algorithms, programming concepts\n"
        "   - BW: Basic workshop, practical engineering skills\n"
        "   - ELEC1: Basic electrical circuits, fundamentals\n"
        "   - ELECTRO1/ELECTRO2: Electronics, advanced electrical concepts\n"
        "   - MEC: Mechanics, mechanical engineering principles\n"
        "   - SFSD: Data and files structure\n\n"
        "2. DOCUMENT TYPE ANALYSIS: Determine what type of material is needed\n"
        "   - COURS: Lectures, theoretical content, course materials\n"
        "   - EXAM: Past exams, exam questions, assessment materials\n"
        "   - INTERRO: Quizzes, short tests, interrogations\n"
        "   - TD: Directed work, exercises, problem sets\n"
        "   - TP: Practical work, lab exercises, hands-on activities\n"
        "   - OTHER: General materials, mixed content\n\n"
        "3. CONTEXT UNDERSTANDING: Consider the intent behind the query\n"
        "   - Questions about theory/concepts → focus on COURS\n"
        "   - Need for practice problems → focus on TD\n"
        "   - Exam preparation → focus on EXAM\n"
        "   - Laboratory work → focus on TP\n\n"
        "Respond with ONLY a valid JSON object. Return each filter field as a list of relevant values, even if there's only one.\n"
        "Example:\n"
        "{\"subject_code\": [\"ASDS1\", \"ASDS2\"], \"document_type\": [\"TD\", \"COURS\"]}\n"
        "No explanations or extra text.\n"
    )

    try:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this query: '{query}'"}
        ]
        
        # Use shorter timeout for classification (simpler task)
        content = call_vllm_api(messages, max_tokens=300, temperature=0.1, timeout=90)
        
        if not content:
            raise Exception("No response from vLLM API")
        
        content = content.strip()
        
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
            if k in FILTER_FIELDS:
                if isinstance(v, list):
                    valid_values = [item for item in v if item in FILTER_FIELDS[k]]
                    if valid_values:
                        validated_result[k] = valid_values
                elif isinstance(v, str) and v in FILTER_FIELDS[k]:
                    validated_result[k] = [v]  # Convert single string to list

        print(f"✅ Successfully parsed filters: {validated_result}")
        return validated_result
        
    except Exception as e:
        print(f"[Warning] Failed to parse classification: {e}")
        print(f"Raw content: {content if 'content' in locals() else 'No response'}")
        
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
        f"- subject_code: {FILTER_FIELDS['subject_code']}\n"
        f"- document_type: {FILTER_FIELDS['document_type']}\n\n"
        "Think about:\n"
        "1. What academic field does this question belong to?\n"
        "2. What type of document would be most helpful?\n"
        "3. What subject code matches the query topic?\n\n"
        "Provide only a JSON object with your analysis."
    ).format(query)
    
    try:
        messages = [
            {"role": "system", "content": "You are an academic classifier. Respond only with JSON."},
            {"role": "user", "content": fallback_prompt}
        ]
        
        content = call_vllm_api(messages, max_tokens=200, temperature=0.1, timeout=90)
        
        if not content:
            raise Exception("No response from vLLM API")
        
        content = content.strip()
        
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
            if k in FILTER_FIELDS:
                if isinstance(v, list):
                    valid_values = [item for item in v if item in FILTER_FIELDS[k]]
                    if valid_values:
                        validated_result[k] = valid_values
                elif isinstance(v, str) and v in FILTER_FIELDS[k]:
                    validated_result[k] = [v]
        
        print(f"🔄 Fallback LLM classification: {validated_result}")
        return validated_result
        
    except Exception as e:
        print(f"❌ Fallback LLM classification also failed: {e}")
        # Last resort: return reasonable defaults for engineering queries
        return {"subject_code": ["ELEC1"], "document_type": ["COURS"]}

def build_milvus_filter(filters: dict) -> str:
    """Build a valid Milvus filter expression (avoiding OR on partition keys)."""
    if not filters:
        return None

    conditions = []
    for field, value in filters.items():
        if isinstance(value, list):
            if field == "subject_code":
                # ⚠️ Milvus does not support OR on partition keys — should only have one value here!
                if len(value) > 1:
                    raise ValueError("Multiple subject_code values not allowed in single filter (OR not supported on partition keys)")
                conditions.append(f'{field} == "{value[0]}"')
            else:
                or_conditions = [f'{field} == "{v}"' for v in value]
                conditions.append("(" + " or ".join(or_conditions) + ")")
        else:
            conditions.append(f'{field} == "{value}"')

    return " and ".join(conditions)

def retrieve_documents(query: str, filters: dict, top_k=10) -> list:
    """Search Milvus using raw query text and filters, handling OR on partition keys like subject_code."""
    try:
        all_results = []

        # Extract potential OR fields (partition key example: subject_code)
        subject_codes = filters.get("subject_code", [])
        other_filters = {k: v for k, v in filters.items() if k != "subject_code"}

        # If no subject_code filter, search once with other filters
        if not subject_codes:
            filter_expr = build_milvus_filter(other_filters)
            print(f"🔍 Milvus filter expression: {filter_expr}")
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query],
                filter=filter_expr,
                output_fields=["chunk", "title"],
                limit=top_k
            )
            all_results.extend(results[0] if results else [])
        else:
            # Perform separate searches per subject_code (Milvus can't do OR on partition keys)
            for code in subject_codes:
                # Create filters for this specific subject_code only
                current_filters = other_filters.copy()
                current_filters["subject_code"] = code  # Single string value, not list!
                
                filter_expr = build_milvus_filter(current_filters)
                print(f"🔍 Filter for subject_code={code}: {filter_expr}")
                
                try:
                    results = client.search(
                        collection_name=COLLECTION_NAME,
                        data=[query],
                        filter=filter_expr,
                        output_fields=["chunk", "title"],
                        limit=top_k
                    )
                    if results:
                        all_results.extend(results[0])
                        print(f"✅ Found {len(results[0])} results for subject_code={code}")
                except Exception as sub_error:
                    print(f"⚠️ Search failed for subject_code={code}: {sub_error}")

        # If we got results, deduplicate and sort by score
        if all_results:
            # Remove duplicates based on chunk content
            seen_chunks = set()
            unique_results = []
            for hit in all_results:
                chunk_data = hit.entity if hasattr(hit, "entity") else hit
                chunk_text = chunk_data.get("chunk", "") if isinstance(chunk_data, dict) else getattr(chunk_data, "chunk", "")
                
                if chunk_text and chunk_text not in seen_chunks:
                    seen_chunks.add(chunk_text)
                    unique_results.append(hit)
            
            # Sort by score (higher scores first)
            unique_results = sorted(
                unique_results,
                key=lambda r: getattr(r, "score", getattr(r, "distance", 0)),
                reverse=True
            )

            # Extract top_k chunks with similarity scores
            chunks = []
            print(f"\n📊 SIMILARITY SCORES:")
            print("-" * 60)
            for i, hit in enumerate(unique_results[:top_k]):
                chunk_data = hit.entity if hasattr(hit, "entity") else hit
                chunk_text = chunk_data.get("chunk", "") if isinstance(chunk_data, dict) else getattr(chunk_data, "chunk", "")
                title = chunk_data.get("title", "Unknown") if isinstance(chunk_data, dict) else getattr(chunk_data, "title", "Unknown")
                
                # Get similarity score (could be 'score' or 'distance' depending on Milvus setup)
                similarity_score = getattr(hit, "score", getattr(hit, "distance", 0))
                
                if chunk_text:
                    # Print similarity score information
                    print(f"📄 Document {i+1}: [{title}]")
                    print(f"🎯 Similarity Score: {similarity_score:.4f}")
                    print(f"📝 Preview: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
                    print("-" * 60)
                    
                    chunks.append({
                        'content': f"[{title}] {chunk_text}",
                        'score': similarity_score,
                        'title': title
                    })
            
            print(f"✅ Found {len(chunks)} unique relevant documents")
            return chunks
        else:
            print("⚠️ No search results returned after filtered search")
            return []

    except Exception as e:
        print(f"[Error] Milvus search failed: {e}")
        print(f"Filters used: {filters}")
        print("🔄 Attempting basic fallback search...")

        # Try basic search without any filters
        try:
            # For basic search without filters, we still need to handle partition key requirements
            # Try searching each subject_code individually if they were specified
            if filters.get("subject_code"):
                fallback_results = []
                for code in filters["subject_code"]:
                    try:
                        results = client.search(
                            collection_name=COLLECTION_NAME,
                            data=[query],
                            filter=f'subject_code == "{code}"',  # Include partition key in basic search
                            output_fields=["chunk", "title"],
                            limit=top_k
                        )
                        if results and len(results) > 0:
                            fallback_results.extend(results[0])
                    except Exception as code_error:
                        print(f"⚠️ Basic search failed for subject_code={code}: {code_error}")
                        continue
                
                if fallback_results:
                    chunks = []
                    print(f"\n📊 FALLBACK SIMILARITY SCORES:")
                    print("-" * 60)
                    for i, hit in enumerate(fallback_results[:top_k]):
                        chunk_data = hit.entity if hasattr(hit, "entity") else hit
                        chunk_text = chunk_data.get("chunk", "") if isinstance(chunk_data, dict) else getattr(chunk_data, "chunk", "")
                        title = chunk_data.get("title", "Unknown") if isinstance(chunk_data, dict) else getattr(chunk_data, "title", "Unknown")
                        similarity_score = getattr(hit, "score", getattr(hit, "distance", 0))
                        
                        if chunk_text:
                            print(f"📄 Document {i+1}: [{title}]")
                            print(f"🎯 Similarity Score: {similarity_score:.4f}")
                            print(f"📝 Preview: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
                            print("-" * 60)
                            
                            chunks.append({
                                'content': f"[{title}] {chunk_text}",
                                'score': similarity_score,
                                'title': title
                            })
                    print(f"✅ Fallback search found {len(chunks)} documents")
                    return chunks
            else:
                # If no subject_code was specified, we can't do a basic search due to partition key requirements
                print("❌ Cannot perform basic search without partition key (subject_code)")
                
        except Exception as fallback_error:
            print(f"❌ Fallback search also failed: {fallback_error}")
        
        return []

# -------------------------------
# Step 3: Send final context to LLM (Improved)
# -------------------------------
def generate_answer(query: str, context_chunks: list) -> str:
    """Use LLM (vLLM) to generate the final answer."""
    if not context_chunks:
        return "I couldn't find any relevant documents to answer your question."
    
    # Limit context to prevent timeout
    max_chunks = 5  # Reduced from 10 to 5 to prevent timeout
    context_chunks = context_chunks[:max_chunks]
    
    # Extract content from chunk dictionaries and limit size
    context_texts = []
    for chunk in context_chunks:
        if isinstance(chunk, dict):
            # Limit chunk content to prevent excessive context
            content = chunk['content']
            if len(content) > 800:  # Limit each chunk to 800 characters
                content = content[:800] + "..."
            context_texts.append(content)
        else:
            content = str(chunk)
            if len(content) > 800:
                content = content[:800] + "..."
            context_texts.append(content)
    
    context_text = "\n\n".join(context_texts)
    
    # Print context information with scores
    print(f"\n🧠 CONTEXT USED FOR ANSWER GENERATION:")
    print("-" * 60)
    for i, chunk in enumerate(context_chunks):
        if isinstance(chunk, dict):
            print(f"📄 Source {i+1}: {chunk['title']}")
            print(f"🎯 Score: {chunk['score']:.4f}")
            print(f"📝 Content: {chunk['content'][:150]}{'...' if len(chunk['content']) > 150 else ''}")
            print("-" * 60)
    
    # Simplified prompt to reduce processing time
    prompt = (
        "Based on the following course documents, answer the student's question clearly and concisely.\n\n"
        "COURSE DOCUMENTS:\n"
        f"{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        "Provide a clear answer based on the course material. If the question is in French, answer in French."
    )

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Use longer timeout for answer generation but reduce max_tokens
    response = call_vllm_api(messages, max_tokens=800, temperature=0.2, timeout=240)
    
    if response:
        return response
    else:
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

# -------------------------------
# Step 4: Full RAG Chain (Improved)
# -------------------------------
def rag_respond(query: str):
    print(f"\n🔍 User Query: {query}")
    print("=" * 50)

    # Step 1: Primary LLM-based filter classification
    print("Step 1: Intelligent filter inference using LLM...")
    start_time1 = time.time()
    filters = classify_query_filters(query)
    print(f"🧠 Primary LLM Classification: {filters}")
    end_time1 = time.time()
    print(f"\n⏱️ Execution Time: {end_time1 - start_time1:.4f} seconds")
    
    # Step 2: If primary classification is insufficient, use LLM fallback
    if not filters or len(filters) == 0:
        print("🔄 Primary classification empty, using enhanced LLM fallback...")
        filters = extract_filters_fallback(query)
        print(f"🎯 Fallback LLM Classification: {filters}")
    filters = {k: v if isinstance(v, list) else [v] for k, v in filters.items()}

    # Step 3: Search documents
    print(f"\nStep 2: Searching documents with intelligently inferred filters: {filters}")
    start_time2 = time.time()
    chunks = retrieve_documents(query, filters)
    end_time2 = time.time()
    print(f"\n⏱️ Execution Time: {end_time2 - start_time2:.4f} seconds")
    
    if not chunks:
        return "I couldn't find any relevant documents to answer your question. This might be due to collection configuration issues or the documents may not be indexed properly."

    # Step 4: Generate answer
    print(f"\nStep 3: Generating answer using {len(chunks)} document(s)...")
    start_time3 = time.time()
    answer = generate_answer(query, chunks)
    end_time3 = time.time()
    print(f"\n⏱️ Execution Time: {end_time3 - start_time3:.4f} seconds")
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

def check_vllm_connection():
    """Check if vLLM server is accessible."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ vLLM server is running and accessible")
            return True
        else:
            print(f"⚠️ vLLM server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to vLLM server: {e}")
        print(f"Make sure your vLLM server is running on {VLLM_BASE_URL}")
        return False

# -------------------------------
# CLI (Improved)
# -------------------------------
if __name__ == "__main__":
    print("RAG Chatbot (Milvus + vLLM) - Updated for vLLM Server")
    print("=" * 70)
    
    # Check vLLM server connection
    print("Checking vLLM server connection...")
    if not check_vllm_connection():
        print("⚠️ vLLM server connection issues detected. Please ensure your vLLM server is running.")
        print("You can start it with: vllm serve \"./models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf\" --max-model-len 4000")
        exit(1)
    
    # Check collection status
    print("Checking Milvus collection status...")
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
            print("- Ask any question about your course materials")
            print("- 'exit' or 'quit' to stop")
            print("- 'help' to see this message")
            print("- The system will show similarity scores for retrieved documents")
            print(f"- Using vLLM server at: {VLLM_BASE_URL}")
            continue
        elif not query.strip():
            continue
            
        try:
            time_all = time.time()
            response = rag_respond(query)
            print(f"\n🤖 Assistant: {response}")
            print("=" * 70)
            end_time_all = time.time()
            print(f"\n⏱️Final Execution Time: {end_time_all - time_all:.4f} seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("=" * 70)