import openai
import re
import requests
from config import embed_model, index

# In-memory session storage (not persistent)
session_questions = []

def estimate_max_tokens(query, context, base_tokens=150, max_limit=500):
    context_text = "\n".join(context)
    extra_tokens = len(context_text) // 4
    estimated_tokens = base_tokens + extra_tokens
    return min(estimated_tokens, max_limit)

### Helper functions for URL extraction and verification

def extract_urls(text):
    """
    Extracts all URLs from the provided text using regex.
    """
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

def verify_urls(url_list):
    """
    Given a list of URLs, performs HTTP HEAD requests to check their status.
    Returns a tuple: (valid_urls, invalid_urls) where invalid_urls is a list of (url, status) tuples.
    """
    valid_urls = []
    invalid_urls = []
    for url in url_list:
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                valid_urls.append(url)
            else:
                invalid_urls.append((url, response.status_code))
        except Exception as e:
            invalid_urls.append((url, str(e)))
    return valid_urls, invalid_urls

def verify_urls_in_text(text):
    """
    Extracts URLs from text, verifies them, and annotates any invalid URLs.
    """
    urls = extract_urls(text)
    if not urls:
        return text
    valid_urls, invalid_urls = verify_urls(urls)
    for url, status in invalid_urls:
        # Annotate invalid URLs in the text.
        text = text.replace(url, f"{url} (invalid: {status})")
    return text

### Fallback Scraper Agent
def fallback_scraper_agent(query):
    prompt = f"""Search the web for detailed information about: '{query}' in the context of Northeastern University. Provide a concise summary.
Also provide a helpful link."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        scraped_text = response["choices"][0]["message"]["content"].strip()
        # Verify and annotate URLs in the scraped text.
        verified_text = verify_urls_in_text(scraped_text)
        return verified_text
    except Exception as e:
        print("Fallback scraper error:", e)
        return ""

### Query Optimizer Agent
def query_optimizer_agent(query):
    prompt = f"As a knowledgeable assistant on Northeastern University, improve the following query for clarity: '{query}'"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=60
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Query optimizer error:", e)
        return query

### Pinecone Agent (Context Retrieval)
def retrieve_context(query, top_k=3, threshold=0.7):
    query_embedding = embed_model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context = []
    if result and "matches" in result:
        for match in result["matches"]:
            if match.get("score", 0) >= threshold:
                context.append(match["metadata"].get("combined_text", ""))
    return context

### RAG Agent (Response Optimizer)
def rag_agent(query, context, chat_history=""):
    if not context or all(not c.strip() for c in context):
        return ("I'm sorry, I don't have sufficient information about this topic at the moment. "
                "Please visit our [FAQs page](https://northeastern.edu/faqs) or contact our support team at "
                "[support@northeastern.edu](mailto:support@northeastern.edu) for further assistance.")
    context_text = "\n\n".join(context)
    prompt = (
        f"Chat History:\n{chat_history}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Please provide a clear, concise, and helpful answer about Northeastern University. "
        "Format your answer in Markdown. Use headings for major sections, bullet points for lists, "
        "and include clickable links which return a 200 OK status where appropriate. At the end write this as a fallback"
        "if the answer has links - 'If above provided link doesn’t work or you need the latest details,"
        "please visit the official [Northeastern program page](https://graduate.northeastern.edu/programs/) or use the [search function](https://www.northeastern.edu/search/).'"
        "If the answer does not have links, do not write the above line."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant for Northeastern University."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=estimate_max_tokens(query, context)
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("RAG error:", e)
        return ("I'm sorry, I couldn't generate a response right now. Please contact support.")


### Retrieve a Question by Index from Session
def get_question_by_index(ordinal: str) -> str:
    """
    Retrieve a past user question from the session_questions list based on an ordinal.
    Ordinal can be a word (e.g., 'first', 'second') or a digit.
    """
    ordinal_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
    }
    try:
        if ordinal.isdigit():
            index_num = int(ordinal)
        else:
            index_num = ordinal_map.get(ordinal.lower(), 1)
    except Exception as e:
        index_num = 1

    if 1 <= index_num <= len(session_questions):
        return session_questions[index_num - 1]
    else:
        return "No such question found in this session."

### Main Chatbot Processing Function
def process_chat(user_query: str, chat_history: list[str] = []) -> str:
    print(f"\n[PROCESS_CHAT] 🔹 Received user query: {user_query}")
    
    # Check if user is asking for a specific past question
    match = re.search(r"what was my (\w+)[\s-]*question", user_query.lower())
    if match:
        ordinal = match.group(1)
        retrieved_question = get_question_by_index(ordinal)
        return f"Your requested question: {retrieved_question}"
    
    # Append this question to the session history
    session_questions.append(user_query)

    # Step 1: Optimize query
    optimized_query = query_optimizer_agent(user_query)
    print(f"[OPTIMIZER] ✨ Optimized query: {optimized_query}")

    # Step 2: Retrieve from Pinecone
    context_docs = retrieve_context(optimized_query)
    print(f"[PINECONE] 📚 Retrieved {len(context_docs)} context docs from Pinecone")

    # Step 3: Fallback if no context found
    if not context_docs or all(not c.strip() for c in context_docs):
        print("[FALLBACK] ⚠️ No valid context found, using fallback scraper agent...")
        fallback_context = fallback_scraper_agent(optimized_query)
        context_docs = [fallback_context]
        print(f"[FALLBACK] 🪄 Fallback context length: {len(fallback_context)}")

    # Step 4: Format chat history (combine provided chat_history with this question)
    formatted_chat_history = "\n".join([f"User: {q}" for q in chat_history + [user_query]])

    # Step 5: Run through RAG agent
    final_response = rag_agent(optimized_query, context_docs, formatted_chat_history)
    print(f"[RAG] ✅ Final response length: {len(final_response)}")

    return final_response
