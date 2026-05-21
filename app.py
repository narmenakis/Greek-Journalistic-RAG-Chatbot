import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List
import streamlit as st
from langchain_chroma import Chroma
from datetime import datetime, date
import time
import json
import re 

# Disable CUDA memory caching to save VRAM
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none" # Disable file watch reload in Streamlit

# Fix PyTorch, Streamlit compatibility issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Constants
CHROMA_PATH = "chroma_sample" # Local vector DB path
device = "cuda" if torch.cuda.is_available() else "cpu" # Choose GPU if available
llm_model_name = "ilsp/Llama-Krikri-8B-Instruct" # Greek Open Source LLM Krikri
embedding_model_name = "intfloat/multilingual-e5-large-instruct" # Multilingual Instruct Embedding model
SYSTEM_PROMPT = (
    "Είσαι ένας έμπειρος δημοσιογραφικός βοηθός. "
    "Απαντάς σε ερωτήσεις δημοσιογραφικού ενδιαφέροντος με βάση διαθέσιμα άρθρα. "
    "Πρέπει να απαντάς με αντικειμενικότητα, ουδετερότητα και χωρίς εικασίες. "
    "ΜΗΝ προσθέτεις προσωπικές υποθέσεις ή πληροφορίες που δεν αναφέρονται ρητά στα κείμενα. "
    "Δώσε έμφαση στα άρθρα τα οποία σχετίζονται με τη Συνομιλία μέχρι τώρα. "
    "Λάβε υπόψη τη σχετική σύνοψη που παρέχεται."
    "Οι απαντήσεις σου να είναι αναλυτικές και περιεκτικές. "
)
MAX_CHAT_HISTORY = 40  # Limit chat history to last 40 messages (20 question and answer pairs)

class E5InstructEmbeddings:
    def __init__(self):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.model.eval()

    # Computes the average embedding of non-padding tokens in the sequence,
    # masking out padding using the attention mask.
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Add 'passage: ' prefix
        instruction_texts = ["passage: " + text for text in texts]
        return self._embed(instruction_texts)
    
    def embed_query(self, text: str) -> List[float]:
        # Add custom 'Instruct: ' prefix
        instruction_text = f'Instruct: Given a journalistic question, retrieve relevant passages from news articles that factually \
        and clearly answer the question\n"Query: {text}'

        return self._embed([instruction_text])[0]
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts, 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()

# Cached loading of the language model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    
    # If using GPU, compile the model for performance
    if device == "cuda":
        model = torch.compile(model)

    return tokenizer, model

# Cached loading of the vector database and the metadata
@st.cache_resource
def load_vector_db():
    embedding_function = E5InstructEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@st.cache_data
def get_metadata_filters(_db):
    metadatas = _db.get()['metadatas']
    authors = sorted(set(meta.get("author") for meta in metadatas if meta.get("author")))
    websites = sorted(set(meta.get("website") for meta in metadatas if meta.get("website")))
    sections = sorted(set(meta.get("section") for meta in metadatas if meta.get("section")))
    return authors, websites, sections


# Function to log user interactions 
def log_interaction(user_id: str, question: str, answer: str):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "user": user_id,
        "timestamp": timestamp,
        "question": question,
        "answer": answer
    }
    with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# Function to log metrics (timings and word count)
def log_timings(user_id: str, question: str, timings: dict, word_count: int):
    log_entry = {
        "user": user_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "timings": timings,
        "word_count": word_count
    }
    with open("timings_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# Generate LLM output 
def generate_with_krikri(tokenizer, model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=(input_ids != tokenizer.eos_token_id).long()
    )

    generated_tokens = output[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not decoded_output:
        return "⚠️ Δεν κατάφερα να δημιουργήσω απάντηση. Δοκίμασε να επαναδιατυπώσεις την ερώτηση."

    return decoded_output

def date_to_timestamp(date_obj):
    return datetime.combine(date_obj, datetime.min.time()).timestamp()

# Main query answering logic
def answer_query(query_text, tokenizer, model, db, chat_history, author_filter, website_filter, section_filter, start_date, end_date, max_docs):
    timings = {}
    total_start_time = time.time()

    # Metadata filters
    filter_clauses = []

    if section_filter:
        filter_clauses.append({"section": {"$in": section_filter}})
    if author_filter:
        cleaned_authors = [a.strip() for a in author_filter]
        filter_clauses.append({"author": {"$in": cleaned_authors}})
    if website_filter:
        cleaned_websites = [w.strip() for w in website_filter]
        filter_clauses.append({"website": {"$in": cleaned_websites}})
    if start_date or end_date:
        if start_date and end_date:
            filter_clauses.append({
                "$and": [
                    {"datetime": {"$gte": date_to_timestamp(start_date)}},
                    {"datetime": {"$lte": date_to_timestamp(end_date)}}
                ]
            })
        elif start_date:
            filter_clauses.append({"datetime": {"$gte": date_to_timestamp(start_date)}})
        elif end_date:
            filter_clauses.append({"datetime": {"$lte": date_to_timestamp(end_date)}})

    filter_dict = {"$and": filter_clauses} if len(filter_clauses) > 1 else (filter_clauses[0] if filter_clauses else None)

    # Start timing retrieval
    start_retrieval_time = time.time()
    results = db.similarity_search_with_relevance_scores(query_text, k=1, filter=filter_dict)
    timings["retrieval_1"] = time.time() - start_retrieval_time

    if not results:
        return "❌ Δεν βρέθηκαν σχετικές πληροφορίες με τα επιλεγμένα φίλτρα.", "", [], {}, 0

    most_relevant_doc, score = results[0]

    if score < 0.80:
        return "❌ Δεν βρέθηκαν σχετικές πληροφορίες.", "", [], {}, 0

    # Start timing summary generation
    start_generation_time = time.time()
    summary_prompt = (
        f"{most_relevant_doc.page_content}"
    )
    
    summary = generate_with_krikri(
        tokenizer, 
        model,
        "Παρακάτω είναι ένα άρθρο. Κάνε μια σύνοψη 5 λέξεων:\n\n",
        summary_prompt
    )

    summary = summary.replace('"', '').replace("'", "").strip()
    timings["summary_generation"] = time.time() - start_generation_time

    # Second retrieval with expanded query
    expanded_query = f"{query_text} [Σύνοψη σχετικού άρθρου: {summary}]"
    print("summary:", summary)

    start_retrieval2_time = time.time()
    expanded_results = db.similarity_search_with_relevance_scores(
        expanded_query, 
        k=max_docs, 
        filter=filter_dict if filter_dict else None
    )
    timings["retrieval_2"] = time.time() - start_retrieval2_time

    final_results = sorted(
        [(doc, score) for doc, score in expanded_results if score >= 0.80],
        key=lambda x: x[1],
        reverse=True
    )[:max_docs] if expanded_results else results

    context_text = "\n\n---\n\n".join([
        f"URL: {doc.metadata.get('url', 'URL λείπει')}\n"
        f"Τίτλος: {doc.metadata.get('title', 'Τίτλος λείπει')}\n"
        f"Ιστοσελίδα: {doc.metadata.get('website', 'Ιστοσελίδα λείπει')}\n"
        f"Ημερομηνία: {doc.metadata.get('datetime', 'Ημερομηνία λείπει')}\n"
        f"Στήλη: {doc.metadata.get('section', 'Στήλη λείπει')}\n\n"
        f"{doc.page_content}"
        for doc, _ in final_results
    ])

    conversation_history = ""
    for role, msg, _ in chat_history:
        role_prefix = "Χρήστης:" if role == "user" else "Βοηθός:"
        conversation_history += f"{role_prefix} {msg}\n"

    user_prompt = f"""
Συνομιλία μέχρι τώρα:
{conversation_history}

Βασισμένος ΜΟΝΟ στο παρακάτω κείμενο και στην Συνομιλία μέχρι τώρα, απάντησε στην ερώτηση.
Σχετική σύνοψη: {summary}

{context_text}

-------

Ερώτηση: {query_text}
"""

    system_prompt = SYSTEM_PROMPT

    # Final answer generation
    start_answer_gen_time = time.time()
    answer = generate_with_krikri(tokenizer, model, system_prompt, user_prompt)
    timings["answer_generation"] = time.time() - start_answer_gen_time

    timings["total_time"] = time.time() - total_start_time

    # Output timings to console
    print("\n⏱️ Timing Stats:")
    for stage, duration in timings.items():
        print(f" - {stage}: {duration:.2f} seconds")
    word_count = len(answer.split())
    print(f" - word_count: {word_count} words")

    # Deduplicate URLs only for sources display:
    seen_urls = set()
    unique_sources = []
    for url, score in [
        (doc.metadata.get('url', 'URL λείπει'), round(score, 2)) for doc, score in final_results
    ]:
        if url not in seen_urls:
            unique_sources.append((url, score))
            seen_urls.add(url)

    sources = unique_sources

    return answer, context_text, sources, timings, word_count

def main():
    st.set_page_config(page_title="Greek Journalism RAG Chatbot", page_icon="🤖")
    tokenizer, model = load_model_and_tokenizer()
    db = load_vector_db()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    all_authors, all_websites, all_sections = get_metadata_filters(db)

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.title("Καλώς ήρθες στο Greek Journalism RAG Chatbot! 💬")
        email_input = st.text_input("Παρακαλώ γράψε το email σου για να συνεχίσεις:")

        if st.button("Υποβολή"):
            # Check email pattern w/ regex
            email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
            if re.match(email_pattern, email_input.strip()):
                st.session_state.user_id = email_input.strip()
                st.rerun() 
            else:
                st.error("Παρακαλώ εισάγετε ένα έγκυρο email.")
        return

    st.title("Greek Journalism RAG Chatbot 💬")
    st.sidebar.markdown(f"### Συνδεδεμένος ως:\n**{st.session_state.user_id}**")
    # Sidebar filters
    st.sidebar.header("📊 Φίλτρα")
    
    author_filter = st.sidebar.multiselect("Συγγραφέας", options=all_authors)
    website_filter = st.sidebar.multiselect("Ιστοσελίδα", options=all_websites)
    section_filter = st.sidebar.multiselect("Στήλη", options=all_sections)
    start_date = st.sidebar.date_input("Από Ημερομηνία", value=None)
    end_date = st.sidebar.date_input("Έως Ημερομηνία", value=None)

    max_docs = st.sidebar.slider("Μέγιστος αριθμός πηγών", min_value=1, max_value=20, value=5)

    # Display past chat history
    for role, msg, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)
        if role == "assistant" and sources:
            with st.expander("🔗 Δες τις Πηγές"):
                for source in sources:
                    st.write(f"- {source}")

    # Handle new user input
    user_question = st.chat_input("Ρώτα κάτι σχετικό με τα δημοσιεύματα...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append(("user", user_question, []))

        with st.spinner("💭 Σκέφτομαι..."):
            answer, context, urls, timings, word_count = answer_query(
                user_question, tokenizer, model, db, st.session_state.chat_history[-MAX_CHAT_HISTORY:],  # trimmed chat_history
                author_filter, website_filter, section_filter,
                start_date if start_date else None,
                end_date if end_date else None,
                max_docs
            )

        with st.chat_message("assistant"):
            st.write(answer)
            if urls:
                with st.expander("🔗 Δες τις Πηγές"):
                    # for url, score in urls:
                    #     st.write(f"- **[{score}]** {url}")
                    for url, _ in urls:
                        st.write(f"- {url}")
               
        st.session_state.chat_history.append(("assistant", answer, urls))
        
        user_id = st.session_state.get("user_id")

        if user_id:
            log_interaction(user_id, user_question, answer)
            log_timings(user_id, user_question, timings, word_count)
        
if __name__ == "__main__":
    main()