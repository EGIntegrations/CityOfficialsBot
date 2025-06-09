# app.py  –  City‑ordinance RAG demo (May‑2025 versions)
# ──────────────────────────────────────────────────────
import os, pickle
from pathlib import Path
import streamlit as st
import random
import requests
import zipfile
import tempfile

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="City Ordinance Bot", page_icon="🏛️")

from streamlit_extras.stoggle import stoggle
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# **new‑style chain objects**
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Updated to use a zip file containing both .faiss and .pkl files
FAISS_INDEX_DIR = "faiss_index"
FAISS_ZIP_URL = "https://drive.google.com/file/d/1Akx96C3Wel41n_omWfsDGrfNff8RudYm/view?usp=share_link"

def download_and_extract_faiss_index():
    """Download and extract FAISS index from Google Drive zip file."""
    if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
        print("FAISS index directory already exists and is not empty.")
        return
    
    print("Downloading FAISS index zip file...")
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    try:
        # Download the zip file
        response = requests.get(FAISS_ZIP_URL, stream=True)
        response.raise_for_status()
        
        # Check if we got HTML instead of a zip file
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            raise ValueError("Received HTML page instead of zip file. Check your Google Drive sharing settings.")
        
        # Save and extract zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract zip file
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(FAISS_INDEX_DIR)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        print(f"FAISS index extracted to {FAISS_INDEX_DIR}")
        print("Files in index directory:", os.listdir(FAISS_INDEX_DIR))
        
    except Exception as e:
        print(f"Error downloading FAISS index: {e}")
        # If download fails, we'll build the index from scratch
        return False
    
    return True

# Try to download the index, if it fails we'll build from scratch
download_and_extract_faiss_index()


# ──────────────────────────────────────────────────────
# 0. ENV / CONFIG
# ──────────────────────────────────────────────────────
load_dotenv()

# Set OpenAI API key as environment variable for langchain
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

DATA_DIR        = "ordinances"
INDEX_DIR       = "faiss_index"
HISTORY_FILE    = ".conv_history.pkl"
MAX_TURNS_SAVED = 50

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set it in Streamlit secrets.")
    st.info("Go to your app settings and add OPENAI_API_KEY to secrets.")
    st.stop()

# ──────────────────────────────────────────────────────
# 1. HELPERS
# ──────────────────────────────────────────────────────
def get_user_city_guess() -> str | None:
    """Best‑effort geo‑IP → city name (very coarse)."""
    try:
        import geoip2.database
        reader = geoip2.database.Reader("GeoLite2-City.mmdb")
        # Get IP from query params or session state
        ip = st.query_params.get("ip", None)
        if not ip and hasattr(st, 'session_state') and 'remote_ip' in st.session_state:
            ip = st.session_state.get('remote_ip')
        if ip:
            result = reader.city(ip)
            return result.city.name.lower() if result.city.name else None
    except Exception:
        return None


def save_history(hist: list[tuple[str, str]]) -> None:
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(hist[-MAX_TURNS_SAVED:], f)


def load_history() -> list[tuple[str, str]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

# ──────────────────────────────────────────────────────
# 2. VECTOR INDEX & QA CHAIN   (cached)
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Building / loading vector store …")
def load_chain():
    try:
        # Initialize embeddings without explicit API key (uses env var)
        embeddings = OpenAIEmbeddings()

        # ---------- build / load FAISS index ----------------
        # Check if we have a valid FAISS index
        faiss_file = Path(INDEX_DIR) / "index.faiss"
        pkl_file = Path(INDEX_DIR) / "index.pkl"
        
        if faiss_file.exists() and pkl_file.exists():
            try:
                # Try to load existing index
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                st.write("✅ Loaded FAISS index from disk. Number of vectors:", vectorstore.index.ntotal)
            except Exception as e:
                st.warning(f"Failed to load existing index: {e}")
                st.write("Building new index from PDFs...")
                vectorstore = build_index_from_pdfs(embeddings)
        else:
            st.write("No valid FAISS index found. Building from PDFs...")
            vectorstore = build_index_from_pdfs(embeddings)

        if vectorstore is None:
            st.error("Failed to create or load vector store")
            return None, None

        # ---------- prompt & chain wiring -------------------
        QA_PROMPT = PromptTemplate(
            template="""You are a compliance assistant for city officials.

RULES:
• Return **one very concise sentence** that directly answers the question with
  an exact quotation from the ordinance (no interpretation).
• Quote must come from the requested city's code (check metadata.city).
• If nothing relevant found, respond exactly:
  "I'm sorry, I could not find a relevant ordinance addressing your question."

Context from ordinances:
{context}

Question: {input}

Answer:""",
            input_variables=["context", "input"]
        )

        # Initialize LLM without explicit API key (uses env var)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        # Create the document chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=QA_PROMPT
        )

        # Create the retrieval chain
        qa_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain=combine_docs_chain
        )
        
        return qa_chain, vectorstore
        
    except Exception as e:
        st.error(f"Error initializing chain: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def build_index_from_pdfs(embeddings):
    """Build FAISS index from PDF files."""
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("No PDF files found in ordinances directory. Please upload some PDFs.")
        # Create a dummy index to prevent errors
        return FAISS.from_texts(["Please upload PDF files to get started."], embeddings)
    
    st.write("📂 PDF files found in ordinances directory:", os.listdir(DATA_DIR))

    docs = []
    for pdf in os.listdir(DATA_DIR):
        if not pdf.endswith(".pdf"):
            continue
        city_name = pdf.removesuffix(".pdf").lower()

        try:
            loader = PyPDFLoader(str(Path(DATA_DIR) / pdf))
            loaded_docs = loader.load()
            st.write(f"📄 {pdf}: loaded {len(loaded_docs)} pages")
            for d in loaded_docs:
                d.metadata.update(
                    source_file=pdf,
                    timestamp="2024-01-01",
                    city=city_name,
                )
                docs.append(d)
        except Exception as e:
            st.warning(f"Failed to load {pdf}: {e}")

    st.write("📝 Total docs loaded from all PDFs:", len(docs))

    if docs:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1_000, chunk_overlap=150
        ).split_documents(docs)
        st.write("🔪 Chunks created:", len(chunks))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.write("📦 Vectorstore created! Number of vectors:", vectorstore.index.ntotal)
        
        # Save the index
        try:
            vectorstore.save_local(INDEX_DIR)
            st.write("💾 FAISS index saved successfully")
        except Exception as e:
            st.warning(f"Failed to save index: {e}")
        
        return vectorstore
    else:
        st.warning("No valid PDF content found. Creating dummy index.")
        return FAISS.from_texts(["Please upload valid PDF files."], embeddings)


# Initialize the chain
chain, vectorstore = load_chain()

# ──────────────────────────────────────────────────────
# 3. STREAMLIT UI
# ──────────────────────────────────────────────────────
st.title("🏛️ City Ordinance Reference Bot")

# Check if chain initialized properly
if chain is None:
    st.error("Failed to initialize the application. Please check your configuration.")
    st.stop()

# ---------- sidebar -----------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Admin")

    st.subheader("Manage Ordinances")
    uploaded = st.file_uploader("Add PDF", type="pdf")
    if uploaded:
        os.makedirs(DATA_DIR, exist_ok=True)
        dest = Path(DATA_DIR) / uploaded.name
        dest.write_bytes(uploaded.read())
        st.success("Uploaded! Click 'Clear Cache' to re-index.")
        if st.button("Clear Cache & Re-index"):
            st.cache_resource.clear()
            st.rerun()

    # List existing files
    if os.path.exists(DATA_DIR):
        existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if existing_files:
            st.subheader("Existing Files")
            del_file = st.selectbox("Select file to delete", [""] + existing_files)
            if st.button("Delete Selected") and del_file:
                try:
                    (Path(DATA_DIR) / del_file).unlink()
                    st.success("Deleted! Click 'Clear Cache' to re-index.")
                    if st.button("Clear Cache After Delete"):
                        st.cache_resource.clear()
                        st.rerun()
                except FileNotFoundError:
                    st.error("File not found.")

    st.markdown("---")
    st.subheader("💳 Billing / Licenses")
    st.write("Handled in admin dashboard (Stripe + Supabase).")

# ---------- main panel --------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history()

# Get available cities
cities = []
if os.path.exists(DATA_DIR):
    cities = sorted(p.removesuffix(".pdf").lower()
                    for p in os.listdir(DATA_DIR) if p.endswith(".pdf"))

if not cities:
    st.warning("No city ordinance PDFs found. Please upload some PDFs using the sidebar.")
    st.stop()

# City selection
guess = get_user_city_guess()
default_idx = cities.index(guess) if guess in cities else 0
city_sel = st.selectbox("Select city", cities, index=default_idx)

# Question input
question = st.text_input("Ask your ordinance question", 
                        placeholder="e.g., What are the noise ordinance hours?")

if st.button("🔍 Search Ordinances", type="primary") and question:
    with st.spinner("Searching ordinances..."):
        # Prepare the question with city context
        q = f"[city:{city_sel}] {question}"
        
        try:
            # Get response from chain
            response = chain.invoke({
                "input": q,
                "chat_history": []  # Simple version without conversation memory
            })
            
            # Extract answer and source documents
            answer = response.get("answer", "")
            source_docs = response.get("context", [])
            
            # Filter documents by selected city
            city_docs = [d for d in source_docs 
                        if d.metadata.get("city") == city_sel]
            
            if not city_docs or "could not find" in answer.lower():
                st.error("I'm sorry, I could not find a relevant ordinance addressing your question.")
            else:
                # Display answer
                st.markdown("### 🎯 Answer")
                st.info(answer)
                
                # Show source context
                if len(city_docs) > 1:
                    context = "\n\n---\n\n".join(
                        f"**From {d.metadata.get('source_file', 'Unknown')}:**\n{d.page_content.strip()}"
                        for d in city_docs[1:3]  # Show up to 2 additional contexts
                    )
                    with st.expander("📄 View Additional Context"):
                        st.markdown(context)
                
                # Save to history
                st.session_state.history.append((question, answer))
                save_history(st.session_state.history)
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.info("Make sure your OpenAI API key is set in Streamlit secrets")

# Show recent history
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Recent Questions")
    for q, a in reversed(st.session_state.history[-5:]):
        with st.expander(f"Q: {q[:50]}..."):
            st.write(f"**Answer:** {a}")
