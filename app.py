# app.py  –  City‑ordinance RAG demo (May‑2025 versions)
# ──────────────────────────────────────────────────────
import os, pickle, geoip2.database
from pathlib import Path
import streamlit as st
from streamlit_extras.stoggle import stoggle
from dotenv import load_dotenv

from langchain_openai               import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders     import PyPDFLoader
from langchain.text_splitter        import RecursiveCharacterTextSplitter

# **new‑style chain objects**
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain

# ──────────────────────────────────────────────────────
# 0. ENV / CONFIG
# ──────────────────────────────────────────────────────
load_dotenv()
DATA_DIR        = "ordinances"
INDEX_DIR       = "faiss_index"
HISTORY_FILE    = ".conv_history.pkl"
MAX_TURNS_SAVED = 50

# ──────────────────────────────────────────────────────
# 1. HELPERS
# ──────────────────────────────────────────────────────
def get_user_city_guess() -> str | None:
    """Best‑effort geo‑IP → city name (very coarse)."""
    try:
        reader = geoip2.database.Reader("GeoLite2‑City.mmdb")
        ip     = (st.experimental_get_query_params().get("ip", [None])[0]
                  or st.request.remote_addr)
        return (city := reader.city(ip).city.name) and city.lower()
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
@st.cache_resource(show_spinner="⏳ Building / loading vector store …")
def load_chain():
    embeddings = OpenAIEmbeddings()

    # ---------- build / load FAISS index ----------------
    if not os.path.exists(INDEX_DIR):
        docs = []
        for pdf in os.listdir(DATA_DIR):
            if not pdf.endswith(".pdf"):
                continue
            city_name = pdf.removesuffix(".pdf").lower()

            for d in PyPDFLoader(Path(DATA_DIR, pdf)).load():
                d.metadata.update(
                    source_file=pdf,
                    timestamp="2024‑01‑01",
                    city=city_name,
                )
                docs.append(d)

        chunks      = RecursiveCharacterTextSplitter(
                         chunk_size=1_000, chunk_overlap=150
                      ).split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)
    else:
        # 0.2.x signature – no kwarg
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings)

    # ---------- prompt & chain wiring -------------------
    tmpl = """
You are a compliance assistant for city officials.

RULES
• Return **one very concise sentence** that directly answers the question with
  an exact quotation from the ordinance (no interpretation).
• Quote must come from the requested city’s code (metadata.city).
• Nothing relevant → respond exactly:
  I'm sorry, I could not find a relevant ordinance addressing your question.

FORMAT
<<ANSWER>>
(Exact quoted text)

<<CONTEXT>>
(other surrounding clauses)"""
    prompt = PromptTemplate(template=tmpl,
                            input_variables=["context", "question"])

    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0)

    # new helper returns a StuffDocumentsChain object
    
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt=QA_PROMPT,
        document_variable_name="context",
    )

    
     qa_chain = ConversationalRetrievalChain(
         retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
         combine_docs_chain=combine_docs_chain,
         # defaults for question‑generator are fine
         return_source_documents=True,
     )
    return qa_chain


chain = load_chain()

# ──────────────────────────────────────────────────────
# 3. STREAMLIT UI
# ──────────────────────────────────────────────────────
st.title("🏛️ City Ordinance Reference Bot")

# ---------- sidebar -----------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Admin")

    st.subheader("Manage Ordinances")
    uploaded = st.file_uploader("Add PDF", type="pdf")
    if uploaded:
        dest = Path(DATA_DIR) / uploaded.name
        dest.write_bytes(uploaded.read())
        st.success("Uploaded.  Restart app to re‑index.")

    del_file = st.text_input("Filename to remove")
    if st.button("Delete") and del_file:
        try:
            (Path(DATA_DIR) / del_file).unlink()
            st.success("Deleted.  Restart app to re‑index.")
        except FileNotFoundError:
            st.error("File not found.")

    st.markdown("---")
    st.subheader("💳 Billing / Licences")
    st.write("Handled in admin dashboard (Stripe + Supabase).")

# ---------- main panel --------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history()

cities   = sorted(p.removesuffix(".pdf").lower()
                  for p in os.listdir(DATA_DIR) if p.endswith(".pdf"))
guess    = get_user_city_guess()
city_sel = st.selectbox("Select city", cities,
                        index=cities.index(guess) if guess in cities else 0)

question = st.text_input("Ask your ordinance question")
if st.button("🔍 Answer") and question:
    q = f"[city:{city_sel}] {question}"
    resp = chain({"question": q,
                  "chat_history": st.session_state.history})

    docs = [d for d in resp["source_documents"]
            if d.metadata.get("city") == city_sel]

    if not docs:
        st.error("I'm sorry, I could not find a relevant ordinance addressing your question.")
    else:
        answer   = docs[0].page_content.strip()
        context  = "\n\n".join(d.page_content.strip() for d in docs[1:])

        st.markdown(f"### 🎯 Answer\n> {answer}")
        stoggle("Show additional context", context or "_No further context available._")

        st.session_state.history.append((question, answer))
        save_history(st.session_state.history)
