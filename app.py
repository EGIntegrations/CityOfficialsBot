###############################################################################
#  CityOfficialsBot ‑ main app
###############################################################################
import os, pickle, time, re, json
from pathlib import Path

import streamlit as st
from streamlit_extras.stoggle import stoggle
from dotenv import load_dotenv

from langchain_openai           import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain.prompts          import PromptTemplate
from langchain.chains           import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

import geoip2.database

###############################################################################
# 0. ENV / CONFIG
###############################################################################
load_dotenv()

DATA_DIR        = "ordinances"        # PDF ordinances live here
INDEX_DIR       = "faiss_index"       # FAISS vector store location
HISTORY_FILE    = ".conv_history.pkl" # persisted chat history
MAX_TURNS_SAVED = 50

###############################################################################
# 1. UTILITY HELPERS
###############################################################################
def get_user_city_guess() -> str | None:
    """Geo‑locate IP to guess the user’s city‑name (lower‑case)."""
    try:
        reader = geoip2.database.Reader("GeoLite2-City.mmdb")
        ip     = st.experimental_get_query_params().get("ip", [None])[0] \
                 or st.request.remote_addr
        resp   = reader.city(ip)
        return resp.city.name.lower() if resp.city and resp.city.name else None
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

###############################################################################
# 2. VECTOR INDEX & LANGCHAIN PIPELINE
###############################################################################
@st.cache_resource(show_spinner="⏳ Loading models & building FAISS index …")
def load_chain() -> ConversationalRetrievalChain:
    embeddings = OpenAIEmbeddings()

    # ────────────────────────────────────────────────────────────────────
    # Build or load FAISS
    # ────────────────────────────────────────────────────────────────────
    if not os.path.exists(INDEX_DIR):
        docs: list = []
        for pdf in Path(DATA_DIR).glob("*.pdf"):
            city_name = pdf.stem.lower()

            loader      = PyPDFLoader(str(pdf))
            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata.update(
                    source_file = pdf.name,
                    timestamp   = "2024‑01‑01",
                    city        = city_name,
                )
            docs.extend(loaded_docs)

        splitter    = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs  = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
    else:
        vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    # ────────────────────────────────────────────────────────────────────
    # Prompt that the *StuffDocumentsChain* will use
    # ────────────────────────────────────────────────────────────────────
    custom_template = """
You are a compliance assistant for city officials.

RULES
• Return **one very concise sentence** that directly answers the question with an
  exact quotation from the ordinance (no interpretation).
• The quotation must come from the requested city's code (metadata.city).
• If nothing relevant is found reply exactly:
  I'm sorry, I could not find a relevant ordinance addressing your question.

FORMAT
<<ANSWER>>
(Exact quoted text)

<<CONTEXT>>
(other surrounding clauses)
"""
    prompt = PromptTemplate(
        template=custom_template,
        input_variables=["context", "question"],
    )

    # Stuff documents into {context} variable explicitly
    combine_chain = StuffDocumentsChain(
        llm                    = ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        prompt                 = prompt,
        document_variable_name = "context",
        verbose                = False,
        llm_chain              = llm_chain,
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Final Conversational Retrieval chain
    return ConversationalRetrievalChain(
        retriever              = vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain     = combine_chain,
        return_source_documents=True,
    )


chain = load_chain()

###############################################################################
# 3. STREAMLIT UI
###############################################################################
st.title("🏛️ City Ordinance Reference Bot")

# ---------- sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Admin")

    # 📂 Upload / remove ordinance PDFs (per‑user SaaS portal)
    st.subheader("Manage Ordinances")
    uploaded = st.file_uploader("Add PDF", type="pdf")
    if uploaded:
        dest = Path(DATA_DIR) / uploaded.name
        dest.write_bytes(uploaded.read())
        st.success(f"Uploaded {uploaded.name}.  Restart to re‑index.")

    remove_file = st.text_input("Filename to remove")
    if st.button("Delete") and remove_file:
        try:
            os.remove(Path(DATA_DIR) / remove_file)
            st.success("Deleted; restart to re‑index.")
        except FileNotFoundError:
            st.error("File not found.")

    st.markdown("---")
    st.subheader("💳 Billing / Licences")
    st.write("Handled in admin dashboard (Stripe + Supabase).")

# ---------- main panel -------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history()

cities = sorted({p.stem for p in Path(DATA_DIR).glob("*.pdf")})
guessed = get_user_city_guess()
city_requested = st.selectbox(
    "Select city", cities, index=cities.index(guessed) if guessed in cities else 0
)

question = st.text_input("Ask your ordinance question")
ask      = st.button("🔍 Answer")

if ask and question:
    combined_q = f"[city:{city_requested}] {question}"

    resp  = chain({"question": combined_q, "chat_history": st.session_state.history})
    docs  = [d for d in resp["source_documents"] if d.metadata.get("city") == city_requested.lower()]

    if not docs:
        st.error("I'm sorry, I could not find a relevant ordinance addressing your question.")
    else:
        quoted   = docs[0].page_content.strip()
        surround = "\n\n".join(d.page_content.strip() for d in docs[1:])

        st.markdown(f"### 🎯 Answer\n> {quoted}")

        stoggle(
            "Show additional context",
            surround or "_No further context available._",
        )

        # persist history
        st.session_state.history.append((question, quoted))
        save_history(st.session_state.history)
