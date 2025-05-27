import streamlit as st
from streamlit_extras.stoggle import stoggle   # ⬅️ expander‑button widget
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, re, pickle, json, time, geoip2.database

################################################################################
# 0. ENV / CONFIG
################################################################################
load_dotenv()
DATA_DIR       = "ordinances"
INDEX_DIR      = "faiss_index"
HISTORY_FILE   = ".conv_history.pkl"          # persisted chat history
MAX_TURNS_SAVED = 50                          # truncate history on disk

################################################################################
# 1. UTILITY HELPERS
################################################################################
def get_user_city_guess() -> str | None:
    """Geo‑locate the user’s IP to guess a city.  (Falls back to None)."""
    try:
        reader = geoip2.database.Reader("GeoLite2‑City.mmdb")
        ip     = st.experimental_get_query_params().get("ip",[None])[0] \
                 or st.request.remote_addr
        resp   = reader.city(ip)
        return resp.city.name.lower() if resp.city.name else None
    except Exception:
        return None


def save_history(chat_hist:list[tuple[str,str]]) -> None:
    with open(HISTORY_FILE,"wb") as f:
        pickle.dump(chat_hist[-MAX_TURNS_SAVED:], f)


def load_history() -> list[tuple[str,str]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,"rb") as f:
            return pickle.load(f)
    return []


################################################################################
# 2. VECTOR INDEX (persists locally)
################################################################################
@st.cache_resource(show_spinner="⏳ Building / loading vector store …")
def load_chain():
    embeddings = OpenAIEmbeddings()

    if not os.path.exists(INDEX_DIR):
        docs       = []
        for pdf in os.listdir(DATA_DIR):
            if not pdf.endswith(".pdf"): continue
            city_name   = pdf.removesuffix(".pdf").lower()

            loader      = PyPDFLoader(os.path.join(DATA_DIR,pdf))
            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata.update(
                    source_file = pdf,
                    timestamp   = "2024‑01‑01",
                    city        = city_name,
                )
            docs.extend(loaded_docs)

        splitter    = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
        docs        = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
    else:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings,
                                       allow_dangerous_deserialization=True)

    #######################################################
    # PROMPT — concise answer first line only
    #######################################################
    custom_template = """
You are a compliance assistant for city officials.

RULES
• Return **one very concise sentence** that directly answers the question with an exact
  quotation from the ordinance (no interpretation).  
• Quote must come from the requested city’s code (metadata.city).  
• Nothing relevant → respond exactly:
  I'm sorry, I could not find a relevant ordinance addressing your question.

FORMAT
<<ANSWER>>
(Exact quoted text)

<<CONTEXT>>
(other surrounding clauses)"""

    prompt = PromptTemplate(template=custom_template,
                            input_variables=["context","question"])
    llm    = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm           = llm,
        retriever     = vectorstore.as_retriever(search_kwargs={"k":6}),
        combine_docs_chain_kwargs = {
        "prompt": prompt,
        "document_variable_name": "context",
        },
        return_source_documents   = True,
    )

chain = load_chain()

################################################################################
# 3. STREAMLIT UI
################################################################################
st.title("🏛️ City Ordinance Reference Bot")

# ---------- sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Admin")
    # 📂 Upload / remove ordinance PDFs (per‑user SaaS portal)
    st.subheader("Manage Ordinances")
    uploaded = st.file_uploader("Add PDF", type="pdf")
    if uploaded is not None:
        dest = Path(DATA_DIR)/uploaded.name
        dest.write_bytes(uploaded.read())
        st.success(f"Uploaded {uploaded.name}.  Please restart to re‑index.")

    remove_file = st.text_input("Filename to remove")
    if st.button("Delete") and remove_file:
        try:
            os.remove(Path(DATA_DIR)/remove_file)
            st.success("Deleted; restart to re‑index.")
        except FileNotFoundError:
            st.error("File not found.")

    st.markdown("---")
    st.subheader("💳 Billing / Licences")
    st.write("Handled in admin dashboard (Stripe + Supabase).")

# ---------- main panel -------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = load_history()   # persisted history

# city selector (pre‑filled with guess)
cities = sorted({p.removesuffix(".pdf") for p in os.listdir(DATA_DIR) if p.endswith(".pdf")})
guessed = get_user_city_guess()
city_requested = st.selectbox("Select city", cities,
                              index = cities.index(guessed) if guessed in cities else 0)

question = st.text_input("Ask your ordinance question")
ask = st.button("🔍 Answer")

if ask and question:
    # include city name inside the question for filtering
    combined_q = f"[city:{city_requested}] {question}"

    resp   = chain({"question": combined_q, "chat_history": st.session_state.history})
    docs   = [d for d in resp['source_documents']
              if d.metadata.get('city') == city_requested.lower()]

    if not docs:
        answer = "I'm sorry, I could not find a relevant ordinance addressing your question."
        st.error(answer)
    else:
        quoted = docs[0].page_content.strip()
        surround = "\n\n".join(d.page_content.strip() for d in docs[1:])

        st.markdown(f"### 🎯 Answer\n> {quoted}")

        stoggle("Show additional context",
                surround or "_No further context available._",)

        # persistent history
        st.session_state.history.append((question, quoted))
        save_history(st.session_state.history)

