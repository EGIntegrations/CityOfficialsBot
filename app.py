import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

load_dotenv()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()

    if not os.path.exists("faiss_index"):
        docs = []
        pdf_folder = "ordinances/"
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf") and not pdf_file.startswith("._"):
                print(f"🔍 Attempting to load {pdf_file}")
                try:
                    loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
                    loaded_docs = loader.load()
                    city_name = pdf_file.replace(".pdf", "").lower()
                    for doc in loaded_docs:
                        doc.metadata['source_file'] = pdf_file
                        doc.metadata['timestamp'] = "January 1, 2024"
                        doc.metadata['category'] = city_name.capitalize()
                        doc.metadata['city'] = city_name
                    docs.extend(loaded_docs)
                except Exception as e:
                    print(f"⚠️ Skipping {pdf_file}: {e}")
                    continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    custom_template = """
    You are a chatbot assisting city officials.

    STRICT RULES:
    - Answer the user's question with the exact quoted ordinance text from the specified city.
    - Clearly and explicitly highlight ONLY the ordinance text that directly answers the question.
    - Provide additional surrounding context separately afterward, clearly marked.
    - Never provide interpretations or opinions.
    - If no relevant ordinance is found, respond exactly:
    "I'm sorry, I could not find a relevant ordinance addressing your question."

    Ordinances Context:
    {context}

    Question:
    {question}

    Answer format:
    HIGHLIGHTED ANSWER:
    "(Exact ordinance quotation answering the question)"

    ADDITIONAL CONTEXT:
    (Surrounding statements and additional relevant information)
    """

    prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

chain = load_chain()

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🏛️ City Officials Ordinance Reference Bot")
user_question = st.text_input("Ask a question about ordinances (mention city name clearly):")

if st.button("Ask"):
    if user_question:
        city_pattern = r"city of ([a-zA-Z\s]+)"
        city_match = re.search(city_pattern, user_question.lower())
        city_requested = city_match.group(1).strip() if city_match else None

        response = chain({"question": user_question, "chat_history": st.session_state.history})
        ordinance_sources = response['source_documents']

        matched_docs = [doc for doc in ordinance_sources if city_requested in doc.metadata.get('city', '').lower()] if city_requested else ordinance_sources

        if matched_docs:
            highlighted_answer = matched_docs[0].page_content.strip()
            context_docs = matched_docs[1:]
            context_text = "\n\n".join([doc.page_content.strip() for doc in context_docs])

            final_response = f"""
### 🎯 Highlighted Answer:
> "{highlighted_answer}"

---

### 📚 Additional Context:
{context_text if context_text else "No additional context available."}

---

- **Timestamp:** {matched_docs[0].metadata.get('timestamp', 'N/A')}
- **City:** {matched_docs[0].metadata.get('category', 'N/A')}
- **Source File:** {matched_docs[0].metadata.get('source_file', 'N/A')}
"""
        else:
            final_response = "I'm sorry, I could not find a relevant ordinance addressing your question."

        st.markdown(final_response, unsafe_allow_html=True)
        st.session_state.history.append((user_question, final_response))
