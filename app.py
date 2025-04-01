import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

load_dotenv()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()

    if not os.path.exists("faiss_index"):
        docs = []
        pdf_folder = "ordinances/"
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    # Clearly add metadata for enhanced responses
                    doc.metadata['source_file'] = pdf_file
                    doc.metadata['timestamp'] = "January 1, 2024"  # Update dynamically if possible
                    doc.metadata['category'] = pdf_file.replace(".pdf", "")
                docs.extend(loaded_docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    custom_template = """
    You are a chatbot assisting city officials. Provide direct quotations from city ordinances along with their timestamp, category, and surrounding context.

    STRICT RULES:
    - ONLY directly quote provided ordinance text.
    - NEVER interpret, paraphrase, or give personal opinions.
    - Include in your answer:
      1. Exact ordinance quotation in quotation marks ("").
      2. Timestamp (ordinance creation date).
      3. Ordinance category.
      4. Relevant surrounding context if available.
    - If no relevant ordinance found, respond exactly:
      "I'm sorry, I could not find a relevant ordinance addressing your question."

    Ordinances Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

chain = load_chain()

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🏛️ City Officials Ordinance Reference Bot")
user_question = st.text_input("Ask a question about ordinances from multiple cities:")

if st.button("Ask"):
    if user_question:
        response = chain({"question": user_question, "chat_history": st.session_state.history})

        ordinance_sources = response['source_documents']
        final_response = ""
        for doc in ordinance_sources:
            text = doc.page_content.strip()
            timestamp = doc.metadata.get('timestamp', 'N/A')
            category = doc.metadata.get('category', 'N/A')
            source_file = doc.metadata.get('source_file', 'N/A')

            final_response += f"""
            Ordinance: "{text}"

            - **Timestamp:** {timestamp}
            - **Category:** {category}
            - **Source File:** {source_file}
            ---
            """

        st.write("🤖", final_response)
        st.session_state.history.append((user_question, final_response))
