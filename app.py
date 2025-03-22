import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()

    if not os.path.exists("faiss_index"):
        # Generate embeddings dynamically from PDF
        loader = PyPDFLoader("ordinances.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    custom_template = """
    You are an Ozark City Ordinances chatbot. Answer ONLY questions about city ordinances. 
    Politely refuse to answer unrelated topics. Keep responses accurate, short, and specific.
    Every time you respond with an answer about a city ordinance, provide the exact code where users can find where you found the information.
    NEVER respond with your own opinion, nor should you ever respond with what you think someone should do. 
    If the ordinances do not contain information clearly answering the question, respond exactly with:
    "I'm sorry, I could not find a specific ordinance addressing your question. Please consult city officials directly for clarification at 417-581-2407."
    Clearly mark all quoted text in quotation marks "".
    DO NOT paraphrase or summarize. 

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    custom_prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={'prompt': custom_prompt}
    )

chain = load_chain()

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("📖 Ozark City Ordinances Assistant")
user_question = st.text_input("Ask a question about city ordinances:")

if st.button("Ask"):
    if user_question:
        response = chain({"question": user_question, "chat_history": st.session_state.history})
        st.write("🤖", response["answer"])
        st.session_state.history.append((user_question, response["answer"]))
