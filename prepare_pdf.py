from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# Load PDF and extract text
loader = PyPDFLoader("ordinances.pdf")  # rename your PDF file here
documents = loader.load()

# Split into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Generate embeddings with OpenAI
embeddings = OpenAIEmbeddings()

# Create a vectorstore with FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ PDF processed and embeddings saved.")