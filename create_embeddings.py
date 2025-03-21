import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

loader = TextLoader("ordinances.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Save to FAISS Vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ Embeddings created and stored!")
