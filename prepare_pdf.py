import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

docs = []
pdf_folder = "ordinances/"
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf") and not pdf_file.startswith("._"):
        print(f"🔍 Attempting to load {pdf_file}")
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            loaded_docs = loader.load()
        except Exception as e:
            print(f"⚠️ Skipping {pdf_file}: {e}")
            continue
        city_name = pdf_file.replace(".pdf", "").lower()
        for doc in loaded_docs:
            doc.metadata['source_file'] = pdf_file
            doc.metadata['timestamp'] = "January 1, 2024"  # Replace dynamically if available
            doc.metadata['category'] = city_name.capitalize()
            doc.metadata['city'] = city_name
        docs.extend(loaded_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ Embeddings successfully regenerated with metadata for all PDFs.")
