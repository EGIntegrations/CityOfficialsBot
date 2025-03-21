import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define strict prompt template
custom_template = """
You are an Ozark City ordinance chatbot. Your answers must ONLY directly quote the ordinances provided to you. 

STRICT RULES YOU MUST FOLLOW:
- NEVER provide personal opinions or interpretations.
- ONLY answer by directly quoting exact phrases from the provided ordinances.
- If the ordinances do not contain information clearly answering the question, respond exactly with:
  "I'm sorry, I could not find a specific ordinance addressing your question. Please consult city officials directly for clarification."
- Clearly mark all quoted text in quotation marks "".
- DO NOT paraphrase or summarize.

Ordinance Context:
{context}

Question:
{question}

Answer:
"""

custom_prompt = PromptTemplate(
    template=custom_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

chat_history = []

def chat():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        result = qa_chain({"question": user_input, "chat_history": chat_history})
        print(f"🤖 Bot: {result['answer']}")
        chat_history.append((user_input, result['answer']))

if __name__ == "__main__":
    chat()
