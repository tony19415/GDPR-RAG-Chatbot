from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from get_embedding_function import get_embedding_function
from ollama import chat
import streamlit as st

#load the pdf
docs = PyPDFLoader("data\CELEX_32016R0679_EN_TXT.pdf").load()

#split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1500)
chunks = splitter.split_documents(docs)

#embed the chunks and store them in the chroma db
embeddings = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

#retrieve most relevant chunks from vectorstore
def retrieve(query:str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def generate_answer(query:str, context:str) -> str:
    response = chat(
        model = "llama3.2:1b",
        messages = [
            {"role": "user", "content": f"Answer the question based on the context provided. \n\nContext: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response["message"]["content"]

st.title("GDPR Regulations RAG with Ollama")

user_query = st.chat_input("Ask a question about GDPR regulations:")

if user_query:
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(user_query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(user_query, context)
    
    st.write("**Answer: ", answer)