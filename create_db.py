from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

base_directory = "base"

def create_db():
    documents = load_documents()
    chunks = split_chunks(documents)
    vetorize_chunks(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(base_directory, glob="*.pdf")
    documents = loader.load()
    return documents

def split_chunks(documents):
    split_documents = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = split_documents.split_documents(documents)
    print(f"Número de chunks criados: {len(chunks)}")
    return chunks

def vetorize_chunks(chunks):
    db= Chroma.from_documents(chunks, OllamaEmbeddings(model="mxbai-embed-large"), persist_directory="db")
    print("Banco de Dados criado com sucesso!")

create_db()