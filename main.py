# ===============================================
# 1. Imports
# ===============================================
import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document

# ===============================================
# 2. Load environment variables
# ===============================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not PINECONE_API_KEY or not HF_TOKEN:
    raise ValueError("Missing API Keys. Please check your .env file.")

# ===============================================
# 3. PDF Loader
# ===============================================
def load_pdfs(pdf_dir: str) -> List[Document]:
    """Load all PDFs from a directory into LangChain Documents"""
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Directory '{pdf_dir}' does not exist.")
        
    loader = DirectoryLoader(
        pdf_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# ===============================================
# 4. Minimal Document Filter
# ===============================================
def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only page_content and source metadata for each document"""
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(
            page_content=doc.page_content,
            metadata={"source": src}
        ))
    return minimal_docs

# ===============================================
# 5. Split text into chunks
# ===============================================
def split_docs(docs: List[Document], chunk_size=500, overlap=20) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)

# ===============================================
# 6. Embeddings
# ===============================================
def load_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

# ===============================================
# 7. Pinecone Setup
# ===============================================
def setup_pinecone(index_name="medibot", dimension=384):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

# ===============================================
# 8. Create Vector Store
# ===============================================
def create_vector_store(docs: List[Document], embeddings, index_name="medibot"):
    return PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )

# ===============================================
# 9. LLM Setup (HuggingFace free model)
# ===============================================
def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base", 
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

# ===============================================
# 10. RAG Chain Setup
# ===============================================
def build_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    system_prompt = (
        "You are a helpful medical assistant. "
        "Use the following context to answer questions in 3 sentences or less. "
        "If unsure, say 'I don't know'.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# ===============================================
# 11. MAIN PIPELINE
# ===============================================
if __name__ == "__main__":
    # --- DATA INGESTION (Run once or when PDF changes) ---
    print("Loading PDFs...")
    try:
        docs = load_pdfs("./Data")
        if not docs:
            print("No PDFs found in ./Data folder.")
        else:
            print(f"Loaded {len(docs)} pages from PDFs.")
            
            minimal_docs = filter_minimal_docs(docs)
            chunks = split_docs(minimal_docs)
            print(f"Split into {len(chunks)} chunks.")

            embeddings = load_embeddings()
            index = setup_pinecone()
            
            print("Upserting vectors to Pinecone...")
            vector_store = create_vector_store(chunks, embeddings)
            
            # --- RETRIEVAL ---
            llm = load_llm()
            rag_chain = build_rag_chain(vector_store, llm)

            question = "What is Acne?"
            print(f"Thinking about: {question}...")
            response = rag_chain.invoke({"input": question})
            
            print("\n=== ANSWER ===")
            print(response["answer"])
            
    except Exception as e:
        print(f"An error occurred: {e}")
