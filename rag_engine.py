import os
import re
import time
import warnings
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Database connection credentials
DB_USER = os.getenv("DB_USER", "fde_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "enterprise_secure_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")  # Falls back to localhost if running outside container
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "knowledge_base")

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "enterprise_knowledge_base"

# Ollama connection
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Initializes and returns the LangChain wrapper for local SentenceTransformers."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={'trust_remote_code': True}
    )

def get_vector_store():
    """Initializes and returns the PGVector connection."""
    embeddings = get_embeddings()
    
    # Establish a reliable driver connection loop to handle database initialization delays
    for retry in range(5):
        try:
            vector_store = PGVector(
                embeddings=embeddings,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
                use_jsonb=True
            )
            return vector_store
        except Exception:
            time.sleep(3)
            
    # Last attempt without try-except to raise the actual error if it fails
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True
    )

def get_llm():
    """Initializes and returns the Ollama LLM Engine."""
    try:
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.0,
            base_url=OLLAMA_HOST,
            num_ctx=4096
        )
        llm.invoke("ping")  # force a real connection check now, not mid-chain later
        return llm
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to local Ollama inference gateway at {OLLAMA_HOST}: {e}"
        ) from e

def get_rag_components():
    """Returns initialized vector_store, retriever, and the RAG execution chain."""
    vector_store = get_vector_store()
    llm_engine = get_llm()

    def extract_ticket_id(question: str):
        match = re.search(r"\b(?:ticket\s*(?:#|id)?\s*|ticket\s+id\s+)\(?(?P<ticket_id>\d+)\)?\b", question, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group("ticket_id"))
        except ValueError:
            return None

    def retrieve_context(query: str):
        ticket_id = extract_ticket_id(query)
        if ticket_id is not None:
            exact_matches = vector_store.similarity_search(
                query=query,
                k=5,
                filter={"ticket_id": ticket_id}
            )
            if exact_matches:
                return exact_matches
        return vector_store.similarity_search(query=query, k=3)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template("""
You are an expert enterprise systems engineer and support analyst.
Answer the user's question accurately using ONLY the provided verified context fragments below.
If the answer cannot be found in the context, say that the information is missing from the ingested logs or files.

Context:
{context}

Question:
{question}

Helpful Answer with Explicit Source References:
""")

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    from langchain_core.runnables import RunnableParallel

    rag_chain = RunnableParallel(
        {"context_docs": RunnableLambda(retrieve_context), "question": RunnablePassthrough()}
    ) | RunnablePassthrough.assign(
        answer=(
            {"context": lambda x: format_docs(x["context_docs"]), "question": lambda x: x["question"]}
            | prompt_template
            | llm_engine
            | StrOutputParser()
        )
    )

    return vector_store, retriever, rag_chain
