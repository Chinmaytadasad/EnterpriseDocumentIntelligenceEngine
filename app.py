import os
import streamlit as str_ui
from rag_engine import get_rag_components

# Web layout styling parameters
str_ui.set_page_config(page_title="Enterprise Intel Engine", layout="wide")
str_ui.title("🏢 Enterprise Document Intelligence Engine")
str_ui.caption("FDE Portfolio Project | Local RAG Infrastructure Connected to Production PGVector")

@str_ui.cache_resource
def initialize_system():
    """Initializes connections to the active PGVector database container engine."""
    vector_store, retriever, rag_chain = get_rag_components()
    return vector_store, retriever, rag_chain

vector_store, retriever, rag_chain = initialize_system()

if "chat_history" not in str_ui.session_state:
    str_ui.session_state.chat_history = []

with str_ui.sidebar:
    str_ui.header("⚙️ Container Monitoring")
    str_ui.success("Web Server Node: Containerized")
    str_ui.success("Database Node: Live (PGVector)")
    str_ui.info("Inference Gateway: Host Gateway")
    
    if str_ui.button("Clear Conversation History"):
        str_ui.session_state.chat_history = []
        str_ui.rerun()

for message in str_ui.session_state.chat_history:
    with str_ui.chat_message(message["role"]):
        str_ui.markdown(message["content"])

if user_query := str_ui.chat_input("Ask a question about your knowledge base..."):
    with str_ui.chat_message("user"):
        str_ui.markdown(user_query)
    str_ui.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with str_ui.chat_message("assistant"):
        with str_ui.spinner("Querying PGVector Container Tables..."):
            try:
                result = rag_chain.invoke(user_query)
                matched_docs = result["context_docs"]
                ai_response = result["answer"]
                str_ui.markdown(ai_response)
                
                if matched_docs:
                    with str_ui.expander("🔍 View PostgreSQL Verified Source Citations"):
                        for i, doc in enumerate(matched_docs):
                            source_name = doc.metadata.get('source', 'Database Matrix Record')
                            chunk_id = doc.metadata.get('chunk_id', 'N/A')
                            doc_type = doc.metadata.get('type', 'document')
                            
                            citation = f"**Source [{i+1}] ({doc_type}):** `{os.path.basename(source_name)}` (Chunk {chunk_id})"
                            if doc_type == 'csv':
                                row_id = doc.metadata.get('row_id', 'N/A')
                                citation += f" (Row {row_id})"
                            elif doc_type == 'pdf':
                                page_number = doc.metadata.get('page_number', 'N/A')
                                citation += f" (Page {page_number})"
                                
                            str_ui.markdown(citation)
            except Exception as e:
                str_ui.error(f"Execution Failure: {str(e)}")
