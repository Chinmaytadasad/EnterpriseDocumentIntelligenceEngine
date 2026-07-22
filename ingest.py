import os
import glob
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Import from our new rag_engine
from rag_engine import get_vector_store

# Configuration Settings
DATA_DIR = os.getenv("DATA_DIR", "./simulated_s3")

def process_csv(file_path):
    print(f"[+] Parsing CSV and formatting rows: {os.path.basename(file_path)}")
    docs = []
    try:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # Build a comprehensive textual representation of the row
            content = f"Ticket {row.get('Ticket ID', 'Unknown')}: Customer {row.get('Customer Name', 'Unknown')} regarding {row.get('Product Purchased', 'Unknown')}. "
            content += f"Subject: {row.get('Ticket Subject', 'Unknown')}. "
            content += f"Description: {row.get('Ticket Description', 'Unknown')}. "
            content += f"Status: {row.get('Ticket Status', 'Unknown')}."
            
            metadata = {
                "source": file_path,
                "row_id": index,
                "ticket_id": row.get('Ticket ID', 'Unknown'),
                "customer": row.get('Customer Name', 'Unknown'),
                "type": "csv"
            }
            docs.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        print(f"[!] Processing error on CSV {file_path}: {str(e)}")
    return docs

def load_and_parse_documents():
    """Scans directories for log streams, PDFs, and CSVs safely."""
    all_docs = []
    file_patterns = [
        os.path.join(DATA_DIR, "**", "*.log"),
        os.path.join(DATA_DIR, "**", "*.pdf"),
        os.path.join(DATA_DIR, "**", "*.csv")
    ]
    
    found_files = []
    for pattern in file_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
        
    print(f"[*] Discovered {len(found_files)} staging records for analysis...")
    
    for file_path in found_files:
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".log":
                print(f"[+] Parsing log stream and optimizing duplicates: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                # Deduplication speeds up script calculation time significantly
                unique_lines = list(set(lines))
                content = "".join(unique_lines[:800])  # Process up to 800 unique operational lines
                all_docs.append(Document(page_content=content, metadata={"source": file_path, "type": "log"}))
                
            elif ext == ".pdf":
                print(f"[+] Extracting document structures from PDF: {os.path.basename(file_path)}")
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                for i, doc in enumerate(pdf_docs):
                    doc.metadata["page_number"] = i + 1
                    doc.metadata["type"] = "pdf"
                all_docs.extend(pdf_docs)
                
            elif ext == ".csv":
                all_docs.extend(process_csv(file_path))
                
        except Exception as e:
            print(f"[!] Processing error on file {file_path}: {str(e)}")

    # Standardized operational boundaries for clean parsing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    
    final_chunks = text_splitter.split_documents(all_docs)
    
    # Add chunk_id to metadata
    for i, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = i
        
    return final_chunks

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"[!] Target data path does not exist: {DATA_DIR}")
    else:
        extracted_chunks = load_and_parse_documents()
        if extracted_chunks:
            print(f"[*] Connecting to PGVector to insert {len(extracted_chunks)} chunks in batches...")
            try:
                vector_store = get_vector_store()
                batch_size = 200
                total_batches = (len(extracted_chunks) + batch_size - 1) // batch_size
                for i in range(0, len(extracted_chunks), batch_size):
                    batch = extracted_chunks[i:i+batch_size]
                    print(f"[-] Inserting batch {i//batch_size + 1}/{total_batches} (Chunks {i} to {i+len(batch)})...")
                    vector_store.add_documents(batch)
                print(f"\n[✓] Successfully ingested {len(extracted_chunks)} chunks into PGVector.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\n[!] Failed to insert into PGVector: {str(e)}")
        else:
            print("[!] No documents were found or successfully parsed.")
