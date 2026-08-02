"""
medical_rag.py

Simple RAG retrieval scaffolding for PubMed (NCBI Entrez) and a local MedQA sqlite store.
This module provides safe, auditable retrieval points and does NOT perform automatic crawling.
"""
import os
import sqlite3
import logging
from typing import List, Tuple, Optional

try:
    from Bio import Entrez
    BIOPYTHON = True
except Exception:
    BIOPYTHON = False

class PubMedClient:
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        self.email = email or os.environ.get("NCBI_EMAIL")
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        if BIOPYTHON and self.email:
            Entrez.email = self.email
            if self.api_key:
                Entrez.api_key = self.api_key

    def fetch_abstracts(self, query: str, max_results: int = 5) -> List[Tuple[str,str]]:
        if not BIOPYTHON:
            logging.warning("Biopython Entrez not available. Install biopython to enable PubMed retrieval.")
            return []
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        results = []
        if ids:
            fetch = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
            text = fetch.read()
            for pid in ids:
                results.append((pid, text))
        return results

class MedQADB:
    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"MedQA DB not found: {db_path}")
        self.db_path = db_path

    def query(self, question: str, limit: int = 5) -> List[Tuple[str,str]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, answer FROM medqa WHERE question LIKE ? LIMIT ?", (f"%{question}%", limit))
        rows = cur.fetchall()
        conn.close()
        return rows

class Retriever:
    def __init__(self, medqa_db: Optional[str] = None, pubmed_client: Optional[PubMedClient] = None):
        self.pubmed = pubmed_client or PubMedClient()
        self.medqa = MedQADB(medqa_db) if medqa_db else None

    def retrieve(self, query: str, top_k: int = 5):
        results = []
        if self.medqa:
            try:
                results.extend([("medqa", r[0], r[1]) for r in self.medqa.query(query, limit=top_k)])
            except Exception as e:
                logging.warning(f"MedQA query failed: {e}")
        try:
            pm = self.pubmed.fetch_abstracts(query, max_results=top_k)
            results.extend([("pubmed", pid, txt) for pid, txt in pm])
        except Exception as e:
            logging.warning(f"PubMed fetch failed: {e}")
        return results[:top_k]

if __name__ == "__main__":
    print("medical_rag module ready. Configure NCBI credentials and MedQA DB to use.")
