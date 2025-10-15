"""
Document processing module for the RAG Agent.

This module handles PDF document loading, chunking, and vector store creation.
"""

import os
from typing import List, Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    PDF_FILENAME, PERSIST_DIRECTORY, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K
)


class DocumentProcessor:
    """Handles PDF document loading, chunking, and vector store creation."""
    
    def __init__(self, pdf_path: str = None, persist_directory: str = None, collection_name: str = None):
        """Initialize the document processor.
        
        Args:
            pdf_path: Path to the PDF file. Defaults to config value.
            persist_directory: Directory to persist the vector store. Defaults to config value.
            collection_name: Name for the Chroma collection. Defaults to config value.
        """
        self.pdf_path = pdf_path or PDF_FILENAME
        self.persist_directory = persist_directory or PERSIST_DIRECTORY
        self.collection_name = collection_name or COLLECTION_NAME
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vectorstore = None
        self.retriever = None
    
    def _validate_pdf_exists(self) -> None:
        """Validate that the PDF file exists."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
    
    def _load_pdf_documents(self) -> List[Any]:
        """Load and validate PDF documents.
        
        Returns:
            List of loaded document pages.
        """
        self._validate_pdf_exists()
        
        pdf_loader = PyPDFLoader(self.pdf_path)
        try:
            pages = pdf_loader.load()
            print(f"PDF has been loaded and has {len(pages)} pages")
            return pages
        except Exception as e:
            print(f"Error loading PDF: {e}")
            raise
    
    def _create_persist_directory(self) -> None:
        """Create the persist directory if it doesn't exist."""
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
    
    def _create_vectorstore(self, documents: List[Any]) -> None:
        """Create the Chroma vector store from documents.
        
        Args:
            documents: List of document chunks to store.
        """
        self._create_persist_directory()
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print("Created ChromaDB vector store!")
        except Exception as e:
            print(f"Error setting up ChromaDB: {str(e)}")
            raise
    
    def _create_retriever(self) -> None:
        """Create the retriever from the vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call process_documents first.")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
    
    def process_documents(self) -> None:
        """Process PDF documents and create vector store with retriever."""
        pages = self._load_pdf_documents()
        chunked_documents = self.text_splitter.split_documents(pages)
        self._create_vectorstore(chunked_documents)
        self._create_retriever()
    
    def get_retriever(self):
        """Get the retriever instance.
        
        Returns:
            The retriever instance.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call process_documents first.")
        return self.retriever
