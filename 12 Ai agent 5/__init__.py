"""
RAG Agent Package

A modular RAG (Retrieval-Augmented Generation) agent system for document-based question answering.

Modules:
- config: Configuration constants and settings
- document_processor: PDF document loading and vector store creation
- agent_visualizer: Process flow visualization
- rag_agent_core: Main RAG agent implementation
- utils: Utility functions and helpers
- rag_agent_main: Main application entry point
"""

from .document_processor import DocumentProcessor
from .rag_agent_core import RAGAgent
from .agent_visualizer import AgentVisualizer
from .utils import create_agent_visualization, test_visualization

__version__ = "1.0.0"
__author__ = "RAG Agent Team"

__all__ = [
    "DocumentProcessor",
    "RAGAgent", 
    "AgentVisualizer",
    "create_agent_visualization",
    "test_visualization"
]
