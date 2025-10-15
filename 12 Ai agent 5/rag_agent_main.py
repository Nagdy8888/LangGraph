"""
Main RAG Agent application.

This is the main entry point for the RAG agent system. It orchestrates the document processing,
agent creation, and interactive session management.
"""

import sys
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from rag_agent_core import RAGAgent
from utils import create_agent_visualization, test_visualization


def create_rag_agent() -> RAGAgent:
    """Create and initialize a RAG agent with document processing.
    
    Returns:
        Initialized RAG agent ready for question answering.
    """
    document_processor = DocumentProcessor()
    document_processor.process_documents()
    
    return RAGAgent(document_processor)


def main() -> None:
    """Main function to run the RAG agent."""
    try:
        rag_agent = create_rag_agent()
        rag_agent.run_interactive_session()
    except Exception as e:
        print(f"Error initializing RAG agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "visualize":
            create_agent_visualization()
        elif sys.argv[1] == "test":
            test_visualization()
        else:
            print("Usage: python rag_agent_main.py [visualize|test]")
            print("  visualize - Create agent flow diagram")
            print("  test      - Test matplotlib functionality")
            print("  (no args) - Run interactive RAG agent")
    else:
        main()
