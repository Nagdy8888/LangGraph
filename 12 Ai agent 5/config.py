"""
Configuration constants for the RAG Agent system.

This module contains all configuration constants used across the RAG agent components.
"""

# Model Configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5

# File and Directory Paths
PDF_FILENAME = "Stock_Market_Performance_2024.pdf"
PERSIST_DIRECTORY = r"C:\Nagdy\Mustafa\DataCamp\LangGraph\12 Ai agent 5"
COLLECTION_NAME = "stock_market"

# UI Messages
WELCOME_MESSAGE = "\n=== RAG AGENT ==="
EXIT_COMMANDS = ['exit', 'quit']
QUESTION_PROMPT = "\nWhat is your question: "
ANSWER_HEADER = "\n=== ANSWER ==="
TOOL_CALL_PREFIX = "Calling Tool:"
TOOL_RESULT_LENGTH_PREFIX = "Result length:"
TOOL_EXECUTION_COMPLETE = "Tools Execution Complete. Back to the model!"
TOOL_NOT_FOUND_ERROR = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

# Visualization Configuration
GRAPH_WIDTH = 12
GRAPH_HEIGHT = 8
NODE_COLORS = {
    'llm': '#4CAF50',      # Green
    'retriever_agent': '#2196F3',  # Blue
    'start': '#FF9800',    # Orange
    'end': '#F44336'       # Red
}
EDGE_COLORS = {
    'normal': '#666666',   # Gray
    'conditional': '#9C27B0'  # Purple
}

# System Prompt
SYSTEM_PROMPT = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 
based on the PDF document loaded into your knowledge base.

Use the retriever tool available to answer questions about the stock market performance data. 
You can make multiple calls if needed.

If you need to look up some information before asking a follow up question, you are allowed to do that!

Please always cite the specific parts of the documents you use in your answers.
"""
