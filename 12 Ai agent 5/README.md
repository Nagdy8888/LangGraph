# RAG Agent System

A modular RAG (Retrieval-Augmented Generation) agent system for document-based question answering with process flow visualization.

## Project Structure

```
12 Ai agent 5/
├── __init__.py              # Package initialization
├── config.py                # Configuration constants
├── document_processor.py    # PDF document processing
├── agent_visualizer.py      # Process flow visualization
├── rag_agent_core.py        # Core RAG agent implementation
├── utils.py                 # Utility functions
├── rag_agent_main.py        # Main application entry point
├── README.md                # This file
└── requirements.txt         # Dependencies
```

## Modules

### `config.py`
Contains all configuration constants, including:
- Model settings (GPT-4, embeddings)
- Document processing parameters
- UI messages and prompts
- Visualization settings

### `document_processor.py`
Handles PDF document processing:
- PDF loading and validation
- Text chunking with RecursiveCharacterTextSplitter
- ChromaDB vector store creation
- Retriever setup

### `agent_visualizer.py`
Creates visual diagrams of the agent process flow:
- Node-based flow diagrams
- Color-coded components
- Interactive matplotlib visualizations
- Process descriptions

### `rag_agent_core.py`
Core RAG agent implementation:
- LangGraph state management
- Tool creation and execution
- LLM interaction handling
- Interactive session management

### `utils.py`
Utility functions:
- Visualization helpers
- Matplotlib testing
- Error handling utilities

### `rag_agent_main.py`
Main application entry point:
- Command-line interface
- Agent initialization
- Session orchestration

## Usage

### Run Interactive Agent
```bash
python rag_agent_main.py
```

### Create Process Flow Diagram
```bash
python rag_agent_main.py visualize
```

### Test Matplotlib
```bash
python rag_agent_main.py test
```

### Programmatic Usage
```python
from document_processor import DocumentProcessor
from rag_agent_core import RAGAgent

# Initialize document processor
doc_processor = DocumentProcessor()
doc_processor.process_documents()

# Create RAG agent
agent = RAGAgent(doc_processor)

# Ask questions
answer = agent.ask_question("What was the stock market performance in 2024?")

# Create visualization
agent.create_flow_diagram("my_flow.png")
```

## Features

- **Modular Design**: Clean separation of concerns
- **Document Processing**: PDF loading and vector store creation
- **RAG Implementation**: Retrieval-augmented generation with LangGraph
- **Process Visualization**: Interactive flow diagrams
- **Error Handling**: Robust error management
- **Clean Code**: Following SOLID principles and clean code practices

## Dependencies

- langgraph
- langchain
- langchain-openai
- langchain-community
- chromadb
- matplotlib
- pandas
- numpy
- pypdf

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Update `config.py` to modify:
- Model settings
- File paths
- UI messages
- Visualization parameters
