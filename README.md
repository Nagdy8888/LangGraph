# LangGraph AI Agents Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.8-green.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange.svg)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-red.svg)](https://openai.com)

A comprehensive collection of AI agents built with LangGraph, featuring ReAct agents, document drafting, and RAG (Retrieval-Augmented Generation) systems. This repository contains implementations from the freeCodeCamp.org LangGraph course by Vaibhav Mehra.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
```bash
# Clone the repository
git clone https://github.com/Nagdy8888/LangGraph.git
cd LangGraph

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Run Agents
```bash
# ReAct Agent (Math Operations)
python "10 Ai agent 3/ReAct_agent.py"

# Document Drafter
python "11 Ai agent 4/drafter.py"

# RAG Agent
python "12 Ai agent 5/rag_agent.py"

# Create process visualization
python "12 Ai agent 5/rag_agent.py" visualize
```

## 📁 Repository Structure

```
LangGraph/
├── 01 Type Annotations/          # Python type system fundamentals
├── 02 Elements/                  # LangGraph core concepts
├── 03 Agent 1/                   # Basic agent implementations
├── 04 Agent 2/                   # Multi-input agents
├── 05 Agent 3/                   # Sequential agents
├── 06 Agent 4/                   # Conditional agents
├── 07 Agent 5/                   # Looping patterns
├── 08 AI agent 1/                # Simple chatbots
├── 09 AI agent 2/                # Memory-enabled chatbots
├── 10 Ai agent 3/                # ReAct agents with tools
├── 11 Ai agent 4/                # Document drafting agents
├── 12 Ai agent 5/                # RAG agents with visualization
├── requirements.txt              # Python dependencies
├── Stock_Market_Performance_2024.pdf  # Sample document for RAG
├── COURSE_SUMMARY.md             # Detailed course overview
└── README.md                     # This file
```

## 🤖 Featured AI Agents

### 1. ReAct Agent (10 Ai agent 3)
**Reasoning and Acting** agent that performs mathematical operations.

**Features:**
- Tool calling and execution
- Mathematical operations (add, subtract, multiply)
- Interactive user interface
- Error handling and validation

**Usage:**
```bash
python "10 Ai agent 3/ReAct_agent.py"
```

### 2. Document Drafter (11 Ai agent 4)
**Document management** agent for creating and editing text documents.

**Features:**
- Document creation and editing
- File saving and management
- Clean code architecture
- Interactive conversation flow

**Usage:**
```bash
python "11 Ai agent 4/drafter.py"
```

### 3. RAG Agent (12 Ai agent 5)
**Retrieval-Augmented Generation** agent for document-based question answering.

**Features:**
- PDF document processing
- Vector database integration (ChromaDB)
- Semantic document search
- Process flow visualization
- Modular architecture
- Clean code principles

**Usage:**
```bash
# Monolithic version
python "12 Ai agent 5/rag_agent.py"

# Modular version
python "12 Ai agent 5/rag_agent_main.py"

# Create visualization
python "12 Ai agent 5/rag_agent.py" visualize
```

## 🛠️ Technologies Used

### Core Framework
- **LangGraph** - Agent workflow framework
- **LangChain** - LLM application framework
- **OpenAI GPT-4** - Large language model

### Document Processing
- **ChromaDB** - Vector database
- **PyPDF** - PDF document parsing
- **RecursiveCharacterTextSplitter** - Text chunking

### Visualization
- **Matplotlib** - Process flow diagrams
- **NumPy** - Numerical operations

### Development
- **Python 3.8+** - Programming language
- **Type Hints** - Type safety
- **Clean Code** - SOLID principles

## 📚 Learning Path

### Beginner Level
1. **Type Annotations** - Python type system
2. **LangGraph Elements** - Core concepts
3. **Basic Agents** - Simple implementations

### Intermediate Level
4. **Multi-input Agents** - Complex inputs
5. **Sequential Agents** - Step-by-step processing
6. **Conditional Agents** - Decision making
7. **Looping Patterns** - Iterative processes

### Advanced Level
8. **Memory Chatbots** - Conversation history
9. **Tool Integration** - External functions
10. **Document Management** - File operations
11. **RAG Systems** - Document-based Q&A

## 🎯 Key Features

### Clean Code Architecture
- **SOLID Principles** - Single responsibility, open/closed, etc.
- **Modular Design** - Separation of concerns
- **Type Safety** - Comprehensive type hints
- **Error Handling** - Robust exception management
- **Documentation** - Detailed docstrings

### Advanced Capabilities
- **Vector Search** - Semantic document retrieval
- **Process Visualization** - Interactive flow diagrams
- **Memory Management** - Conversation context
- **Tool Execution** - External API integration
- **Document Processing** - PDF parsing and chunking

## 📊 Project Highlights

### RAG Agent Features
- **Document Processing**: PDF loading and vectorization
- **Semantic Search**: Intelligent document retrieval
- **Process Visualization**: Interactive flow diagrams
- **Modular Architecture**: Professional code organization
- **Clean Code**: SOLID principles and best practices

### Visualization Capabilities
- **Process Flow Diagrams**: Visual representation of agent workflows
- **Interactive Charts**: Matplotlib-based visualizations
- **Color-coded Components**: Easy-to-understand diagrams
- **Professional Styling**: Publication-ready graphics

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nagdy8888/LangGraph.git
cd LangGraph
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment
Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 4. Run Your First Agent
```bash
python "10 Ai agent 3/ReAct_agent.py"
```

## 📖 Documentation

- **[Course Summary](COURSE_SUMMARY.md)** - Detailed course overview
- **[RAG Agent README](12%20Ai%20agent%205/README.md)** - RAG agent documentation
- **[Jupyter Notebooks](01%20Type%20Annotations/)** - Interactive tutorials

## 🤝 Contributing

This repository serves as a learning resource and portfolio of LangGraph implementations. Feel free to:

- Explore the code and learn from examples
- Fork the repository for your own projects
- Submit issues or suggestions
- Share your own LangGraph implementations

## 📄 License

This educational project is created for learning purposes. All course materials and implementations are based on the freeCodeCamp.org LangGraph course by Vaibhav Mehra.

## 🙏 Acknowledgments

- **[freeCodeCamp.org](https://www.freecodecamp.org/)** - For providing excellent educational content
- **[Vaibhav Mehra](https://www.linkedin.com/in/vaibhav-mehra-ai/)** - For the comprehensive LangGraph course
- **Open Source Community** - For the amazing tools and libraries

## 📞 Contact

- **GitHub**: [@Nagdy8888](https://github.com/Nagdy8888)
- **Repository**: [LangGraph Repository](https://github.com/Nagdy8888/LangGraph)

---

**Happy Learning! 🎉**

*Built with ❤️ following the excellent freeCodeCamp.org LangGraph course by Vaibhav Mehra*

---

## 🔗 Quick Links

- [Course Summary](COURSE_SUMMARY.md)
- [RAG Agent Documentation](12%20Ai%20agent%205/README.md)
- [freeCodeCamp.org](https://www.freecodecamp.org/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Vaibhav Mehra's LinkedIn](https://www.linkedin.com/in/vaibhav-mehra-ai/)
