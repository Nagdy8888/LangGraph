# LangGraph Course Summary

## 📚 Course Overview

This repository contains a comprehensive collection of LangGraph implementations and AI agent projects developed as part of the **LangGraph Course** by **freeCodeCamp.org**.

### 🎓 Course Credits

- **Institution**: [freeCodeCamp.org](https://www.freecodecamp.org/)
- **Instructor**: [Vaibhav Mehra](https://www.linkedin.com/in/vaibhav-mehra-ai/)
- **Course**: LangGraph - Building AI Agents with LangChain
- **Platform**: YouTube / freeCodeCamp.org

### 🙏 Acknowledgments

Special thanks to:
- **freeCodeCamp.org** for providing high-quality, free educational content
- **Vaibhav Mehra** for his excellent teaching and comprehensive course material
- The open-source community for the amazing tools and libraries used in this course

---

## 🏗️ Course Structure

### **01 Type Annotations**
- `01_Typed_Dictionary.ipynb` - Understanding TypedDict for type safety
- `02_union.ipynb` - Working with Union types
- `03_Optional.ipynb` - Handling Optional types
- `04_Any.ipynb` - Using Any type annotations
- `05_Lambda_Function.ipynb` - Lambda functions in Python

### **02 Elements**
- `01_state.ipynb` - LangGraph state management
- `02_Nodes.ipynb` - Creating and managing nodes
- `03_Graph.ipynb` - Building graphs with LangGraph
- `04_Edges.ipynb` - Connecting nodes with edges
- `05_Conditional_Edges.ipynb` - Conditional routing
- `06_Start.ipynb` - Entry points in graphs
- `07_End.ipynb` - Exit points and termination
- `08_Tools.ipynb` - Tool integration
- `09_ToolNode.ipynb` - Tool execution nodes
- `10_StateGraph.ipynb` - State management in graphs
- `11-Runnable.ipynb` - Runnable interfaces
- `12_Messages.ipynb` - Message handling
- `13_Complete_LangGraph_Example.ipynb` - End-to-end example

### **03 Agent 1**
- `Hello_world.ipynb` - First LangGraph agent
- `exercise_1.ipynb` - Basic agent exercises

### **04 Agent 2**
- `exercise_2.ipynb` - Multi-input agent
- `multiple_inputs.ipynb` - Handling multiple inputs

### **05 Agent 3**
- `exercise_3.ipynb` - Sequential agent patterns
- `sequential_agent.ipynb` - Sequential processing

### **06 Agent 4**
- `conditional_agent.ipynb` - Conditional logic in agents
- `exercise_4.ipynb` - Conditional agent exercises

### **07 Agent 5**
- `exercise.ipynb` - Looping patterns
- `loopin.ipynb` - Loop implementation

### **08 AI Agent 1**
- `bot.py` - Simple chatbot implementation
- `simple_bot.ipynb` - Basic chatbot with LangGraph

### **09 AI Agent 2**
- `Chatbot_memory.py` - Chatbot with memory
- `conversation_history.txt` - Conversation history storage

### **10 AI Agent 3**
- `ReAct_agent.py` - ReAct (Reasoning and Acting) agent with tool execution

### **11 AI Agent 4**
- `drafter.py` - Document drafting agent with clean code principles

### **12 AI Agent 5**
- `rag_agent.py` - RAG (Retrieval-Augmented Generation) agent
- `rag_agent_flow.png` - Process flow visualization
- **Modular Components**:
  - `config.py` - Configuration management
  - `document_processor.py` - PDF document processing
  - `agent_visualizer.py` - Process flow visualization
  - `rag_agent_core.py` - Core RAG agent logic
  - `rag_agent_main.py` - Main application entry point
  - `utils.py` - Utility functions
  - `README.md` - Detailed documentation
  - `__init__.py` - Package initialization

---

## 🛠️ Technologies Learned

### **Core Technologies**
- **LangGraph** - Building AI agent workflows
- **LangChain** - LLM application framework
- **OpenAI GPT-4** - Large language model integration
- **Python** - Programming language and best practices

### **Advanced Features**
- **Vector Databases** - ChromaDB for document retrieval
- **Document Processing** - PDF loading and chunking
- **Embeddings** - Text vectorization with OpenAI
- **RAG Systems** - Retrieval-Augmented Generation
- **Visualization** - Matplotlib for process flow diagrams

### **Clean Code Principles**
- **SOLID Principles** - Single Responsibility, Open/Closed, etc.
- **Modular Design** - Separation of concerns
- **Type Hints** - Python type annotations
- **Documentation** - Comprehensive docstrings
- **Error Handling** - Robust exception management

---

## 🎯 Key Learning Outcomes

### **1. LangGraph Fundamentals**
- Understanding state management in AI agents
- Creating and connecting nodes in agent workflows
- Implementing conditional logic and routing
- Building reusable agent components

### **2. AI Agent Development**
- **ReAct Agents**: Reasoning and Acting patterns
- **RAG Systems**: Document-based question answering
- **Memory Management**: Conversation history and context
- **Tool Integration**: External API and function calling

### **3. Advanced Concepts**
- **Vector Search**: Semantic document retrieval
- **Document Processing**: PDF parsing and chunking
- **Process Visualization**: Creating flow diagrams
- **Clean Architecture**: Professional code organization

### **4. Real-World Applications**
- **Document Q&A**: Stock market analysis system
- **Interactive Agents**: Conversational AI systems
- **Tool Execution**: Mathematical operations and file handling
- **Process Monitoring**: Visual workflow representation

---

## 📊 Project Highlights

### **ReAct Agent (10 AI Agent 3)**
- Implements mathematical operations (add, subtract, multiply)
- Demonstrates tool calling and execution
- Shows reasoning and acting patterns
- Clean error handling and user interaction

### **Document Drafter (11 AI Agent 4)**
- Document creation and editing capabilities
- File saving and management
- Clean code architecture with proper separation
- Interactive user interface

### **RAG Agent (12 AI Agent 5)**
- **Document Processing**: PDF loading and vectorization
- **Semantic Search**: Intelligent document retrieval
- **Process Visualization**: Interactive flow diagrams
- **Modular Architecture**: Professional code organization
- **Clean Code**: SOLID principles and best practices

---

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Environment Setup**
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### **Running the Agents**

**ReAct Agent:**
```bash
python "10 Ai agent 3/ReAct_agent.py"
```

**Document Drafter:**
```bash
python "11 Ai agent 4/drafter.py"
```

**RAG Agent:**
```bash
# Monolithic version
python "12 Ai agent 5/rag_agent.py"

# Modular version
python "12 Ai agent 5/rag_agent_main.py"

# Create visualization
python "12 Ai agent 5/rag_agent.py" visualize
```

---

## 📚 Resources

### **Course Materials**
- [freeCodeCamp.org LangGraph Course](https://www.freecodecamp.org/)
- [Vaibhav Mehra's LinkedIn](https://www.linkedin.com/in/vaibhav-mehra-ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### **Dependencies**
- `langgraph` - Agent workflow framework
- `langchain` - LLM application framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools
- `chromadb` - Vector database
- `matplotlib` - Visualization
- `pypdf` - PDF processing

---

## 🏆 Course Completion

This repository represents a comprehensive journey through LangGraph development, from basic concepts to advanced AI agent implementations. The projects demonstrate:

- **Progressive Learning**: From simple agents to complex RAG systems
- **Clean Code**: Professional development practices
- **Real-World Applications**: Practical, deployable solutions
- **Visual Learning**: Process flow diagrams and documentation
- **Modular Design**: Scalable and maintainable architecture

---

## 📝 License

This educational project is created for learning purposes. All course materials and implementations are based on the freeCodeCamp.org LangGraph course by Vaibhav Mehra.

---

## 🤝 Contributing

This repository serves as a learning resource and portfolio of LangGraph implementations. Feel free to explore, learn, and build upon these examples.

---

**Happy Learning! 🎉**

*Built with ❤️ following the excellent freeCodeCamp.org LangGraph course by Vaibhav Mehra*
