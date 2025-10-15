"""
RAG (Retrieval-Augmented Generation) Agent for Stock Market Performance Analysis.

This module provides a conversational AI agent that can answer questions about stock market
performance in 2024 by retrieving relevant information from a PDF document using vector search.
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add as add_messages

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5
MAX_RESULT_LENGTH_DISPLAY = 100

# File and directory paths
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

# Visualization constants
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

# System prompt
SYSTEM_PROMPT = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 
based on the PDF document loaded into your knowledge base.

Use the retriever tool available to answer questions about the stock market performance data. 
You can make multiple calls if needed.

If you need to look up some information before asking a follow up question, you are allowed to do that!

Please always cite the specific parts of the documents you use in your answers.
"""


class AgentState(TypedDict):
    """State structure for the RAG agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DocumentProcessor:
    """Handles PDF document loading, chunking, and vector store creation."""
    
    def __init__(self, pdf_path: str, persist_directory: str, collection_name: str):
        """Initialize the document processor.
        
        Args:
            pdf_path: Path to the PDF file.
            persist_directory: Directory to persist the vector store.
            collection_name: Name for the Chroma collection.
        """
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
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


class AgentVisualizer:
    """Handles visualization of the RAG agent's process flow."""
    
    def __init__(self):
        """Initialize the visualizer with matplotlib settings."""
        plt.style.use('default')
        self.fig = None
        self.ax = None
    
    def create_agent_flow_diagram(self, save_path: str = "rag_agent_flow.png") -> None:
        """Create a visual diagram of the RAG agent's process flow.
        
        Args:
            save_path: Path to save the generated diagram.
        """
        self.fig, self.ax = plt.subplots(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.axis('off')
        
        # Define node positions
        nodes = {
            'start': (1, 6),
            'llm': (3, 6),
            'retriever_agent': (7, 6),
            'end': (9, 6),
            'decision': (5, 4)
        }
        
        # Draw nodes
        self._draw_node('start', nodes['start'], 'START', NODE_COLORS['start'])
        self._draw_node('llm', nodes['llm'], 'LLM\nAgent', NODE_COLORS['llm'])
        self._draw_node('retriever_agent', nodes['retriever_agent'], 'Retriever\nAgent', NODE_COLORS['retriever_agent'])
        self._draw_node('end', nodes['end'], 'END', NODE_COLORS['end'])
        self._draw_decision_diamond(nodes['decision'])
        
        # Draw edges
        self._draw_arrow(nodes['start'], nodes['llm'], 'User Question')
        self._draw_conditional_arrow(nodes['llm'], nodes['decision'], 'Has Tool Calls?', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['decision'], nodes['retriever_agent'], 'Yes', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['decision'], nodes['end'], 'No', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['retriever_agent'], nodes['llm'], 'Tool Results')
        
        # Add title and description
        self.ax.text(5, 7.5, 'RAG Agent Process Flow', 
                    fontsize=16, fontweight='bold', ha='center')
        
        # Add legend
        self._add_legend()
        
        # Add process description
        self._add_process_description()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Agent flow diagram saved to: {save_path}")
        plt.show()
    
    def _draw_node(self, node_id: str, position: tuple, label: str, color: str) -> None:
        """Draw a rectangular node.
        
        Args:
            node_id: Unique identifier for the node.
            position: (x, y) coordinates for the node center.
            label: Text label for the node.
            color: Color of the node.
        """
        x, y = position
        width, height = 1.2, 0.8
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(rect)
        
        # Add label
        self.ax.text(x, y, label, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    def _draw_decision_diamond(self, position: tuple) -> None:
        """Draw a decision diamond.
        
        Args:
            position: (x, y) coordinates for the diamond center.
        """
        x, y = position
        size = 0.6
        
        # Create diamond shape
        diamond = patches.RegularPolygon(
            (x, y), 4, radius=size,
            orientation=np.pi/4,
            facecolor='#FFC107',
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(diamond)
        
        # Add label
        self.ax.text(x, y, 'Decision', ha='center', va='center', 
                    fontsize=9, fontweight='bold')
    
    def _draw_arrow(self, start: tuple, end: tuple, label: str = "", color: str = EDGE_COLORS['normal']) -> None:
        """Draw an arrow between two points.
        
        Args:
            start: Starting (x, y) coordinates.
            end: Ending (x, y) coordinates.
            label: Label for the arrow.
            color: Color of the arrow.
        """
        # Calculate arrow properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Adjust start and end points to avoid overlapping with nodes
        start_adj = (start[0] + dx_norm * 0.6, start[1] + dy_norm * 0.4)
        end_adj = (end[0] - dx_norm * 0.6, end[1] - dy_norm * 0.4)
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            start_adj, end_adj,
            arrowstyle='->',
            mutation_scale=20,
            color=color,
            linewidth=2
        )
        self.ax.add_patch(arrow)
        
        # Add label
        if label:
            mid_x = (start_adj[0] + end_adj[0]) / 2
            mid_y = (start_adj[1] + end_adj[1]) / 2
            self.ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', 
                        fontsize=8, color=color, fontweight='bold')
    
    def _draw_conditional_arrow(self, start: tuple, end: tuple, label: str, color: str) -> None:
        """Draw a conditional arrow with special styling.
        
        Args:
            start: Starting (x, y) coordinates.
            end: Ending (x, y) coordinates.
            label: Label for the arrow.
            color: Color of the arrow.
        """
        self._draw_arrow(start, end, label, color)
    
    def _add_legend(self) -> None:
        """Add a legend explaining the diagram elements."""
        legend_elements = [
            patches.Patch(color=NODE_COLORS['start'], label='Start Node'),
            patches.Patch(color=NODE_COLORS['llm'], label='LLM Processing'),
            patches.Patch(color=NODE_COLORS['retriever_agent'], label='Tool Execution'),
            patches.Patch(color=NODE_COLORS['end'], label='End Node'),
            patches.Patch(color='#FFC107', label='Decision Point'),
            patches.Patch(color=EDGE_COLORS['normal'], label='Normal Flow'),
            patches.Patch(color=EDGE_COLORS['conditional'], label='Conditional Flow')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    def _add_process_description(self) -> None:
        """Add a description of the agent process."""
        description = """
Process Description:
1. User asks a question
2. LLM Agent processes the question
3. If tool calls are needed, go to Retriever Agent
4. Retriever Agent searches documents and returns results
5. LLM Agent processes results and provides final answer
6. Process ends when no more tool calls are needed
        """
        
        self.ax.text(0.5, 1.5, description, fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))


class RAGAgent:
    """RAG agent that handles question answering using document retrieval."""
    
    def __init__(self, document_processor: DocumentProcessor):
        """Initialize the RAG agent.
        
        Args:
            document_processor: Processed document processor with retriever.
        """
        self.document_processor = document_processor
        self.llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
        self.tools = [self._create_retriever_tool()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.graph = self._create_agent_graph()
        self.agent = self.graph.compile()
    
    def _create_retriever_tool(self) -> tool:
        """Create the retriever tool for document search."""
        @tool
        def retriever_tool(query: str) -> str:
            """Search and return information from the Stock Market Performance 2024 document.
            
            Args:
                query: Search query to find relevant information.
                
            Returns:
                Retrieved document content or no results message.
            """
            docs = self.document_processor.retriever.invoke(query)
            
            if not docs:
                return "I found no relevant information in the Stock Market Performance 2024 document."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i+1}:\n{doc.page_content}")
            
            return "\n\n".join(results)
        
        return retriever_tool
    
    def _should_continue(self, state: AgentState) -> bool:
        """Check if the last message contains tool calls.
        
        Args:
            state: Current agent state.
            
        Returns:
            True if tool calls are present, False otherwise.
        """
        last_message = state['messages'][-1]
        return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0
    
    def _call_llm(self, state: AgentState) -> AgentState:
        """Call the LLM with the current state.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with LLM response.
        """
        messages = list(state['messages'])
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        message = self.llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with tool execution results.
        """
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            query = tool_args.get('query', 'No query provided')
            
            print(f"{TOOL_CALL_PREFIX} {tool_name} with query: {query}")
            
            if tool_name not in self.tools_dict:
                print(f"\nTool: {tool_name} does not exist.")
                result = TOOL_NOT_FOUND_ERROR
            else:
                result = self.tools_dict[tool_name].invoke(query)
                print(f"{TOOL_RESULT_LENGTH_PREFIX} {len(str(result))}")
            
            results.append(ToolMessage(
                tool_call_id=tool_id,
                name=tool_name,
                content=str(result)
            ))
        
        print(TOOL_EXECUTION_COMPLETE)
        return {'messages': results}
    
    def _create_agent_graph(self) -> StateGraph:
        """Create and configure the agent state graph.
        
        Returns:
            Configured StateGraph instance.
        """
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("llm", self._call_llm)
        graph.add_node("retriever_agent", self._execute_tools)
        
        # Add edges
        graph.add_conditional_edges(
            "llm",
            self._should_continue,
            {True: "retriever_agent", False: END}
        )
        graph.add_edge("retriever_agent", "llm")
        graph.set_entry_point("llm")
        
        return graph
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer from the RAG agent.
        
        Args:
            question: User's question.
            
        Returns:
            Agent's answer to the question.
        """
        messages = [HumanMessage(content=question)]
        result = self.agent.invoke({"messages": messages})
        return result['messages'][-1].content
    
    def create_flow_diagram(self, save_path: str = "rag_agent_flow.png") -> None:
        """Create a visual diagram of the agent's process flow.
        
        Args:
            save_path: Path to save the generated diagram.
        """
        visualizer = AgentVisualizer()
        visualizer.create_agent_flow_diagram(save_path)
    
    def run_interactive_session(self) -> None:
        """Run an interactive question-answering session."""
        print(WELCOME_MESSAGE)
        print("\nAvailable commands:")
        print("- Ask any question about stock market performance")
        print("- Type 'visualize' to see the agent process flow diagram")
        print("- Type 'exit' or 'quit' to end the session")
        
        while True:
            user_input = input(QUESTION_PROMPT)
            if user_input.lower() in EXIT_COMMANDS:
                break
            elif user_input.lower() == 'visualize':
                try:
                    self.create_flow_diagram()
                except Exception as e:
                    print(f"Error creating visualization: {e}")
                continue
            
            try:
                answer = self.ask_question(user_input)
                print(ANSWER_HEADER)
                print(answer)
            except Exception as e:
                print(f"Error processing question: {e}")


def create_rag_agent() -> RAGAgent:
    """Create and initialize a RAG agent with document processing.
    
    Returns:
        Initialized RAG agent ready for question answering.
    """
    document_processor = DocumentProcessor(
        pdf_path=PDF_FILENAME,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    
    document_processor.process_documents()
    
    return RAGAgent(document_processor)


def create_agent_visualization(save_path: str = "rag_agent_flow.png") -> None:
    """Create a standalone visualization of the RAG agent process flow.
    
    Args:
        save_path: Path to save the generated diagram.
    """
    try:
        print("Creating agent flow diagram...")
        visualizer = AgentVisualizer()
        visualizer.create_agent_flow_diagram(save_path)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def test_visualization() -> None:
    """Test function to verify matplotlib is working."""
    try:
        print("Testing matplotlib...")
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'ro-')
        plt.title('Test Plot - Matplotlib is Working!')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid(True)
        plt.show()
        print("Matplotlib test successful!")
    except Exception as e:
        print(f"Matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main function to run the RAG agent."""
    try:
        rag_agent = create_rag_agent()
        rag_agent.run_interactive_session()
    except Exception as e:
        print(f"Error initializing RAG agent: {e}")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "visualize":
            create_agent_visualization()
        elif sys.argv[1] == "test":
            test_visualization()
        else:
            print("Usage: python rag_agent.py [visualize|test]")
    else:
        main()