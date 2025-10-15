"""
Core RAG Agent module.

This module contains the main RAG agent class that handles question answering using document retrieval.
"""

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from config import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, SYSTEM_PROMPT,
    WELCOME_MESSAGE, EXIT_COMMANDS, QUESTION_PROMPT, ANSWER_HEADER,
    TOOL_CALL_PREFIX, TOOL_RESULT_LENGTH_PREFIX, TOOL_EXECUTION_COMPLETE, TOOL_NOT_FOUND_ERROR
)
from document_processor import DocumentProcessor
from agent_visualizer import AgentVisualizer


class AgentState(TypedDict):
    """State structure for the RAG agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
            docs = self.document_processor.get_retriever().invoke(query)
            
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
