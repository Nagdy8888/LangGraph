"""
Multi-Modal Agent Core Module.

This module contains the main multi-modal agent class that orchestrates
all the different tools and capabilities available to the agent.
"""

from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from operator import add as add_messages
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

from config import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, SYSTEM_PROMPT, WELCOME_MESSAGE,
    EXIT_COMMANDS, QUESTION_PROMPT, ANSWER_HEADER, TOOL_CALL_PREFIX,
    TOOL_RESULT_PREFIX, TOOL_EXECUTION_COMPLETE, TOOL_NOT_FOUND_ERROR,
    HELP_MESSAGE, WORKSPACE_DIR, CONVERSATION_FILE, MAX_CONVERSATION_LENGTH
)
from tools import ALL_TOOLS, TOOL_CATEGORIES


class AgentState(TypedDict):
    """State structure for the multi-modal agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    conversation_history: List[Dict[str, Any]]
    current_tools: List[str]
    memory: Dict[str, Any]


class MultiModalAgent:
    """Multi-modal agent that handles various types of tasks and interactions."""
    
    def __init__(self):
        """Initialize the multi-modal agent."""
        self.llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
        self.tools = ALL_TOOLS
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.conversation_history = []
        self.memory = {}
        self.graph = self._create_agent_graph()
        self.agent = self.graph.compile()
    
    def _should_continue(self, state: AgentState) -> str:
        """Check if the last message contains tool calls.
        
        Args:
            state: Current agent state.
            
        Returns:
            'tools' if tool calls are present, 'end' otherwise.
        """
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
            return 'tools'
        return 'end'
    
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
        
        # Update conversation history
        self._update_conversation_history("assistant", message.content)
        
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
            
            print(f"{TOOL_CALL_PREFIX} {tool_name}")
            print(f"Arguments: {tool_args}")
            
            if tool_name not in self.tools_dict:
                print(f"Tool: {tool_name} does not exist.")
                result = TOOL_NOT_FOUND_ERROR
            else:
                try:
                    result = self.tools_dict[tool_name].invoke(tool_args)
                    print(f"{TOOL_RESULT_PREFIX} {str(result)[:100]}...")
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                    print(f"Error: {result}")
            
            results.append(ToolMessage(
                tool_call_id=tool_id,
                name=tool_name,
                content=str(result)
            ))
        
        print(TOOL_EXECUTION_COMPLETE)
        return {'messages': results}
    
    def _update_conversation_history(self, role: str, content: str) -> None:
        """Update the conversation history.
        
        Args:
            role: Role of the speaker ('user' or 'assistant')
            content: Content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history length
        if len(self.conversation_history) > MAX_CONVERSATION_LENGTH:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_LENGTH:]
    
    def _create_agent_graph(self) -> StateGraph:
        """Create and configure the agent state graph.
        
        Returns:
            Configured StateGraph instance.
        """
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self._call_llm)
        graph.add_node("tools", self._execute_tools)
        
        # Add edges
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {True: "tools", False: END}
        )
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        
        return graph
    
    def process_message(self, message: str) -> str:
        """Process a user message and return the agent's response.
        
        Args:
            message: User's message.
            
        Returns:
            Agent's response.
        """
        # Handle special commands
        if message.lower() in EXIT_COMMANDS:
            return "Goodbye! Thanks for using the Multi-Modal Agent."
        
        if message.lower() == 'help':
            return HELP_MESSAGE
        
        if message.lower() == 'tools':
            return self._list_available_tools()
        
        if message.lower() == 'memory':
            return self._show_memory_status()
        
        if message.lower() == 'clear':
            self.conversation_history = []
            return "Conversation history cleared."
        
        if message.lower() == 'save':
            return self._save_conversation()
        
        if message.lower().startswith('load '):
            filename = message[5:].strip()
            return self._load_conversation(filename)
        
        if message.lower() == 'visualize':
            return "Use the visualize_agent_flow() method to create a flow diagram."
        
        # Update conversation history
        self._update_conversation_history("user", message)
        
        # Process with agent
        messages = [HumanMessage(content=message)]
        result = self.agent.invoke({"messages": messages})
        
        return result['messages'][-1].content
    
    def _list_available_tools(self) -> str:
        """List all available tools organized by category.
        
        Returns:
            Formatted list of available tools.
        """
        result = "Available Tools by Category:\n\n"
        
        for category, tools in TOOL_CATEGORIES.items():
            result += f"📁 {category}:\n"
            for tool in tools:
                result += f"  • {tool.name}: {tool.description}\n"
            result += "\n"
        
        return result
    
    def _show_memory_status(self) -> str:
        """Show current memory status.
        
        Returns:
            Memory status information.
        """
        memory_file = f"{WORKSPACE_DIR}/{CONVERSATION_FILE}"
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            return f"Memory Status:\n• Conversation entries: {len(self.conversation_history)}\n• Memory file: {memory_file}\n• Memory entries: {len(memory_data) if isinstance(memory_data, dict) else 'N/A'}"
        except:
            return f"Memory Status:\n• Conversation entries: {len(self.conversation_history)}\n• Memory file: {memory_file}\n• Memory entries: Not available"
    
    def _save_conversation(self) -> str:
        """Save conversation to file.
        
        Returns:
            Success or error message.
        """
        try:
            memory_file = f"{WORKSPACE_DIR}/{CONVERSATION_FILE}"
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2)
            
            return f"Conversation saved to {memory_file}"
        except Exception as e:
            return f"Error saving conversation: {str(e)}"
    
    def _load_conversation(self, filename: str) -> str:
        """Load conversation from file.
        
        Args:
            filename: Name of the file to load from.
            
        Returns:
            Success or error message.
        """
        try:
            file_path = f"{WORKSPACE_DIR}/{filename}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            self.conversation_history = loaded_history
            return f"Conversation loaded from {filename}"
        except Exception as e:
            return f"Error loading conversation: {str(e)}"
    
    def run_interactive_session(self) -> None:
        """Run an interactive session with the agent."""
        print(WELCOME_MESSAGE)
        
        while True:
            try:
                user_input = input(QUESTION_PROMPT)
                
                if user_input.lower() in EXIT_COMMANDS:
                    print("Goodbye! Thanks for using the Multi-Modal Agent.")
                    break
                
                response = self.process_message(user_input)
                print(ANSWER_HEADER)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using the Multi-Modal Agent.")
                break
            except Exception as e:
                print(f"Error processing message: {e}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history.
        
        Returns:
            List of conversation entries.
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names.
        """
        return [tool.name for tool in self.tools]
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools by category.
        
        Args:
            category: Tool category name.
            
        Returns:
            List of tool names in the category.
        """
        if category in TOOL_CATEGORIES:
            return [tool.name for tool in TOOL_CATEGORIES[category]]
        return []
