"""
Document Drafter Agent - A LangGraph-based AI assistant for document editing and management.

This module provides a conversational AI agent that helps users create, update, and save documents
through natural language interactions.
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"
TEXT_FILE_EXTENSION = ".txt"
MAX_RECENT_MESSAGES = 3

# UI Messages
WELCOME_MESSAGE = "I'm ready to help you update a document. What would you like to create?"
USER_PROMPT = "\nWhat would you like to do with the document? "
USER_PREFIX = "\n👤 USER: "
AI_PREFIX = "\n🤖 AI: "
TOOL_PREFIX = "\n🔧 USING TOOLS: "
TOOL_RESULT_PREFIX = "\n🛠️ TOOL RESULT: "
SAVE_PREFIX = "\n💾 Document has been saved to: "
APP_TITLE = "\n ===== DRAFTER ====="
APP_FINISHED = "\n ===== DRAFTER FINISHED ====="

# System prompts
SYSTEM_PROMPT_TEMPLATE = """
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

The current document content is: {document_content}
"""


class AgentState(TypedDict):
    """State structure for the document drafter agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str


class DocumentManager:
    """Manages document content and operations."""
    
    def __init__(self):
        self._content = ""
    
    @property
    def content(self) -> str:
        """Get current document content."""
        return self._content
    
    def update_content(self, new_content: str) -> str:
        """Update document content and return confirmation message."""
        self._content = new_content
        return f"Document has been updated successfully! The current content is:\n{self._content}"
    
    def save_to_file(self, filename: str) -> str:
        """Save document to file and return status message."""
        if not filename.endswith(TEXT_FILE_EXTENSION):
            filename = f"{filename}{TEXT_FILE_EXTENSION}"
        
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(self._content)
            print(f"{SAVE_PREFIX}{filename}")
            return f"Document has been saved successfully to '{filename}'."
        except Exception as e:
            return f"Error saving document: {str(e)}"


# Global document manager instance
document_manager = DocumentManager()


@tool
def update_document(content: str) -> str:
    """Updates the document with the provided content.
    
    Args:
        content: The new content to replace the current document content.
        
    Returns:
        Confirmation message with updated content.
    """
    return document_manager.update_content(content)


@tool
def save_document(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file (extension will be added if missing).
        
    Returns:
        Status message indicating success or failure.
    """
    return document_manager.save_to_file(filename)


# Tool configuration
TOOLS = [update_document, save_document]
MODEL = ChatOpenAI(model=DEFAULT_MODEL).bind_tools(TOOLS)


def create_system_message(document_content: str) -> SystemMessage:
    """Create system message with current document content."""
    return SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(document_content=document_content))


def get_user_input(is_first_interaction: bool) -> HumanMessage:
    """Get user input and return as HumanMessage."""
    if is_first_interaction:
        user_input = WELCOME_MESSAGE
    else:
        user_input = input(USER_PROMPT)
        print(f"{USER_PREFIX}{user_input}")
    
    return HumanMessage(content=user_input)


def display_ai_response(response: AIMessage) -> None:
    """Display AI response and tool usage information."""
    print(f"{AI_PREFIX}{response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tool_call['name'] for tool_call in response.tool_calls]
        print(f"{TOOL_PREFIX}{tool_names}")


def document_agent(state: AgentState) -> AgentState:
    """Main agent function that handles user interactions and AI responses.
    
    Args:
        state: Current agent state containing messages and document content.
        
    Returns:
        Updated state with new messages.
    """
    is_first_interaction = not state["messages"]
    system_message = create_system_message(state["document_content"])
    user_message = get_user_input(is_first_interaction)
    
    all_messages = [system_message] + list(state["messages"]) + [user_message]
    response = MODEL.invoke(all_messages)
    
    display_ai_response(response)
    
    return {
        "messages": list(state["messages"]) + [user_message, response],
        "document_content": document_manager.content
    }


def execute_tool_calls(state: AgentState) -> AgentState:
    """Execute tool calls from the last AI message and return results.
    
    Args:
        state: Current agent state containing messages.
        
    Returns:
        Updated state with tool execution results.
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls
    tool_results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Find and execute the tool
        for tool in TOOLS:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    ))
                except Exception as e:
                    error_message = f"Error executing {tool_name}: {str(e)}"
                    tool_results.append(ToolMessage(
                        content=error_message,
                        tool_call_id=tool_id
                    ))
                break
    
    return {
        "messages": tool_results,
        "document_content": document_manager.content
    }


def should_continue_conversation(state: AgentState) -> str:
    """Determine if the conversation should continue or end.
    
    Args:
        state: Current agent state containing messages.
        
    Returns:
        "continue" to keep the conversation going, "end" to finish.
    """
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Check if the most recent tool message indicates document was saved
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    
    return "continue"


def display_tool_results(messages: List[BaseMessage]) -> None:
    """Display tool execution results in a readable format.
    
    Args:
        messages: List of messages to display.
    """
    if not messages:
        return
    
    for message in messages[-MAX_RECENT_MESSAGES:]:
        if isinstance(message, ToolMessage):
            print(f"{TOOL_RESULT_PREFIX}{message.content}")


def create_agent_graph() -> StateGraph:
    """Create and configure the LangGraph state graph.
    
    Returns:
        Configured StateGraph instance.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", document_agent)
    graph.add_node("tools", execute_tool_calls)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add edges
    graph.add_edge("agent", "tools")
    graph.add_conditional_edges(
        "tools",
        should_continue_conversation,
        {
            "continue": "agent",
            "end": END,
        },
    )
    
    return graph


def run_document_agent() -> None:
    """Main function to run the document drafter agent."""
    print(APP_TITLE)
    
    initial_state = {
        "messages": [],
        "document_content": document_manager.content
    }
    
    app = create_agent_graph().compile()
    
    for step in app.stream(initial_state, stream_mode="values"):
        if "messages" in step:
            display_tool_results(step["messages"])
    
    print(APP_FINISHED)


if __name__ == "__main__":
    run_document_agent()