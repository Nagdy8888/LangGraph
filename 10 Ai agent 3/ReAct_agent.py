from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage #BaseMessage is the parent class for all message types
from langchain_core.messages import ToolMessage #ToolMessage is the message type for tool calls
from langchain_core.messages import SystemMessage #SystemMessage is the instruction for the agent
from langchain_core.messages import HumanMessage #HumanMessage is the message type for user input

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
# ToolNode is not available in this version, we'll create our own tool execution function

load_dotenv()





class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]



@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 
@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]
model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)




def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"

def tool_execution(state: AgentState) -> AgentState:
    """Execute tool calls and return results"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Get the tool calls from the last message
    tool_calls = last_message.tool_calls
    tool_results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    ))
                except Exception as e:
                    tool_results.append(ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_id
                    ))
                break
    
    return {"messages": tool_results}





graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", tool_execution)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()




def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))