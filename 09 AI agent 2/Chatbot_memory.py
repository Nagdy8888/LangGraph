import os 
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print('Current State:',state['messages'])

    return state

graph = StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter your message: ")
while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({'messages':conversation_history})
    conversation_history = result['messages']

    user_input = input("Enter your message: ")

with open('conversation_history.txt','w') as f:
    f.write("conversation_history:\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            f.write(f"Human: {message.content}\n")
        elif isinstance(message,AIMessage):
            f.write(f"AI: {message.content}\n\n\n")
    f.write('End of Conversation')

print('Conversation Saved to conversion_history.txt')