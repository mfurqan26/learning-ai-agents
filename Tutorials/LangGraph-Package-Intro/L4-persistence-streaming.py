from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import nest_asyncio
import asyncio
import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

_ = load_dotenv()

tool = TavilySearchResults(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

async def main():
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Use context manager for SqliteSaver
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        
        # Streaming Example 1
        messages = [HumanMessage(content="What is the weather in sf?")]
        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v['messages'])
        
        # Streaming Example 2: continued ask about weather in different city
        messages = [HumanMessage(content="What about in la?")]
        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)

        # Streaming Example 3: continued ask about which city is warmer by using thread id
        messages = [HumanMessage(content="Which one is warmer?")]
        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)

        # Streaming Example 4: different thread id will confuse the agent
        messages = [HumanMessage(content="Which one is warmer?")]
        thread = {"configurable": {"thread_id": "2"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)

    # Replace the async example with:
    memory = AsyncSqliteSaver.from_conn_string(":memory:")
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)
    messages = [HumanMessage(content="What is the weather in SF?")]
    thread = {"configurable": {"thread_id": "4"}}
    async def stream_events(messages, thread, abot):
        async for event in abot.graph.astream_events({"messages": messages}, thread):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Empty content in the context of OpenAI means that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(content, end="|")
    await stream_events(messages, thread, abot)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())