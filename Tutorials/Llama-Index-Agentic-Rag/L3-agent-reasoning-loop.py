from utils import get_openai_api_key, get_doc_tools
import nest_asyncio
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

vector_tool, summary_tool = get_doc_tools("./Tutorials/Llama-Index-Agentic-Rag/papers/metagpt.pdf", "metagpt")

# Set up agent worker and runner
llm = OpenAI(model="gpt-4o-mini", temperature=0)
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Test the agent
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

print(response.source_nodes[0].get_content(metadata_mode="all"))
response = agent.chat(
    "Tell me about the evaluation datasets used."
)
response = agent.chat("Tell me the results over one of the above datasets.")

# Debugging and control
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)
step_output = agent.run_step(task.task_id)
completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)

step_output = agent.run_step(task.task_id)
print(step_output.is_last)

response = agent.finalize_response(task.task_id)
print(str(response))