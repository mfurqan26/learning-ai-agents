import nest_asyncio
from utils import get_doc_tools, get_openai_api_key
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.objects import ObjectIndex

OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

# Create agent summary over 3 papers
papers_folder = "./Tutorials/Llama-Index-Agentic-Rag/papers/"
papers = [
    papers_folder + "metagpt.pdf",
    papers_folder + "longlora.pdf",
    papers_folder + "selfrag.pdf",
]

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
llm = OpenAI(model="gpt-4o-mini")
len(initial_tools)

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)
response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))

# 10 papers agent chat
papers = [
    papers_folder + "metagpt.pdf",
    papers_folder + "longlora.pdf",
    papers_folder + "loftq.pdf",
    papers_folder + "swebench.pdf",
    papers_folder + "selfrag.pdf",
    papers_folder + "zipformer.pdf",
    papers_folder + "values.pdf",
    papers_folder + "finetune_fair_diffusion.pdf",
    papers_folder + "knowledge_card.pdf",
    papers_folder + "metra.pdf",
]

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
 
# Extend the tools to include the object index
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)
tools[2].metadata

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))
response = agent.query(
    "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
    "Analyze the approach in each paper first. "
)