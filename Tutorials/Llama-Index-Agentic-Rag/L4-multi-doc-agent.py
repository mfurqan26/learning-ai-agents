from helper import get_openai_api_key
import nest_asyncio
from utils import get_doc_tools
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner



OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

# Create agent summary over 3 papers
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

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