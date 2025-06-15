from utils import get_openai_api_key
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

# load documents
documents = SimpleDirectoryReader(input_files=["./Tutorials/Llama-Index-Agentic-Rag/metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Define the LLM and embedding model
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Create indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Create query engines and tools
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

# Test the router engine
response = query_engine.query("What is the summary of the document?")
print(str(response))
print(len(response.source_nodes))
response = query_engine.query(
    "How do agents share information with other agents?"
)
print(str(response))

# Same router engine defined in utils.py as helper function to use later easily
from utils import get_router_query_engine
query_engine = get_router_query_engine("metagpt.pdf")
response = query_engine.query("Tell me about the ablation study results?")
print(str(response))