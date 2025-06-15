from utils import get_openai_api_key
import nest_asyncio
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

# Define simple tools, Note that type annotations and function doctsrings are important for llm context to use these tools
def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y
def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)

add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

# Simple test of the simple tools
llm = OpenAI(model="gpt-4o-mini")
response = llm.predict_and_call(
    [add_tool, mystery_tool], 
    "Tell me the output of the mystery function on 2 and 9", 
    verbose=True
)
print(str(response))


# load documents
documents = SimpleDirectoryReader(input_files=["./Tutorials/Llama-Index-Agentic-Rag/metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
print(nodes[0].get_content(metadata_mode="all"))

# Create vector index and query engine
vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)
query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)
response = query_engine.query(
    "What are some high-level results of MetaGPT?", 
)
print(str(response))
for n in response.source_nodes:
    print(n.metadata)

# Define auto retrieval tool
def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response
    
vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)

# Test the auto retrieval tool
llm = OpenAI(model="gpt-4o-mini", temperature=0)
response = llm.predict_and_call(
    [vector_query_tool], 
    "What are the high-level results of MetaGPT as described on page 2?", 
    verbose=True
)
for n in response.source_nodes:
    print(n.metadata)
    
# Define other tools like summary
summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of MetaGPT"
    ),
)
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What are the MetaGPT comparisons with ChatDev described on page 8?", 
    verbose=True
)
for n in response.source_nodes:
    print(n.metadata)
    
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What is a summary of the paper?", 
    verbose=True
)