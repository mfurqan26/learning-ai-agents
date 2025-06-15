# Add your utilities or helper functions to this file.
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from multion.client import MultiOn
import base64
from io import BytesIO
from PIL import Image
from IPython.display import display, HTML, Markdown
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_serper_api_key():
    load_env()
    serper_api_key = os.getenv("SERPER_API_KEY")
    return serper_api_key

def get_openai_client():
    openai_api_key = get_openai_api_key()
    return OpenAI(api_key=openai_api_key)

def get_multi_on_api_key():
    load_env()
    multi_on_api_key = os.getenv("MULTION_API_KEY")
    return multi_on_api_key

def get_multi_on_client():
    multi_on_api_key = get_multi_on_api_key()
    return MultiOn(api_key=multi_on_api_key)

# Router Query Engine for llama-index RAG
def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-4o-mini")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    
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
    return query_engine

# get doc tools
def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
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
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool

# Params
async def visualizeCourses(result, screenshot, target_url, instructions, base_url):
    # Run the async process that returns an instance of DeeplearningCourseList and screenshot bytes
    if result:
        # Convert each course to a dict (using model_dump from Pydantic v2)
        courses_data = [course.model_dump() for course in result.courses]

        for course in courses_data:
          if course['courseURL']:
            course['courseURL'] = f'<a href="{base_url}{course["courseURL"]}" target="_blank">{course["title"]}</a>'


        # Build an HTML table if course data is available
        if courses_data:
            # Extract headers from the first course
            headers = courses_data[0].keys()
            table_html = '<table style="border-collapse: collapse; width: 100%;">'
            table_html += '<thead><tr>'
            for header in headers:
                table_html += (
                    f'<th style="border: 1px solid #dddddd; text-align: left; padding: 8px;">'
                    f'{header}</th>'
                )
            table_html += '</tr></thead>'
            table_html += '<tbody>'
            for course in courses_data:
                table_html += '<tr>'
                for header in headers:
                    value = course[header]
                    # If the field is "imageUrl", embed the image in the table cell
                    if header == "imageUrl":
                        value = (f'<img src="{value}" alt="Course Image" '
                                 f'style="max-width:100px; height:auto;">')
                    elif isinstance(value, list):
                        value = ', '.join(value)
                    table_html += (
                        f'<td style="border: 1px solid #dddddd; text-align: left; padding: 8px;">'
                        f'{value}</td>'
                    )
                table_html += '</tr>'
            table_html += '</tbody></table>'
        else:
            table_html = "<p>No course data available.</p>"

        # Display the course data table
        display(Markdown("### Scraped Course Data:"))
        display(HTML(table_html))

        # Convert the screenshot bytes into a base64 string and embed it in an <img> tag
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        img_html = (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'alt="Website Screenshot" style="max-width:100%; height:auto;">'
        )
        display(Markdown("### Website Screenshot:"))
        display(HTML(img_html))