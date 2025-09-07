from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def search(query: str) -> str:
    """
    param query: a search query
    returns: the top search result
    """
    search_tool = TavilySearch(max_results=1)
    results = search_tool.run(query)
    return results


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [search, triple]

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0
).bind_tools(tools)
