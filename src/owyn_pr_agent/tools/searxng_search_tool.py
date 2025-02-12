from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import SearxSearchWrapper


class SearxNGSearchTool(BaseTool):
    name: str = "Search (SearxNG)"
    description: str = "Useful for search"
    search: SearxSearchWrapper = Field(default_factory=lambda: SearxSearchWrapper(searx_host="http://searxng.pdx.internal.owynrichen.com"))

    def _run(self, query: str) -> str:
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"

# @tool
# def searxng_search(query: str) -> str:
#     """
#     Search the web using a locally hosted SearxNG instance.
#     Args:
#         query (str): Search term
#     Returns:
#         str: Top results
#     """
#     SEARXNG_URL = "http://searxng.pdx.internal.owynrichen.com/search"
#     params = {'q': query, 'format': 'json'}
#     response = requests.get(SEARXNG_URL, params=params)

#     if response.status_code == 200:
#         results = response.json()
#         return "\n".join(
#             f"{res['title']} - {res['url']}" for res in results['results']
#         )
#     else:
#         return f"Error: {response.status_code}"