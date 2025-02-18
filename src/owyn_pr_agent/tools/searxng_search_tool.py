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
