from github import Github, Auth
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
import os

load_dotenv()

wrapper = GitHubAPIWrapper()

class GithubPRTool(BaseTool):
    name: str = "Github Interface"
    description: str = "Interacts with Github"
    github: GitHubToolkit = Field(default_factory=lambda: GitHubToolkit.from_github_api_wrapper(wrapper))

    def _run(self, action: str, **kwargs) -> str:
        # TODO: clone the repo, checkout the branch, return the filename
        # content = self.github.get_repo("owynrichen/owynrichen.com").get_contents("content/playing_with_ai_agents.md")
        # return content.decoded_content.decode()
        return "This is a TODO"


# TODO: implement langchain github tool in the CrewAI format
# https://python.langchain.com/docs/integrations/tools/github/