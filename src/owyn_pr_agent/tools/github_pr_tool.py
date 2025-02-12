from github import Github, Auth
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

load_dotenv()

class GithubPRTool(BaseTool):
    name: str = "Github Interface"
    description: str = "Interacts with Github"

    def _run(self):
        # TODO
        return None


# auth = Auth.token(

# TODO: implement langchain github tool in the CrewAI format
# https://python.langchain.com/docs/integrations/tools/github/