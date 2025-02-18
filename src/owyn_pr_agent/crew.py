from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from .tools.searxng_search_tool import SearxNGSearchTool
from .tools.github_pr_tool import GithubPRTool
from crewai_tools import FileReadTool, FileWriterTool, ScrapeWebsiteTool
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from typing import List, Callable, Union, cast
from pydantic import BaseModel, Field
from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class OllamaEmbeddingLongContextFunction(OllamaEmbeddingFunction):
    def __init__(self) -> None:
        super().__init__("http://ollama.pdx.internal.owynrichen.com:11434/api/embeddings", "nomic-embed-text:latest")
        self.num_ctx = 8192

    def __call__(self, input: Union[Documents, str]) -> Embeddings:
        texts = input if isinstance(input, list) else [input]

        # print(input)

        embeddings = [
            self._session.post(
                self._api_url, json={"model": self._model_name, "prompt": text, "options": { "num_ctx" : self.num_ctx }},
                # self._api_url, json={"model": self._model_name, "prompt": text },
            ).json()
            for text in texts
        ]

        # print(embeddings)

        return cast(
            Embeddings,
            [
                embedding["embedding"]
                for embedding in embeddings
                if "embedding" in embedding
            ],
        )

class Document(BaseModel):
    """The document to review"""
    title: str = Field(..., description="The title of the document")
    body: str = Field(..., description="The contents of the document")
    references: List[str] = Field(..., description="The urls of the sources referenced in the document")

class EditingStrategy(BaseModel):
    document: Document = Field(..., description="The document")
    editing_strategy: str = Field(..., description="The managing editors notes for how to approach editing the document")

class TopicResearchReport(BaseModel):
    """Research Report"""
    name: str = Field(..., description="Name of the report")
    sources: List[str] = Field(..., description="List of sources seen as authoritative on the topic")
    themes: List[str] = Field(...,description="List of themes the topic covers")
    report: str = Field(..., description="The full report, with sources and themes annotated")

class Evidence(BaseModel):
    """An individual review of a claim"""
    source: str = Field(..., description="The url of the source referenced")
    quote: str = Field(..., description="Quoted text supporting the evidence")

class ClaimReview(BaseModel):
    """A claim in the document that has been reviewed for facts"""
    claim: str = Field(..., description="The text referencing the claim in the document")
    start_pos: int = Field(..., description="The starting character position of the claim in the document")
    is_true: bool = Field(..., description="Whether the analyst believes this claim is true or false")
    evidence: List[Evidence] = Field(..., description="The urls of the sources referenced to evaluate the claim")
    summary: str = Field(..., description="A long-form description of the claim and the assessment of if it is true or false, and why")

class FactCheckReport(BaseModel):
    name: str = Field(..., description="The name of the fact check report")
    analysis: List[ClaimReview] = Field(..., descrption="The list of claims and whether or not they are factual, with evidence")


@CrewBase
class OwynPrAgent():
    """owyn_trial_agent crew"""

    embedding_function: Callable[[], EmbeddingFunction]

    def __init__(self):
        self.ollama_llm = LLM(
            # model="ollama/llama3.2:3b",
            # model="ollama/qwen2.5:7b",
            model="ollama_chat/deepseek-r1:7b",
            base_url="http://ollama.pdx.internal.owynrichen.com:11434",
            timeout=600,
            max_tokens=8192,
        )

        # switch back to this to hit Cloudflare APIs instead
        self.cf_llama_llm =  "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    @before_kickoff
    def before_kickoff_function(self, inputs):
        print(f"Before kickoff function with inputs: {inputs}")
        return inputs # You can return the inputs or modify them as needed

    @after_kickoff
    def after_kickoff_function(self, result):
        print(f"After kickoff function with result: {result}")
        return result # You can return the result or modify it as needed

    @task
    def retrieve_branch(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_branch'],
            output_pydantic=Document
        )

    @task
    def plan_edit(self) -> Task:
        return Task(
            config=self.tasks_config['plan_edit'],
            output_pydantic=EditingStrategy
        )

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
            context=[self.plan_edit()],
            output_pydantic=TopicResearchReport
        )

    @task
    def edit_copy(self) -> Task:
        return Task(
            config=self.tasks_config['edit_copy'],
            context=[self.plan_edit(), self.research_topic()],
            output_pydantic=Document,
            output_file="{revised_file_path}"
        )

    @task
    def fact_check(self) -> Task:
        return Task(
            config=self.tasks_config['fact_check'],
            context=[self.edit_copy()],
            output_pydantic=FactCheckReport
        )

    @agent
    def branch_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['branch_retriever'],
            tools=[
                GithubPRTool()
            ],
            llm=self.ollama_llm,
            verbose=True,
            memory=False
        )

    @agent
    def managing_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['managing_editor'],
            tools=[
                FileReadTool()
            ],
            llm=self.ollama_llm,
            verbose=True,
            memory=False,
            output_pydantic=EditingStrategy,
        )

    @agent
    def lead_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_researcher'],
            tools=[
                SearxNGSearchTool(),
                ScrapeWebsiteTool()
            ],
            llm=self.ollama_llm,
            verbose=True,
            memory=True,
            output_pydantic=TopicResearchReport,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'],
            tools=[
                SearxNGSearchTool(),
                ScrapeWebsiteTool()
            ],
            llm=self.ollama_llm,
            verbose=True,
            memory=False,
            output_pydantic=FactCheckReport,
        )

    @agent
    def copy_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['copy_editor'],
            tools=[
                FileWriterTool()
            ],
            llm=self.ollama_llm,
            verbose=True,
            memory=False,
            # output_pydantic=Document,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            memory=False,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path="./memory.db")
            ),
            embedder={
                "provider" : "custom",
                "config" : {
                    "embedder" : OllamaEmbeddingLongContextFunction(),
                }
            },
        )