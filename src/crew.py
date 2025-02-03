from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import agentstack

@CrewBase
class OwyntrialagentCrew():
    """owyn_trial_agent crew"""

    @task
    def edit_blog_post(self) -> Task:
        return Task(
            config=self.tasks_config['edit_blog_post'],
        )

    @agent
    def musings_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['musings_editor'],
            tools=[*agentstack.tools['file_read']], # add tools here or use `agentstack tools add <tool_name>
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Test crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )