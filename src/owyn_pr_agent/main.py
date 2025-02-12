#!/usr/bin/env python
import sys
import argparse
import os
import base64
from datetime import date
from dotenv import load_dotenv
import openlit

load_dotenv()
LANGFUSE_AUTH=base64.b64encode(f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

openlit.init(disable_metrics=True)

from .crew import OwynPrAgent
instance = OwynPrAgent().crew()

def get_inputs():
    inputs: dict[str, str] = {}

    # TODO, read inputs from yaml and file
    inputs["file_path"] = "../owynrichen.com/content/playing_with_ai_agents.md"
    inputs["revised_file_path"] = "../owynrichen.com/content/revised_playing_with_ai_agents.md"
    inputs["current_date"] = str(date.today())
    return inputs

def run():
    """
    Run the agent.
    """
    print("Running the crew...")
    instance.kickoff(inputs=get_inputs())


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        instance.train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=get_inputs(),
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        instance.replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        instance.test(
            n_iterations=int(sys.argv[1]),
            openai_model_name=sys.argv[2],
            inputs=get_inputs(),
        )
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


if __name__ == '__main__':
    run()