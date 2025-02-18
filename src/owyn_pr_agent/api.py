from fastapi import FastAPI
from crewai import Crew

class CrewAPIServer:
    def __init__(self, crew: Crew):
        self.app = FastAPI()
        self.crew = crew
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def read_root():
            return {"Hello": "World"}

        @self.app.post("/github/webhook")
        def github_webhook(**kwargs):
            print(f"Github Webhook received with kwargs: {kwargs}")
            # self.crew.kickoff(inputs={})
            return {"message": "Github Webhook received"}


    def run(self):
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=9042)


