FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    python3-pip \
    uv

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY . /app

RUN export PATH="/root/.local/bin:$PATH"
RUN /root/.local/bin/poetry install
RUN /root/.local/bin/poetry run crewai install

CMD ["/app/.venv/bin/uv", "run", "run_server"]