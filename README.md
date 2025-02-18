# Owyn PR Agent

# Setup

I need to switch this from a ```poetry``` setup to a pure ```uv``` setup since CrewAI uses uv
under the covers, but haven't yet.

To get this installed, run this:

```bash
poetry install
poetry run crewai install
```

You'll need to setup a ```.env``` file, so you can/should run this:

```bash
copy .env.example .env
# edit the file
```

# Running

```bash
poetry run crewai run
```

# Building the Container

Assuming you're using podman (vs. docker) you build it like this:

```bash
podman build -t owyn-pr-agent:latest .
```