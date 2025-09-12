FROM python:3.10.12

# system update
RUN apt-get update

# install Git and Bash, Git LFS dependencies
RUN apt-get install -y \
    curl \
    git \
    bash

# install Python dependencies
RUN --mount=type=bind,source=.devcontainer/requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --requirement /tmp/requirements.txt

# Dev Environment:
    # Python: ✅ 3.10   
    # Python libraries: LangChain, LangSmith, LangGrapph, FastAPI, Pymongo, ChromaDB
    # Linux: ✅ Alpine 3.18
    # Git: ✅ 
    # Bash: ✅