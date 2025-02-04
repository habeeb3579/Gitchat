ARG group=appgroup
ARG user=appuser
ARG PYTHON_VERSION=3.10.10

FROM python:${PYTHON_VERSION}-slim as base


# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
RUN addgroup appgroup && \
    adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    appuser && \
    usermod -a -G appgroup appuser

WORKDIR /app

# Create the directory
RUN mkdir -p docs/autogen

# Recursively change ownership of the /app directory and its contents to appuser:appgroup
RUN chown -R appuser:appgroup /app

#enable buildkit
RUN export DOCKER_BUILDKIT=1

#install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*


# Install Git
RUN apt-get update && apt-get install -y git

# Add exception for directory
RUN git config --global --add safe.directory /app/docs/autogen

# Retrieve the path to Git executable
RUN GIT_EXEC=$(which git) && \
    echo "GIT_PYTHON_GIT_EXECUTABLE=$GIT_EXEC" >> /etc/environment

# Copy the source code into the container.
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt \   
    && python -m pip install GitPython \
    && python -m pip install --upgrade pip setuptools


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

USER ${user}

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
