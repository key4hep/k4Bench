FROM python:3.13-slim

WORKDIR /app

# setuptools-scm requires git tags; pretend a version so it works without .git
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Install dd4bench package (provides the analysis loader)
COPY pyproject.toml README.md ./
COPY dd4bench/ ./dd4bench/
RUN pip install --no-cache-dir .

# Install dashboard dependencies
COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

# Copy dashboard source last (most likely to change)
COPY dashboard/ ./dashboard/

# OpenShift runs containers with arbitrary UIDs; ensure group-writable dirs
RUN mkdir -p /tmp/.streamlit && chmod -R g+rwX /tmp /app
ENV HOME=/tmp

WORKDIR /app/dashboard
EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
