FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    pip install .

EXPOSE 8000

CMD ["python", "-m", "strategylab.apps.api.main"]

