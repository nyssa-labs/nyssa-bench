FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY nyssa_bench ./nyssa_bench
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install -e ".[dev]"

CMD ["nyssa", "list-suites"]
