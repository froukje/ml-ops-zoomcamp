FROM python:3.9.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["./mlruns", "./mlruns"]
COPY ["mlruns.db", ".mlruns.db"]

RUN pipenv install --system --deploy

COPY ["predict-docker.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict-docker:app"]
