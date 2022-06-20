FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

RUN mkdir output

COPY ["starter.py", "./"]

ENTRYPOINT ["python3", "starter.py"]
