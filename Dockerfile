FROM python:3.10-slim

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "web_crawl_rag.py"]