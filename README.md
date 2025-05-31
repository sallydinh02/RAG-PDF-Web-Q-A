# RAG-Langchain-project

## Steps to run the code (install docker first):

docker build -t ragproject .

docker run -p 8501:8501 ragproject
 
## Steps to run the project (without docker):

cd to the project

pip install -r requirements.txt

ollama pull llama3.2

ollama pull gemma3:4b

streamlit run web_crawl_rag.py