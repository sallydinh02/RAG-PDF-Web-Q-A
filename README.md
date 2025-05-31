# RAG-Langchain-project

## About
A website for users to ask questions about a website or a pdf file using LangChain, Ollama, RAG and Streamlit
* web_crawl_rag.py: Ask questions about a website
* pdf_rag.py: Ask questions about a pdf file (text only)
* multimodal_rag_pdf.py: Ask questions about a pdf containing text, image and table

## Steps to run the code (with docker):
* Install Docker
* Run: docker build -t ragproject .
* Run: docker run -p 8501:8501 ragproject

## Steps to run the project (without docker):
* Download Ollama
* cd to the project
* pip install -r requirements.txt
* ollama pull llama3.2
* ollama pull gemma3:4b
* streamlit run filename

## Demo
* Ask chatbot about a website
![demo web crawler rag](https://github.com/user-attachments/assets/45b27d57-eed0-4878-aff2-1ab1afa6146a)
