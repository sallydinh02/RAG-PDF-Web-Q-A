import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

template = """
You are an assistant for question-answering tasks. Use the context retrieved bellow to answer the question. If you don't know the answer, just say you don't know. Your answer should be concise.
Question: {question}
Context: {context}
Answer:
"""

pdfs_dir='./pdfs'
fig_dir='./figures'

embeddings=OllamaEmbeddings(model="llama3.2")
vector_store=InMemoryVectorStore(embeddings)

model=OllamaLLM(model="gemma3:4b")

def upload_pdf(file):
    with open(pdfs_dir+file.name, "wb") as f:
        f.write(file.getbuffer())
    
def load_pdf(path):
    elements=partition_pdf(
        path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=fig_dir
    )

    text_elements=[element.text for element in elements if element.category not in ["Image", "Table"]]

    for f in os.listdir(fig_dir):
        extracted_text=extract_text(fig_dir+f)
        text_elements.append(extracted_text)

    return "\n\n".join(text_elements)

def extract_text(path):
    model_with_image_context=model.bind(images=[path])
    return model_with_image_context.invoke("Tell me what you can see in this picture")

def split_text(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_text(text)

def index_docs(texts):
    vector_store.add_texts(texts)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context="\n\n".join([doc.page_content for doc in documents])
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model

    return chain.invoke({"question": question, "context": context})

uploaded_file=st.file_uploader(
    "Upload pdf file",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    text=load_pdf(pdfs_dir+uploaded_file.name)
    chunked_texts=split_text(text)
    index_docs(chunked_texts)

    question=st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_docs=retrieve_docs(question)
        answer=answer_question(question, related_docs)
        st.chat_message("assistant").write(answer)