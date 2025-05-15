import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = './pdfs/'
figures_directory = './figures/'

embeddings = OllamaEmbeddings(model="llama3.2")
#embeddings = OllamaEmbeddings(model="gemma3:4b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="gemma3:4b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)

    return "\n\n".join(text_elements)

    # # using pymupdf
    # doc=fitz.open(file_path)
    # text_elements=[]
    # for page in doc:
    #     text_elements.append(page.get_text())
    #     image_list=page.get_images()
    #     for img_index, img in enumerate(image_list):
    #         xref=img[0]
    #         base_img=doc.extract_image(xref)
    #         image_bytes=base_img["image"]
    #         #image_ext=base_img["ext"]
    #         image_name=f"image{img_index}.png"
    #         image_path=os.path.join(figures_directory, image_name)
    #         with open(image_path, "wb") as image_file:
    #             image_file.write(image_bytes)
    #     table_list=page.find_tables()
    #     # if table_list.tables:
    #     #     extracted_table=table_list.extract()
    #     #     text_elements.append(extracted_table)
    #     for i, table in enumerate(table_list.tables):
    #         bbox = table.bbox  # (x0, y0, x1, y1)
            
    #         # Create a clip of the table area
    #         mat = fitz.Matrix(2, 2)  # upscale resolution (2x)
    #         pix = page.get_pixmap(matrix=mat, clip=bbox)
            
    #         # Save as image
    #         image_filename = f"table{i}.png"
    #         image_path = os.path.join(figures_directory, image_filename)
    #         pix.save(image_path)

    # for file in os.listdir(figures_directory):
    #     extracted_text = extract_text(figures_directory + file)
    #     text_elements.append(extracted_text)

    # return "\n\n".join(text_elements)

def extract_text(file_path):
    model_with_image_context = model.bind(images=[file_path])
    return model_with_image_context.invoke("Tell me what do you see in this picture.")

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
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
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

st.set_page_config(page_title="Ask Questions About Your PDF", layout="centered")
st.title("📄 RAG PDF Q&A with LangChain + Ollama")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:
    with st.spinner("Processing PDF"):
        upload_pdf(uploaded_file)
        text = load_pdf(pdfs_directory + uploaded_file.name)
        chunked_texts = split_text(text)
        index_docs(chunked_texts)
    st.success("PDF processed successfully")
    question = st.chat_input()
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)