from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values
import streamlit as st
from PyPDF2 import PdfReader


def process_pdf(pdfs):
    splitter = CharacterTextSplitter(separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len)
    
    text = ''

    for pdf in pdfs:
        file = PdfReader(pdf)
        for page in file.pages:
            text += str(page.extract_text())

    splitted_text = splitter.split_text(text)

    print(splitted_text)

    return splitted_text

def database(splitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.OPEN_AI_API)
    db = Chroma.from_texts(splitted_text, embeddings)
    return db



