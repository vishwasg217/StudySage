from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
from PyPDF2 import PdfReader


pdf = st.file_uploader("pdf", type=['pdf'])

if st.button("pdf"):
    file = PdfReader(pdf)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    text = ''

    for page in file.pages:
        text += page.extract_text()

    text = splitter.split_text(text)
    st.write(text)
