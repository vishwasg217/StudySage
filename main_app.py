from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values
import streamlit as st
from PyPDF2 import PdfReader
import pickle

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]


model = ChatOpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

def process_pdf(pdfs):
    splitter = CharacterTextSplitter(separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len)
    
    text = ''

    for pdf in pdfs:
        file = PdfReader(pdf)
        for page in file.pages:
            text += page.extract_text()

        splitted_text = splitter.split_text(text)

    return splitted_text

def database(splitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = Chroma.from_texts(splitted_text, embeddings)
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(db, f)
    return db


def handle_query(query: str):
    result = st.session_state.conversation({"question": query, "chat_history": ""})
    history = st.session_state.memory.load_memory_variables({})['chat_history']
    for i, msg in enumerate(history):
        if i%2 == 0:
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


if __name__ == "__main__":

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.set_page_config(layout="wide", page_title="StudySage", page_icon="📚")
    st.title(":books: StudySage - Your AI Study Buddy")


    if "process_pdf" not in st.session_state:
        st.session_state.process_pdf = False

    pdfs = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
    if st.sidebar.button("Process PDF"):
        st.session_state.process_pdf = True
        with st.spinner("Processing PDF..."):
            splitted_text = process_pdf(pdfs)
        db = database(splitted_text)
        st.session_state.memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=5)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(llm=model, retriever=db.as_retriever(), memory=st.session_state.memory)


    if st.session_state.process_pdf:
        query = st.chat_input("Ask a question")
        if query:
            handle_query(query)

    
    

    

