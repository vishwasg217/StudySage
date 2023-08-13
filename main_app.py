from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values
import streamlit as st
from PyPDF2 import PdfReader

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

model = ChatOpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

def process_pdf(pdf):
    file = PdfReader(pdf)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    text = ''

    for page in file.pages:
        text += page.extract_text()

    text = splitter.split_text(text)

    return text

def database(splitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = Chroma.from_texts(splitted_text, embeddings)
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

    st.title("PDF Q&A")

    if "process_pdf" not in st.session_state:
        st.session_state.process_pdf = False

    pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if st.sidebar.button("Process PDF"):
        st.session_state.process_pdf = True
        with st.spinner("Processing PDF..."):
            splitted_text = process_pdf(pdf)
        db = database(splitted_text)
        # conversation(db)
        retriever = db.as_retriever()
        st.session_state.memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=5)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, memory=st.session_state.memory)


    if st.session_state.process_pdf:
        query = st.chat_input("Ask a question")
        if query:
            handle_query(query)


# what is the total number of AI publications?
# What is this number divided by 2?
    
    

    

