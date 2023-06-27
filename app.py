import os
import pickle
from dotenv import dotenv_values
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

config = dotenv_values(".env")
OPENAI_API_KEY = config["OPENAI_API_KEY"]

# llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.9, model_name="gpt-3.5-turbo")
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

st.set_page_config(layout="wide", page_title="StudySage", page_icon="📚")
st.title("StudySage - Your AI Study Buddy")

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(chunks):
    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)
    return knowledge_base

def get_converation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})
        # response = llm(st.session_state.chat_history)
    st.session_state.chat_history.append(
        AIMessage(content=response['answer']))
    
    messages = st.session_state.get('chat_history', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')



if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant.")
        ]



pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if st.sidebar.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        text = get_pdf_text(pdf)
        chunks = get_chunks(text)

        # create embeddings
        knowledge_base = vector_store(chunks)
        st.session_state.conversation = get_converation_chain(knowledge_base)

user_input = st.sidebar.text_input("Ask questions")
if user_input:
    handle_user_input(user_input)
        
    # st.session_state.conversation
st.sidebar.divider()
st.sidebar.markdown("""
                    Made with ❤️ by [Vishwas Gowda](https://www.linkedin.com/in/vishwasgowda217/)""")


        