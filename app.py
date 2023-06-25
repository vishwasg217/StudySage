import os
import pickle
from dotenv import dotenv_values
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

config = dotenv_values(".env")
OPENAI_API_KEY = config["OPENAI_API_KEY"]

llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.9, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


st.title("StudySage - Your AI Study Buddy")


pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

    # show user input

    # user_question = st.text_input("Ask a question about your PDF:")
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
    
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
            
        st.write(response)

    prompt1 = st.text_input("topic")

    prompt1_template = PromptTemplate(
        input_variables=['topic'],
        template="summarize the content in {topic}?", 
    )
    if prompt1:
        docs = knowledge_base.similarity_search(prompt1)

    
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=prompt1)
            print(cb)
        
        st.write(response)



# title_chain = LLMChain(llm=llm, prompt=prompt1_template)

# if prompt1:
#     docs = knowledge_base.similarity_search(prompt1)
#     response = title_chain.run(topic=prompt1, input_documents=docs)
#     st.write(response)