import streamlit as st
from langchain.llms import OpenAI

questions = st.text_area("Enter Text")

if questions != "":

    OPEN_AI_API = "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"

    model = OpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

    answers = model(f"format the questions into a list {questions}")

    st.write(questions)
    st.write(answers)