
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


import streamlit as st
from pydantic import BaseModel, Field, validator
from typing import List

from utils import process_pdf, database

llm = None



def generate_answers(splitted_text, db):
    pass


template = """
You have the task of generating answers for the following questions from the notes given.
The answers should be explained in a simple manner with examples.


Here are the questions you need to generate answers for:

==================
Questions: {questions}

==================

{output_format_instructions}

"""

# creating a Pydantic model to parse the output
class GenerateAnswers(BaseModel):
    question: str = Field(description="Questions")
    answer: List[str] = Field(description="Answers")

    # @validator('summary', allow_reuse=True)
    # def has_three_or_more_lines(cls, list_of_lines):
    #     if len(list_of_lines) < 3:
    #         raise ValueError("Generated summary has less than three bullet points!")
    #     return list_of_lines


def generate_answers(splitted_text, formatted_questions, db):
    parser = PydanticOutputParser(pydantic_object=GenerateAnswers)  

    prompt = PromptTemplate(template=template, 
                            input_variables=['questions'],
                            partial_variables={"output_format_instructions": parser.get_format_instructions()},  # used to format the output
    )
    formatted_prompt = prompt.format_prompt(questions=formatted_questions)
    messages = [HumanMessage(content=formatted_prompt.to_string())]
    response = llm(messages)

    return response


def format_questions():
    formatted_questions = llm(f"format the questions into a list {st.session_state.questions}")
    return formatted_questions


if __name__ == "__main__":

    st.session_state.OPEN_AI_API = st.text_input("OpenAI API Key", key="OPEN_AI_API")

    if st.session_state.OPEN_AI_API == "":
        llm = OpenAI(openai_api_key=st.session_state.OPEN_AI_API, temperature=0.5)
        notes = st.file_uploader("Upload your notes", type=["pdf"])
        if st.sidebar.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                splitted_text = process_pdf(notes)
                db = database(splitted_text)

            st.session_state.questions = st.text_area("Enter your questions here", key="question")
            formatted_questions = format_questions()


            answers = generate_answers(splitted_text, formatted_questions, db)

            