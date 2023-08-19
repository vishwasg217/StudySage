from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

OPEN_AI_API = "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"

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0, openai_api_key=OPEN_AI_API)

_input = prompt.format(subject="ice cream flavors")
output = model(_input)

print(output_parser.parse(output))
