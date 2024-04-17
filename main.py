#print("Om Shree Ganeshay Namah !")

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv  
import argparse
import warnings

warnings.filterwarnings("ignore")#, category=DeprecationWarning)

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task",default="return a list of numbers")
parser.add_argument("--language",default="python")
args = parser.parse_args()


llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language","task"]
)

unittest_prompt = PromptTemplate(
    template="Write a short unit test for following {language} code:\n{code}",
    input_variables=["language","code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

unittest_chain = LLMChain(
    llm=llm,
    prompt=unittest_prompt,
    output_key="unittest"
)

chain = SequentialChain(
    chains=[code_chain,unittest_chain],
    input_variables=["task","language"],
    output_variables=["unittest","code"]
)

result = chain({
    "language": args.language,
    "task": args.task
})

print("GENERATED CODE >>>>>>>>")
print(result["code"])

print("GENERATED UNIT TEST >>>>>>>>")
print(result["unittest"])