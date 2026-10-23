from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

from rich import print

#1 creating a tool

@tool
def get_text_length(text: str) -> int:
  """returns the number of character in a given text"""
  return len(text)

llm = ChatGoogleGenerativeAI()

#tool binding
llm_with_tool = llm.bind_tools([get_text_length])

