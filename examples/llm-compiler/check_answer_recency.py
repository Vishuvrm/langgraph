import re
from datetime import datetime, date
from typing import List, Optional, Union
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.output_parsers import JsonOutputToolsParser, JsonOutputKeyToolsParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import END
import dateparser


class DateExtraction(BaseModel):
    """Model to extract date information from a response."""
    date: Optional[str] = Field(
        description="The date mentioned in the response in the format dd-mm-yyyy.",
        default=""
    )
    # context: Optional[list[str]] = Field(
    #     ...,
    #     description=["Context related to the date provided which help agent decide if answer is relevant."]
    # )

@tool(args_schema=DateExtraction)
def extract_date(date:Optional[str]):
    "Tool to extract date information from a response in the format dd-mm-yyyy. If no date is mentioned, return ''"
    pass

def answer_recent_node(messages: List[BaseMessage], config=None):
    last_message = messages[-1]
    # Initialize the language model
    llm = ChatOpenAI()
    extractor = llm.bind_tools([extract_date], tool_choice="extract_date")|JsonOutputToolsParser()

    date_str = extractor.invoke(last_message.content)

    if date_str[0]["args"]:
        date_str = date_str[0]["args"]["date"]
        try:
            try:
                date_parsed = datetime.strptime(date_str, r"%d-%m-%Y").date()
            except:
                date_parsed = dateparser.parse(date_str, (r"%d-%m-%Y", r"%d/%m/%Y")).date()
        except:
            return last_message
        
        if date_parsed >= datetime.today().date():
            return last_message
        return f"Thought: The answer seems to be outdated w.r.t. current date, which is {datetime.today().date()}. Please provide the most recent answer. Replan if necessary."
    else:
        return last_message