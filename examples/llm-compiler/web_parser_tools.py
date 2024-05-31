import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

_WEBPARSER_DESCRIPTION = (
    "web_parser(website_url: str, context: Optional[list[str]]) -> dict:\n"
    " - Extracts all the text content from the provided URL.\n"
    " - `website_url` should be a valid website URL.\n"
    " - You can optionally provide a list of strings as `context` to help the agent extract specific information.\n"
    " - The returned output contains website text content.\n"
    " - Minimize the number of `web_parser` actions as much as possible.\n"
)

_WEBSITE_SYSTEM_PROMPT = """Extract website text content from the website URL provided. Use the output of this extraction to answer the question.

Wesite URL: ${{website_url}}
```json
${{extracted website content}}
Answer: ${{Answer}}

Begin.

Website URL: https://www.example.com/
Extracted Metadata and Comments:

{{This is the sample text from the website.}}
Answer: Extracted website content successfully.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.
Use it to substitute into any ${{#}} variables or other words in the problem.
\n\n${context}\n\nNote that context variables are not defined in code yet.
You must extract the relevant numbers and directly put them in code."""


class WebsiteExtractor(BaseModel):
    """The extracted data from the website URL."""

    website_url: str = Field(
        ...,
        description="The URL of the webite to extract data from.",
    )

    context: Optional[List[str]] = Field(
        None,
        description="Additional context to help the agent extract specific information.",
    )

def extract_website_content(website_url: str) -> dict:
    # Dummy implementation of YouTube data extraction.
    # You should replace this with actual API calls and data parsing logic.
    # For example, use YouTube Data API or scrape the page for metadata and comments.
    try:
        response = requests.get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        text_content = soup.get_text(separator='\n', strip=True)
        return text_content
    except:
        return "Sorry! I can't access this website."
    

def extract_website_content(url: str) -> str:
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Function to extract text and preserve the structure
        def extract_text(element, level=0):
            text = ""
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span', 'button', 'a', 'img']:
                text += '  ' * level + element.get_text(strip=True) + '\n'
            for child in element.children:
                if child.name:
                    text += extract_text(child, level + 1)
            return text

        # Start extracting from the body
        body = soup.body
        formatted_content = extract_text(body)

        # formatted_content = re.sub(r"[\n]+", "\n", re.sub(r"[\s]+", ' ', formatted_content))

        return formatted_content

    except requests.exceptions.RequestException as e:
        return f"Error fetching the website content: {e}"

def get_web_parser_tool(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system", _WEBSITE_SYSTEM_PROMPT),
    ("user", "{website_url}"),
    MessagesPlaceholder(variable_name="context", optional=True),
    ]
    )

    extractor = create_structured_output_runnable(WebsiteExtractor, llm, prompt)

    def parse_website(
        website_url: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"website_url": website_url}
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
                chain_input["context"] = [SystemMessage(content=context_str)]
        data_model = extractor.invoke(chain_input, config)
        print("DATA MODEL =", data_model)
        try:
            web_content = extract_website_content(data_model.website_url).strip() or SystemMessage("Couldn't parse this url. I think there might be some protection. Please try some other way.")
            return web_content
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="web_parser",
        func=parse_website,
        description=_WEBPARSER_DESCRIPTION,
    )


if __name__ == "__main__":
    r = extract_website_content("https://twitter.com/elonmusk")
    print(r)