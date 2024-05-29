import re
from typing import List, Optional

import requests
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

_YOUTUBE_DESCRIPTION = (
    "youtube_parser(video_url: str, context: Optional[list[str]]) -> dict:\n"
    " - Extracts metadata and comments from the provided YouTube video URL.\n"
    " - `video_url` should be a valid YouTube video URL.\n"
    " - You can optionally provide a list of strings as `context` to help the agent extract specific information.\n"
    " - The returned dictionary contains video title, description, views, likes, dislikes, and comments.\n"
    " - Minimize the number of `youtube_parser` actions as much as possible.\n"
)

_YOUTUBE_SYSTEM_PROMPT = """Extract metadata and comments from the YouTube video URL provided. Use the output of this extraction to answer the question.

Video URL: ${{video_url}}
```json
${{metadata and comments in JSON format}}
Answer: ${{Answer}}

Begin.

Video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Extracted Metadata and Comments:

{{
  "title": "Rick Astley - Never Gonna Give You Up (Video)",
  "description": "Rick Astley's official music video for “Never Gonna Give You Up”...",
  "views": "1,000,000,000",
  "likes": "10,000,000",
  "dislikes": "500,000",
  "comments": [
    {{"author": "User1", "text": "Great song!"}},
    {{"author": "User2", "text": "Classic!"}},
    ...
  ]
}}
Answer: Extracted video metadata and comments successfully.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.
Use it to substitute into any ${{#}} variables or other words in the problem.
\n\n${context}\n\nNote that context variables are not defined in code yet.
You must extract the relevant numbers and directly put them in code."""


class YouTubeVideoData(BaseModel):
    """The extracted data from the YouTube video."""

    video_url: str = Field(
        ...,
        description="The URL of the YouTube video to extract data from.",
    )

    context: Optional[List[str]] = Field(
        None,
        description="Additional context to help the agent extract specific information.",
    )

def extract_youtube_data(video_url: str) -> dict:
    # Dummy implementation of YouTube data extraction.
    # You should replace this with actual API calls and data parsing logic.
    # For example, use YouTube Data API or scrape the page for metadata and comments.
    return {
    "title": "Sample Title",
    "description": "Sample description of the video.",
    "views": "12345",
    "likes": "678",
    "dislikes": "90",
    "comments": [
    {"author": "User1", "text": "Sample comment 1"},
    {"author": "User2", "text": "Sample comment 2"}
    ]
    }

def get_youtube_parser_tool(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system", _YOUTUBE_SYSTEM_PROMPT),
    ("user", "{video_url}"),
    MessagesPlaceholder(variable_name="context", optional=True),
    ]
    )

    extractor = create_structured_output_runnable(YouTubeVideoData, llm, prompt)

    def parse_youtube_video(
        video_url: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"video_url": video_url}
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
            return extract_youtube_data(data_model.video_url)
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="youtube_parser",
        func=parse_youtube_video,
        description=_YOUTUBE_DESCRIPTION,
    )