from langchain_openai import ChatOpenAI
from youtube_tools import get_youtube_parser_tool  # Adjust the import based on your file structure

# Initialize the language model
llm = ChatOpenAI()

# Get the youtube_parser tool
youtube_parser_tool = get_youtube_parser_tool(llm)

# Define the YouTube video URL
video_url = "What is the description of this video https://www.youtube.com/watch?v=dQw4w9WgXcQ by Krish naik on datascience?"

# Call the tool with the video URL
result = youtube_parser_tool.invoke({
    "video_url": video_url,
    # "context": ["A youtube video by Krish Naik on DataScience"]
})

# Print the result
print(result)