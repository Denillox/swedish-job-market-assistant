# assistant/agent.py
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from assistant.tools import (
    get_top_skills_tool,
    get_top_employers_tool,
    get_top_locations_tool,
    get_workplace_type_distribution_tool,
    get_experience_distribution_tool,
)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

tools = [
    get_top_skills_tool,
    get_top_employers_tool,
    get_top_locations_tool,
    get_workplace_type_distribution_tool,
    get_experience_distribution_tool,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that answers questions about the Swedish AI/Data job market using the provided tools.",
)