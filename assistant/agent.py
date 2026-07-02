# assistant/agent.py
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError, ClientError

from assistant.tools import (
    get_top_skills_tool,
    get_top_employers_tool,
    get_top_locations_tool,
    get_workplace_type_distribution_tool,
    get_experience_distribution_tool,
    retrieve_market_context_tool,
)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
)

tools = [
    get_top_skills_tool,
    get_top_employers_tool,
    get_top_locations_tool,
    get_workplace_type_distribution_tool,
    get_experience_distribution_tool,
    retrieve_market_context_tool,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that answers questions about the Swedish AI/Data job market using the provided tools.",
)

@retry(
    retry=retry_if_exception_type((ServerError, ClientError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)

def _invoke_agent(messages: list):
    return agent.invoke({"messages": messages})


def chat(message: str, history: list) -> tuple[str, list]:
    """
    Send a message and get a response, maintaining conversation history.

    Args:
        message: the user's new message
        history: list of previous messages in LangChain format,
                  e.g. [("user", "..."), ("assistant", "...")]

    Returns:
        tuple of (response string, updated history list)
    """
    full_history = history.copy() + [("user", message)]
    response = _invoke_agent(full_history)
    reply = _extract_text(response["messages"][-1].content)
    updated_history = full_history + [("assistant", reply)]
    return reply, updated_history

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([_extract_text(item) for item in content])
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return ""
    return ""