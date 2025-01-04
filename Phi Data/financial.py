from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
import openai

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")
os.environ["PHI_API_KEY"] = os.getenv("PHI_DATA_API")

openai.api_key = os.getenv("OPENAI_API_KEY")
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    role="Search Engine",
    description="Use this tool to search the web for information.",
    instructions=[
        "Please search for the answer to the user's question. And provide sources"
    ],
    show_tool_calls=True,
    markdown=True,
)
finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    role="Financial Analyst",
    tools=[
        YFinanceTools(
            stock_price=True, analyst_recommendations=True, stock_fundamentals=True
        )
    ],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=[
        "Format your response using markdown and use tables to display data where possible."
    ],
)
multi_ai_agents = Agent(
    team=[web_agent, finance_agent],
    instructions=[
        "Please search for the answer to the user's question. And provide sources",
        "Format your response using markdown and use tables to display data where possible.",
    ],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agents.print_response(
    "Summerize analyst recommendation and share the leatest news for NVDA", stream=True
)
