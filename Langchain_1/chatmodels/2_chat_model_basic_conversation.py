from langchain_huggingface import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

messages = [
    SystemMessage(content="Solve this mathmatical problem"),
    HumanMessage(content="What 7 multiply 5"),
    AIMessage(content="7 multiplied by 5 is **35**"),
    HumanMessage(content="what is 3 multiply 7"),
]
result = model.invoke(messages)
print(f"Result is: {result.content}")
