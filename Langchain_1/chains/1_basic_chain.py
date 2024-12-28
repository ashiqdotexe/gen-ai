from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)
chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic": "doctors", "joke_count": 3})
print(result)
