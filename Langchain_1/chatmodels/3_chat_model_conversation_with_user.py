from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

#Chat history will store here
chat_history=[]

system_message=SystemMessage(content="You are good model giving me free service")
chat_history.append(system_message)#Updating the chat history

#Chat Loop
while True:
    query=input("You: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))

    result=model.invoke(query)
    response=result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("______Chat History______")
print(chat_history)