from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv

PRODUCT_ID = "chatmodel-langchain"
SESSION_ID = "new_user"
COLLECTION_ID = "langchain_save"

print("Connecting with client....")
client_id = firestore.Client(project=PRODUCT_ID)
print("Cloud connected")
# Retrivieng chat history
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID, collection=COLLECTION_ID, client=client_id
)

print("Chat history initilaized")
print("Chat history is: ", chat_history.messages)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Chat Loop
while True:
    human = input("You: ")
    if human.lower() == "exit":
        break

    chat_history.add_user_message(human)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
