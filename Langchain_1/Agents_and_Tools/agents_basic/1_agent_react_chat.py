from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os


load_dotenv()

########PineCone Vectore store
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
pc = Pinecone(api_key=api_key)
index_name = "chatinit"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
index = pc.Index(index_name)
vectore_Store = PineconeVectorStore(index=index, embedding=embedding_model)
retriver = vectore_Store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.6},
)


def search_wiki(query):
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that"


###Import the tools you wanna work on
tool = [
    Tool(
        name="Wikipedia",
        func=search_wiki,
        description="Useful for when you need to know information about a topic",
    ),
]

##This is the prompt system gonna use
prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=llm, prompt=prompt, tools=tool)

agent_executer = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tool,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True,
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.aadd_messages(SystemMessage(content=initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    response = agent_executer.invoke({"input": user_input})

    print("Bot: ", response["output"])

    memory.chat_memory.add_message(AIMessage(content=response["output"]))
