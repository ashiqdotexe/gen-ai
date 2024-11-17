import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader  
from langchain.schema import Document  # Import the Document class
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=api_key)
index_name = "chatinit"

print("\nQuery Part")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index = pc.Index(index_name)
vector_Store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_Store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# Sales, Support, Feedback, General prompt templates
sales_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "Provide a response for this sales-related query: {query}")]
)
support_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "Provide a response for this support-related query: {query}")]
)
feedback_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "Respond to this feedback query: {query}")]
)
general_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "Provide a general response to this query: {query}")]
)

# Define branching for different types of queries with retriever
category_branches = RunnableBranch(
    (
        lambda x: "sales" in x,
        sales_prompt | model | StrOutputParser()
    ),
    (
        lambda x: "support" in x,
        support_prompt | model | StrOutputParser()
    ),
    (
        lambda x: "feedback" in x,
        feedback_prompt | model | StrOutputParser()
    ),
    general_prompt | model | StrOutputParser()  # Default to general response
)

# Identify query type: sales, support, feedback, or general
identify_query = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant trained to classify user queries."), 
     ("human", "Classify this query as 'sales', 'support', 'feedback', or 'general': {query}")]
)

# Create the identify chain and connect it with the category branches
identify_chain = identify_query | model | StrOutputParser()
chain1 = identify_chain | category_branches 
chain = chain1 | retriever
# Initialize chat history
chat_history = []
human_history = []
system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)  # Updating the chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    human_history.append(query)

    # Classify query and route to appropriate response
    try:
        result = chain.invoke({"query": query})
        response = result
        chat_history.append(AIMessage(content=response))
        print(f"AI: {response}")
    except Exception as e:
        print(f"Error: {e}")
        continue

    # Optionally review the entire conversation feedback
    feedback_result = chain.invoke({"query": human_history})
    print(f"Whole conversation feedback: \n\n{feedback_result}")