from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


# Sales prompt template
sales_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Provide a response for this sales-related query: {query}"),
    ]
)

# Support prompt template
support_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Provide a response for this support-related query: {query}"),
    ]
)

# Feedback prompt template
feedback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Respond to this feedback query: {query}"),
    ]
)

# General prompt template
general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Provide a general response to this query: {query}"),
    ]
)
# branching

# Define branching for different types of queries
category_branches = RunnableBranch(
    (lambda x: "sales" in x, sales_prompt | model | StrOutputParser()),
    (lambda x: "support" in x, support_prompt | model | StrOutputParser()),
    (lambda x: "feedback" in x, feedback_prompt | model | StrOutputParser()),
    general_prompt | model | StrOutputParser(),  # Default to general response
)

# Identifying the feedback-->
identify_query = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant trained to classify user queries."),
        (
            "human",
            "Classify this query as 'sales', 'support', 'feedback', or 'general': {query}",
        ),
    ]
)

# Create the identify chain and connect it with the category branches
identify_chain = identify_query | model | StrOutputParser()
chain = identify_chain | category_branches


# review = "The product is okay. It works as expected but nothing exceptional."


# result=chain.invoke({"feedback": review})

# print(result)

chat_history = []
human_history = []
system_message = SystemMessage(content="You are good model giving me free service")
chat_history.append(system_message)  # Updating the chat history

# Chat Loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    human_history.append(query)
    result = model.invoke(query)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

result = chain.invoke({"query": human_history})
print(f"Whole conversation feedback: \n\n {result}")
