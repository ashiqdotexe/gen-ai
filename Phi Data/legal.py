from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from phi.assistants import Agent
from phi.memory import VectorStoreMemory
import pinecone

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="your_pinecone_env")
index = pinecone.Index("your_pinecone_index")

# Define Memory for Agents
global_memory = VectorStoreMemory(index, namespace="global")
premium_memory = VectorStoreMemory(index, namespace="premium")

# Define Prompt Templates
global_prompt_template = PromptTemplate(
    template="""
    You are a legal analyst providing insights for general legal queries across different jurisdictions.
    Ensure your responses are well-researched and cover broad legal aspects relevant to a global audience.
    Focus on accuracy, legality, and references when applicable.
    Query: {query}
    """,
    input_variables=["query"],
)

premium_prompt_template = PromptTemplate(
    template="""
    You are a specialized legal consultant for premium users, providing in-depth legal analysis.
    Your responses should include case law, precedents, and jurisdiction-specific regulations.
    Offer expert-level insights tailored to the specific legal scenario provided in the query.
    Query: {query}
    """,
    input_variables=["query"],
)

# Define Agents using PHI Data Framework
global_agent = Agent(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
    memory=global_memory,
    prompt_template=global_prompt_template,
)

premium_agent = Agent(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
    memory=premium_memory,
    prompt_template=premium_prompt_template,
)


class QueryRequest(BaseModel):
    user_type: str  # "global" or "premium"
    query: str


@app.post("/analyze")
def analyze_legal_data(request: QueryRequest):
    if request.user_type == "global":
        response = global_agent.run({"query": request.query})
    elif request.user_type == "premium":
        response = premium_agent.run({"query": request.query})
    else:
        raise HTTPException(
            status_code=400, detail="Invalid user_type. Use 'global' or 'premium'"
        )

    return {"user_type": request.user_type, "response": response}


# Additional API endpoints for future expansions
@app.get("/health")
def health_check():
    return {"status": "API is running smoothly"}


@app.get("/agent-info/{user_type}")
def get_agent_info(user_type: str):
    if user_type == "global":
        return {
            "user_type": "global",
            "description": "Handles broad legal queries with a global perspective.",
        }
    elif user_type == "premium":
        return {
            "user_type": "premium",
            "description": "Provides expert-level legal insights with detailed case studies and references.",
        }
    else:
        raise HTTPException(
            status_code=400, detail="Invalid user_type. Use 'global' or 'premium'"
        )
