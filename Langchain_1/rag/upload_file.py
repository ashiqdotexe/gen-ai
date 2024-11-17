import os
from fastapi import FastAPI, File, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import pinecone as pc
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
import tempfile
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyMuPDFLoader
# Initialize FastAPI
app = FastAPI()

# Pinecone initialization
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")  # Ensure you have this in your environment variables

# Create a Pinecone instance
pc = Pinecone(api_key=api_key)

# Ensure the index exists
index_name = "chatinit"  # Replace with your actual index name

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Create FastAPI app
app = FastAPI()

@app.post("/upload-pdf/{namespace}")
async def upload_pdf(namespace: str, file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    try:
        # Load the PDF file
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()

        # Extract text and split it using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Adjust sizes as needed
        texts = text_splitter.split_documents(documents)

        # Generate embeddings
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  # or your preferred embeddings model
        embeddings = embeddings_model.embed_documents([text.page_content for text in texts])

        vectorstore=PineconeVectorStore(index_name=index_name,embedding=embeddings_model)

        # Store the embeddings in Pinecone with a specified namespace
        for i, embedding in enumerate(embeddings):
            vectorstore.add_texts(
                texts=[texts[i].page_content],  # Text you want to store
                embeddings=[embedding],  # Corresponding embedding
                metadatas=[{'source': file.filename, 'name': f'new_name_{i}'}],  # Change the name here
                namespace=namespace  # Use the specified namespace
            )

        return JSONResponse(content={"message": "PDF uploaded and stored in vector store."})

    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
print("Ran successfully")
# Run the FastAPI app using 'uvicorn script_name:app --reload'

