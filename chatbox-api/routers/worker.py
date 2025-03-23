import asyncio
import os
import multiprocessing
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents  import Document
from docx import Document as DocxDocument
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect,File, UploadFile
from langdetect import detect
from models.question import Question
import numpy as np
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from langchain.schema import SystemMessage, HumanMessage


CONNECTION_STRING = os.getenv("VECTOR_DB_URL")

# Initialize FastAPI app
router = APIRouter()

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")

# Initialize chat model (GPT-4o)
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=API_KEY
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=API_KEY
)

# Global vector_store
finance_db = None
hr_db = None

def load_docx(file_path):
    """
    Custom function to load content from a .docx file and return it as LangChain Document objects.
    """
    doc = DocxDocument(file_path)
    content = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():  # Ignore empty paragraphs
            content.append(paragraph.text.strip())

    # Return as a list of LangChain Document objects
    return [Document(page_content="\n".join(content), metadata={"source": str(file_path)})]

def load_and_embed_pdfs_and_docs(vector_store, folder, collection_name):
    """
    Load PDFs and DOCX files from a folder and create embeddings.
    """
    docs = []
    pdf_files = Path(folder).glob("*.pdf")
    docx_files = Path(folder).glob("*.docx")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        docs.extend(documents)

    for docx_file in docx_files:
        documents = load_docx(str(docx_file))
        docs.extend(documents)
    
    csv_files = Path(folder).glob("*.csv")
    # for csv_file in csv_files:
    #     df = pd.read_csv(csv_file)
        
    #     for _, row in df.iterrows():
    #         content = " | ".join(map(str, row.values))
    #         docs.append(Document(page_content=content))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        content = df.to_csv(index=False)
        
        docs.append(Document(page_content=content, metadata={"source": str(csv_file)}))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    if vector_store is None:
        vector_store = PGVector.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
    else:
        vector_store.add_documents(split_docs)

    return vector_store



def classify_with_llm(question):
    prompt = [
        SystemMessage(content="You are an AI assistant that classifies questions into 'HR' or 'Finance'."),
        HumanMessage(content=f"Question: {question}\nCategory:")
    ]
    response = chat_model.invoke(prompt).content.strip()
    return "HR" if "HR" in response else "Finance"

def retrieve_answer(question, category):    
    if category == "HR":
        results = hr_db.similarity_search(question, k=1)
    else:
        results = finance_db.similarity_search(question, k=1)
    
    return results[0] if results else None

def generate_answer(relevant_doc, question, category):
    print(relevant_doc)
    if category == "HR":
        system_message = "You are an HR assistant. Use the following context to answer the question."
    else:
        system_message = "You are an Finance assistant. Use the following context to answer the question."
    
    prompt = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Context: {relevant_doc.page_content}\n\nQuestion: {question}\nAnswer:")
        ]
    
    response = chat_model.invoke(prompt).content
    return response

@router.post("/ask")
async def ask_question(payload: Question):
    """
    API to process user question and return summary of the most relevant PDF.
    """
    pdf_folder = "./data_warehouse"
    question = payload.question

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    category = classify_with_llm(question)
    relevant_doc = retrieve_answer(question, category)

    if relevant_doc is None:
        return {"summary": "We still collecting data for your question."}
    
    result = generate_answer(relevant_doc, question, category)

    if not result:
        raise HTTPException(status_code=404, detail="No relevant document found.")

    return {
        "summary": result
    }

async def stream_lines(data):
    lines = data.split("\n")
    for line in lines:
        yield line
        await asyncio.sleep(1)

@router.on_event("startup")
async def initialize_vector_store():
    """
    Load the vector_store during app startup.
    """
    global finance_db
    global hr_db
    finance_db = load_and_embed_pdfs_and_docs(finance_db, "./data_warehouse/finance", "finance")
    hr_db = load_and_embed_pdfs_and_docs(hr_db, "./data_warehouse/hr", "hr")
    print("Vector store initialized!")
