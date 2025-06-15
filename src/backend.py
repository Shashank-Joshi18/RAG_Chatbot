import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities import SQLDatabase

# Environment Setup
def load_environment_variables(env_path):
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"The .env file at {env_path} was not found.")
    load_dotenv(env_path)

def initialize_openai(api_key, api_endpoint):
    # Create an instance of AzureOpenAI
    azure_openai = AzureOpenAI(
        api_key=api_key,
        api_base=api_endpoint,
        api_version="2025-01-01-preview"
    )
    return azure_openai
# PDF Processing and FAISS Management
def process_pdfs(folder_path):
    all_pages = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            all_pages.extend(pages)
            print(f"Processed {filename} - {len(pages)} pages loaded.")
    return all_pages

def manage_faiss_database(folder_path, embeddings, faiss_save_path):
    if not os.path.exists(faiss_save_path):
        pages = process_pdfs(folder_path)
        if not pages:
            raise ValueError("No pages were loaded from the PDFs.")
        db = FAISS.from_documents(documents=pages, embedding=embeddings)
        db.save_local(faiss_save_path)
    else:
        db = FAISS.load_local(faiss_save_path, embeddings, allow_dangerous_deserialization=True)
    return db

def connect_sql_database(db_path):
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Retrieval Chain Setup
def create_prompt():
    return PromptTemplate.from_template("""
    You are a helpful assistant. Answer the following question using the context provided.
    Question: {question}
    Context: {chat_history}
    Answer:""")

def setup_retrieval_chain(api_key, api_endpoint, retriever):
    llm = AzureChatOpenAI(
        deployment_name="gpt-4",
        model_name="gpt-4",
        azure_endpoint=api_endpoint,
        openai_api_version="2025-01-01-preview",
        openai_api_key=api_key,
        openai_api_type="azure",
        streaming=True
    )

    prompt_template = create_prompt()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt_template,
        return_source_documents=True,
        verbose=False
    )