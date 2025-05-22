import os
import streamlit as st
from backend import (
    load_environment_variables,
    manage_faiss_database,
    setup_retrieval_chain,
    )
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
def main():
    # Setup
    project_root = os.path.dirname(os.path.abspath(__file__))  
    dotenv_path = os.path.join(project_root, '.env')  # Correct path to the .env file
    # Load SQL Database
    db_path = os.path.join(project_root, 'data', 'database.db')
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    load_environment_variables(dotenv_path)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
    OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
        openai_api_type="azure",
        chunk_size=64
    )

    llm = AzureChatOpenAI(
        openai_api_version=OPENAI_DEPLOYMENT_VERSION,
        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
        openai_api_key=OPENAI_API_KEY,
        azure_deployment="gpt-4",
        model_name="gpt-4",
        temperature=0.0
    )

    # Streamlit UI
    st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")
    logo_path = os.path.join(project_root, "Bosch.png")
    col1, col2 = st.columns([0.7, 1])
    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
    with col2:
        st.markdown("<h1 style='font-size:30px; color:#333; font-weight:bold;'>InsightBot</h1>", unsafe_allow_html=True)

    st.sidebar.title("User Selection")
    user_type = st.sidebar.selectbox("Select data source:", ["---", "Smart search", "Contact for AM"], index=0)
    
    if user_type == "---":
        st.warning("Please select a data source from the sidebar to continue.")
        st.stop()

    qa_chain = None  # Initialize qa_chain to avoid UnboundLocalError

    if user_type != "Contact for AM":    
        data_path = os.path.join(project_root, 'data', 'pdf files', user_type)
        faiss_save_path = os.path.join(data_path, 'faiss_store')
        vector_db = manage_faiss_database(data_path, embeddings, faiss_save_path)  
    
        # Retrieval Chain
        retriever = vector_db.as_retriever()
        qa_chain = setup_retrieval_chain(OPENAI_API_KEY, OPENAI_DEPLOYMENT_ENDPOINT, retriever)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything!")  # Single common input box

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
     
        if user_type == "Contact for AM":
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

            agent_executor = initialize_agent(
                tools=toolkit.get_tools(),
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # More predictable than openai-tools
                verbose=True,
                agent_kwargs={
                    "prefix": """You are an assistant who can answer questions using the following database schema.
            Only use the columns and tables mentioned below. Do not make up table or column names.

            Schema:
            Table: IdM-Reporting-2-OrganizationalR
            - full_name
            - organization_name
            - organization_type
            - ge
            - organizational_role
            Important Instructions:
            - The column 'organizational_role' always contains the value "Contact for AIM" for all rows.
            - So if the user asks for a "contact", it always refers to "Contact for AIM".
            - To find the contact for an organization, filter only by `organization_name`.
            - Do not filter by `organizational_role`  — these are the same across all rows.
            - Always wrap the table name in double quotes like this: "IdM-Reporting-2-OrganizationalR".
            - Always use LOWER(organization_name) = LOWER('user_input') to make case-insensitive matches.
            Examples:
            Q: Who is the contact for AIM for organization GS/HRZ3?
            A: SELECT full_name FROM "IdM-Reporting-2-OrganizationalR" WHERE organization_name = 'GS/HRZ3';

            Q: Who is the contact person for C/SCL?
            A: SELECT full_name FROM "IdM-Reporting-2-OrganizationalR" WHERE organization_name = 'C/SCL';
            Q: Who can i contact for topics related to office C/SCL?
            A: SELECT full_name FROM "IdM-Reporting-2-OrganizationalR" WHERE organization_name = 'C/SCL';
            If no result is found, say: "No contact found for this organization."

            Respond concisely and use only valid SQL syntax for SQLite."""
                }
            )
            # agent_executor = create_sql_agent(
            # llm=llm,
            # db=db,
            # agent_type="openai-tools",
            # verbose=True  # You can set this to False for production
            # )    
            response = agent_executor.invoke({"input": prompt})
            answer = response.get("output", "I'm not sure how to answer that.")
            
        else:
            # FAISS-based Retrieval
            chat_history = []
            response = qa_chain({'question': prompt, 'chat_history': chat_history})
            answer = response.get('answer', "I'm not sure how to answer that.") if isinstance(response, dict) else str(response)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
if __name__ == "__main__":
    main()