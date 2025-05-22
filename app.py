import os
import streamlit as st
from backend import (
    load_environment_variables,
    manage_faiss_database,
    setup_retrieval_chain,
    )
from langchain_openai import AzureOpenAIEmbeddings


from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_openai import AzureChatOpenAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def main():
    # Setup
    project_root = os.path.dirname(__file__) 
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
        azure_deployment="gpt-35-turbo",
        model_name="gpt-35-turbo",
        temperature=0.0
    )

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDatabaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )

    answer = answer_prompt | llm | StrOutputParser()
    sql_chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    # Streamlit UI
    st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")
    logo_path = os.path.join(project_root, "images", "Bosch.png")
    col1, col2 = st.columns([0.7, 1])
    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
    with col2:
        st.markdown("<h1 style='font-size:30px; color:#333; font-weight:bold;'>InsightBot</h1>", unsafe_allow_html=True)

    st.sidebar.title("User Selection")
    user_type = st.sidebar.selectbox("Select data source:", ["---", "C", "GS", "M", "SQL"], index=0)
    
    if user_type == "---":
        st.warning("Please select a data source from the sidebar to continue.")
        st.stop()

    qa_chain = None  # Initialize qa_chain to avoid UnboundLocalError

    if user_type != "SQL":    
        data_path = os.path.join(project_root, 'data', 'pdf files', user_type)
        faiss_save_path = os.path.join(data_path, 'faiss_store')
        db = manage_faiss_database(data_path, embeddings, faiss_save_path)  
    
        # Retrieval Chain
        retriever = db.as_retriever()
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

        if user_type == "SQL":
            # SQL Query Execution
            response = sql_chain.invoke({"question": prompt})
            answer = response.get("answer", "I'm not sure how to answer that.") if isinstance(response, dict) else str(response)
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
