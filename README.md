# Q&A-and-RAG-with-SQL-and-TabularData:
Q&A-and-RAG-with-SQL-and-TabularData is a chatbot project that utilizes GPT 4, Langchain, SQLite, and ChromaDB and allows users to interact (perform Q&A and RAG) with SQL databases, CSV, and XLSX files using natural language.

# Features:

1. Chat with SQL data.
2. Chat with preprocessed CSV and XLSX data.
3. RAG with Tabular datasets.

# General structure of the projects:
Project-folder
  ├── README.md           <- The top-level README for developers using this project.
  ├── .env                <- dotenv file for local configuration.
  ├── .here               <- Marker for project root.
  ├── configs             <- Holds yml files for project configs
  ├── data                <- Contains the sample data for the project.
  ├── src                 <- Contains the source code(s) for executing the project.
  └── images              <- Contains all the images used in the user interface and the README file.

# Key Notes:
## Key Note 1: 
All the project uses OpenAI models.

## Key Note 2 : 
When we interact with databases using LLM agents, good informative column names can help the agents to navigate easier through the database.

## Key Note 3: 
When we interact with sensitive databases using LLM agents, remember to NOT use the database with WRITE privileges. Use only READ and limit the scope. Otherwise your user can manupulate the data (e.g ask your chain to delete data).

## Key Note 4: 
Familiarity with database query languages such as Pandas for Python, SQL, and Cypher can enhance the user's ability to ask more better questions and have a richer interaction with the graph agent.
