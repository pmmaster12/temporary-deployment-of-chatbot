from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain_groq import ChatGroq
import faiss
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import warnings
warnings.filterwarnings("ignore")
# Initialize embeddings and LLM models once
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
# local_model = "llama3.2:3b"
# llm = ChatOllama(model=local_model)

# Define the Redis configuration
# redis_config = RedisConfig(
#     index_name="db0",
#     redis_url='redis://localhost:6379',
#     metadata_schema=[
#         {"name": "embedding", "type": "embedding"},
#         {"name": "source", "type": "string"}
#     ]
# )

# Define the query prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question:
    <question>
    {question}
    </question>""",
)

def retrieval() : 
    # Check if vector_db_new is already in session state
    
    vector_db_new = RedisVectorStore.from_existing_index(index_name='new', embedding=embeddings)

    # Check if retriever is already initialized
    # retriever = MultiQueryRetriever.from_llm(
    ret=vector_db_new.as_retriever(search_type="similarity",kwargs={'k':50,'fetch_k':343,'lambda_build':0.5})
            # llm,
            # prompt=QUERY_PROMPT
        # ) 
    # print(retriever)
    # print(type(retriever))
    retriever_tool=create_retriever_tool(ret,"ret tool","your task is to retrieve relevant documents for the query")

    

    return ret,retriever_tool

# Example usage
# retriever = retrieval()
# ans = retriever.invoke("ceo of miniOrange")
# with st.expander("Document similarity Search"):
#         for doc in (ans):
#             st.write(doc.page_content)
#             st.write('------------------------')
                                                 
