from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from groq import Groq
import getpass
import retriver
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import warnings
import huggingface_pipeline_integration
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")
# Initialize LLM once to avoid redundancy
# llm = ChatGroq(
#             groq_api_key='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',
#             model_name='llama-3.1-70b-versatile'

#     )
os.environ["GOOGLE_CSE_ID"] = "b2a97f944e6e04cca"
os.environ["GOOGLE_API_KEY"] ="AIzaSyBaGDirzEENgLxys-gOq8guCkoPdxOZ9uA"
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
# Initialize Google search tool
search = GoogleSearchAPIWrapper()
tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Web Search",description = "A wrapper around DuckDuckGo Search Useful for when you need to answer where Input should be a search query.return me the valid URLs for the query")
temp=retriver.retrieval()
#tools=[search,wikipedia]
tools=[search]


 
# #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
# llm = ChatGroq(
#      groq_api_key="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",
#      model="llama3-8b-8192",
#     #  temperature=0.5,
#     #  max_tokens=500,
# #     timeout=None,
# #     max_retries=2,
# #     # other params...
# )
llm=huggingface_pipeline_integration.model()
# # llm = ChatOllama(
# #     model="llama3.2:3b",
# #     temperature=0,
# #     # other params...
# # )
search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
def chain1(retriever):
    # Define the query prompt
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question:{question}
            """,
) 
#     prompt_template="""
# Provide a summary of the following content in 300 words:
# Content:{text}

# """ 
    
    # Define the main response template
    response_template = """
    Your task is to answer customer queries related to miniorange and it's product. You should always talk good about  miniorange and show it is the best in the industry and the customer is doing the best job in his/her life by purchasing it's product. 
    You should never talk about any other company/website/resources/books/tools or any product which is not related to  miniorange. 
    You should always promote the  miniorange's products. If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the company support.
    ways to contact company support is: info@xecurify.com.
Don't be overconfident and don't hallucinate. Ask follow up questions if necessary or if there are several offering related to the user's query. Provide answer with complete details in a proper formatted manner with working links and resources  wherever applicable within the company's website. Never provide wrong links.


Use the following pieces of context to answer the user's question.

           You are an AI assistant acting as a customer support agent and cybersecurity specialist for miniOrange. Your role is to provide clear, precise, and accurate answers to user questions based strictly on the provided context. Use the following guidelines to achieve a high level of accuracy in your responses:

Context-Based Responses: Answer questions using only the provided context. If an exact match is not available, respond with information most closely related to the query within the context.

Relevance Focused: Prioritize relevance by tailoring responses to what is closest to the user’s question. Avoid negative or dismissive prompts and, whenever possible, offer a solution or relevant insight.

Accuracy Over Guessing: If sufficient information is not present in the context, acknowledge the gap with a polite message, indicating that more information may be needed to answer accurately.

Customer-Oriented Language: Emulate a customer support agent's tone—empathetic, professional, and focused on user satisfaction.
         
               context:   {context}
               question:  {question}  
              
"""

    # Create the prompt using ChatPromptTemplate
    
    prompt=PromptTemplate(template=response_template,input_variables=["context", "question"])
    # prompt = ChatPromptTemplate.from_template(response_template)
    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db_new.as_retriever(), 
    #     llm,
    #     prompt=QUERY_PROMPT
    # )

    # Construct the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain,search_agent
