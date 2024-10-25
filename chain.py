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

search=DuckDuckGoSearchRun(name="Search",description = "A wrapper around DuckDuckGo Search Useful for when you need to answer where Input should be a search query.return me the valid URLs for the query")
temp=retriver.retrieval()
#tools=[search,wikipedia]
tools=[search]


 
# #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
# llm = ChatGroq(
#      groq_api_key="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",
# #     model="llama3-8b-8192",
# #     temperature=0,
# #     max_tokens=100,
# #     timeout=None,
# #     max_retries=2,
# #     # other params...
# )
llm=huggingface_pipeline_integration.model()
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
           -You are an AI assistant acting as a customer support agent and cybersecurity specialist for7 miniOrange.
           - Your task is to provide a precise,detailed and accurate answer to the user’s question based *only* on the following context. 
            -Your should answer things which is most similar to context.
            -If you don't find exact relevance to query in context then answer which is most likey to be related from the context with reference to user question.
            -Try to answer something regarding query you should avoid negative prompt regarding query as long as it is avoidable. 
            -Consider last option for response that don't have sufficient information.
            -If the context does not provide sufficient information, then just say don't have sufficient information.
            -Your answer should reflect how an actual customer agent would talk, ensuring user satisfaction.
         
               context:   {context}
               question:  {question}  
              
"""

    # Create the prompt using ChatPromptTemplate
    
    prompt=PromptTemplate(template=response_template,input_variables=["question","context"])
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
