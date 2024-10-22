from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from duckduckgo_search import ddg
from langchain.prompts import PromptTemplate
search_prompt_template = PromptTemplate.from_template(
    "Find relevant url link about {topic}. Provide top 3 links from miniorange website matching with the query."
)
llm = ChatGroq(
     groq_api_key="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",
#     model="llama3-8b-8192",
#     temperature=0,
#     max_tokens=100,
#     timeout=None,
#     max_retries=2,
#     # other params...
)
# def search_duckduckgo(query, max_results=5):
#     results = ddg(query, max_results=max_results)
#     for i, result in enumerate(results):
#         print(f"{i+1}: {result['title']} - {result['href']}")

# def duckduckgo_search_chain(query):
#     prompt = search_prompt_template.format(topic=query)
#     # Here you would typically call your language model, for example, OpenAI's GPT
#     # For demonstration, we will print the prompt
#     print(prompt)
    # After this, you could use the prompt in an LLM chain
def duckduckgo_search_and_summarize(query):
    results = ddg(query, max_results=5)
    summarized_results = ""
    for result in results:
        summarized_results += f"{result['title']}: {result['href']}\n"

    # Use the summarized results in a prompt for summarization
    summary_prompt = search_prompt_template.format(topic=summarized_results)
    
    # Here you would call your language model
    model_output = llm(summary_prompt)  # assuming openai_complete is your function to interact with the model.
    return model_output

x=duckduckgo_search_and_summarize("iam vs pam")