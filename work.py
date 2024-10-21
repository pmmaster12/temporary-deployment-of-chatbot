from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import WebBaseLoader
import psycopg2
import pandas as pd
from langchain_groq import ChatGroq
import streamlit as st
from PIL import Image
import json
import sounddevice as sd
import numpy as np
import speech_recognition as sr

#path for pdf file 
local_path = "miniorange.pdf"

# Local PDF file uploads
@st.cache_data
def load_data():
    #load pdf data
    # loader = UnstructuredPDFLoader(file_path=local_path)
    #load urldata
    loader = WebBaseLoader("https://www.miniorange.com/about_us")

    return loader.load()
data=load_data()
# else:
#   print("Upload a PDF file")
# with open('about.txt', 'w', encoding='utf-8') as file:
#     file.write(str(data)) 
# data[0].page_content

# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

vector_db_new=Chroma(persist_directory="iam_pam",embedding_function=OllamaEmbeddings(model="nomic-embed-text"),collection_name="local-rag" )
# make a new embedding and store on ia_pam
# vector_db_new = Chroma.from_documents(
#     documents=chunks, 
#     embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
#     collection_name="local-rag",
#     persist_directory="iam_pam"
# )

vector_db_new.persist()

# LLM from Ollama
local_model = "llama3"
# llm = ChatOllama(model=local_model)
llm = ChatGroq(

            groq_api_key='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',

            model_name='llama3-8b-8192'
    )

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# read all vector stores from directory 
retriever = MultiQueryRetriever.from_llm(
    vector_db_new.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ============================== UI Code ================================ 
# =======================================================================

# import streamlit as st
# from PIL import Image
# import json
# import http.cookies
# import numpy as np
# import sounddevice as sd
# import speech_recognition as sr

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Sampling rate for voice input
# fs = 44100

# def record_audio(duration=5):
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     return np.array(audio, dtype=np.int16)

# def process_input(inp):
#     if inp and not st.session_state.input_disabled:
#         st.session_state.input_disabled = True
#         st.session_state.user_input = inp

#         # Dummy chain.invoke function for processing
#         op = chain.invoke(inp)

#         # If no chat is selected, start a new chat
#         if st.session_state.selected_chat is None:
#             st.session_state.first_question = inp
#             st.session_state.chat_sessions.append((inp, [("You", inp), ("Assistant", op)]))
#             st.session_state.selected_chat = inp
#         else:
#             # Find the current chat session
#             for idx, (fq, messages) in enumerate(st.session_state.chat_sessions):
#                 if fq == st.session_state.selected_chat:
#                     st.session_state.chat_sessions[idx] = (fq, messages + [("You", inp), ("Assistant", op)])
#                     break

#         # Save chat history to cookies after every input
#         save_chat_history_to_cookies()

#         # Reset input
#         st.session_state.input_disabled = False
#         st.session_state.user_input = ""

#         # Display the current messages on the main screen
#         display_messages(st.session_state.chat_sessions)

# def new_chat():
#     st.session_state.messages = []
#     st.session_state.first_question = None
#     st.session_state.selected_chat = None

# def display_messages(chat_sessions):
#     if st.session_state.selected_chat:
#         # Find the messages for the selected chat
#         selected_messages = next((messages for fq, messages in chat_sessions if fq == st.session_state.selected_chat), [])

#         # Calculate the number of recent exchanges (each exchange is a pair of messages)
#         num_exchanges = 5
#         recent_exchanges = []
#         for i in range(len(selected_messages) - 1, 0, -2):
#             if len(recent_exchanges) >= num_exchanges * 2:
#                 break
#             recent_exchanges.insert(0, selected_messages[i-1])
#             recent_exchanges.insert(1, selected_messages[i])

#         st.write("### Chat History")
#         for sender, message in recent_exchanges:
#             st.write(f"**{sender}:** {message}")
#             st.write("===========================================================================")
#     else:
#         st.write("No chat selected.")

# def load_chat_history_from_cookies():
#     params = st.query_params
#     cookie_str = params.get('cookie', '')
#     cookie = http.cookies.SimpleCookie(cookie_str)
#     chat_sessions = []
#     selected_chat = None

#     if "chat_sessions" in cookie:
#         try:
#             chat_sessions = json.loads(cookie["chat_sessions"].value)
#         except json.JSONDecodeError:
#             chat_sessions = []

#     if "selected_chat" in cookie:
#         selected_chat = cookie["selected_chat"].value

#     return chat_sessions, selected_chat

# def save_chat_history_to_cookies():
#     chat_data = {"chat_sessions": st.session_state.chat_sessions}
    
#     cookie = http.cookies.SimpleCookie()
#     cookie["chat_sessions"] = json.dumps(chat_data["chat_sessions"])
#     cookie["chat_sessions"]["expires"] = 60 * 3   # Cookie valid for 3 minutes

#     # Save selected chat
#     if st.session_state.selected_chat:
#         cookie["selected_chat"] = st.session_state.selected_chat
#         cookie["selected_chat"]["expires"] = 60 * 3  # Cookie valid for 3 minutes
    
#     cookie_str = cookie.output(header='', sep='')
#     st.query_params.cookie = cookie_str

# # Initialize session state variables
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'input_disabled' not in st.session_state:
#     st.session_state.input_disabled = False
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""
# if 'first_question' not in st.session_state:
#     st.session_state.first_question = None
# if 'chat_sessions' not in st.session_state:
#     st.session_state.chat_sessions, st.session_state.selected_chat = load_chat_history_from_cookies()

# if 'selected_chat' not in st.session_state:
#     st.session_state.selected_chat = None

# # Load and display logo
# img = Image.open("C:/Sakshi Miniorange Work/miniorange-logo.png")
# st.image(img, width=300, use_column_width=False)
# hide_full_screen = """
#     <style>
#     button[title="View fullscreen"] {
#         display: none;
#     }
#     </style>
#     """
# st.markdown(hide_full_screen, unsafe_allow_html=True)

# st.markdown("""
#     <style>
#         .sidebar-header {
#             font-size: 29px;
#             font-weight: bold;
#             height: 66px;
#             margin: 0;
#             padding: 0;
#             display: flex;
#             align-items: center;
#         }
#         .sidebar-button button {
#             font-size: 24px;
#             margin: 0;
#             padding: 0;
#             height: 100%; 
#             display: flex;
#             align-items: center;
#             justify-content: center;
#         }
#         .prompt-bar {
#             position: sticky;
#             top: 0;
#             z-index: 1;
#             background-color: #ffffff;
#             padding-top: 20px;
#             padding-bottom: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar logic
# with st.sidebar:
#     col1, col2 = st.columns([4, 1])
#     with col1:
#         st.markdown('<p class="sidebar-header">Chat History</p>', unsafe_allow_html=True)
#         unique_chats = list(dict.fromkeys(fq for fq, _ in st.session_state.chat_sessions))
        
#         # Show all threads in the same session
#         for i, first_question in enumerate(unique_chats):
#             if st.button(first_question, key=f"chat_{i}"):
#                 st.session_state.selected_chat = first_question
                
#     with col2:
#         st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
#         if st.button(":heavy_plus_sign:", key="new_chat", help="Start a new chat"):
#             new_chat()
#         st.markdown('</div>', unsafe_allow_html=True)

# # Prompt bar
# st.markdown('<div class="prompt-bar">', unsafe_allow_html=True)
# main_col, mic_col = st.columns([7, 1])

# with main_col:
#     if not st.session_state.input_disabled:
#         inp = st.chat_input("How can I help you .........?")
#         if inp:
#             process_input(inp)
#     else:
#         inp = st.chat_input("How can I help you .........?", disabled=True)

# with mic_col:
#     if st.button(":material/mic:", key="voiceLogo"):
#         audio_data = record_audio()
#         audio = sr.AudioData(audio_data.tobytes(), fs, 2)
        
#         try:
#             text = recognizer.recognize_google(audio, language='en-US')
#             with main_col:
#                 process_input(text)
#         except sr.UnknownValueError:
#             st.write("Sorry, I did not understand that.")
#         except sr.RequestError:
#             st.write("Sorry, my speech service is down.")
# st.markdown('</div>', unsafe_allow_html=True)

# # Display chat messages
# display_messages(st.session_state.chat_sessions)


#=================================== logo fixed at top code ===================================
# import streamlit as st
# from PIL import Image
# import json
# import http.cookies
# import numpy as np
# import sounddevice as sd
# import speech_recognition as sr

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Sampling rate for voice input
# fs = 44100

# def record_audio(duration=5):
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     return np.array(audio, dtype=np.int16)

# def process_input(inp):
#     if inp and not st.session_state.input_disabled:
#         st.session_state.input_disabled = True
#         st.session_state.user_input = inp

#         # Dummy chain.invoke function for processing
#         op = chain.invoke(inp)

#         # If no chat is selected, start a new chat
#         if st.session_state.selected_chat is None:
#             st.session_state.first_question = inp
#             st.session_state.chat_sessions.append((inp, [("You", inp), ("Assistant", op)]))
#             st.session_state.selected_chat = inp
#         else:
#             # Find the current chat session
#             for idx, (fq, messages) in enumerate(st.session_state.chat_sessions):
#                 if fq == st.session_state.selected_chat:
#                     st.session_state.chat_sessions[idx] = (fq, messages + [("You", inp), ("Assistant", op)])
#                     break

#         # Save chat history to cookies after every input
#         save_chat_history_to_cookies()

#         # Reset input
#         st.session_state.input_disabled = False
#         st.session_state.user_input = ""

#         # Display the current messages on the main screen
#         display_messages(st.session_state.chat_sessions)

# def new_chat():
#     st.session_state.messages = []
#     st.session_state.first_question = None
#     st.session_state.selected_chat = None

# def display_messages(chat_sessions):
#     if st.session_state.selected_chat:
#         # Find the messages for the selected chat
#         selected_messages = next((messages for fq, messages in chat_sessions if fq == st.session_state.selected_chat), [])

#         # Calculate the number of recent exchanges (each exchange is a pair of messages)
#         num_exchanges = 5
#         recent_exchanges = []
#         for i in range(len(selected_messages) - 1, 0, -2):
#             if len(recent_exchanges) >= num_exchanges * 2:
#                 break
#             recent_exchanges.insert(0, selected_messages[i-1])
#             recent_exchanges.insert(1, selected_messages[i])

#         st.write("### Chat History")
#         for sender, message in recent_exchanges:
#             st.write(f"**{sender}:** {message}")
#             st.write("===========================================================================")
#     else:
#         st.write("No chat selected.")

# def load_chat_history_from_cookies():
#     params = st.query_params
#     cookie_str = params.get('cookie', '')
#     cookie = http.cookies.SimpleCookie(cookie_str)
#     chat_sessions = []
#     selected_chat = None

#     if "chat_sessions" in cookie:
#         try:
#             chat_sessions = json.loads(cookie["chat_sessions"].value)
#         except json.JSONDecodeError:
#             chat_sessions = []

#     if "selected_chat" in cookie:
#         selected_chat = cookie["selected_chat"].value

#     return chat_sessions, selected_chat

# def save_chat_history_to_cookies():
#     chat_data = {"chat_sessions": st.session_state.chat_sessions}
    
#     cookie = http.cookies.SimpleCookie()
#     cookie["chat_sessions"] = json.dumps(chat_data["chat_sessions"])
#     cookie["chat_sessions"]["expires"] = 60 * 3   # Cookie valid for 3 minutes

#     # Save selected chat
#     if st.session_state.selected_chat:
#         cookie["selected_chat"] = st.session_state.selected_chat
#         cookie["selected_chat"]["expires"] = 60 * 3  # Cookie valid for 3 minutes
    
#     cookie_str = cookie.output(header='', sep='')
#     st.query_params.cookie = cookie_str

# # Initialize session state variables
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'input_disabled' not in st.session_state:
#     st.session_state.input_disabled = False
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""
# if 'first_question' not in st.session_state:
#     st.session_state.first_question = None
# if 'chat_sessions' not in st.session_state:
#     st.session_state.chat_sessions, st.session_state.selected_chat = load_chat_history_from_cookies()

# if 'selected_chat' not in st.session_state:
#     st.session_state.selected_chat = None

# import base64
# img = "C:/Sakshi Miniorange Work/miniorange-logo.png"
# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
#     return encoded_string
# base64_image = image_to_base64(img)

# # CSS to fix the image at the top
# st.markdown("""
#     <style>
#         .header {
#             position: fixed;
#             top: 50px;
#             left: 0;
#             width: 100%;
#             background-color: #ffffff;
#             padding: 10px;
#             border-bottom: 2px solid #ddd;
#             z-index: 1000;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }
#         .header img {
#             max-width: 300px;
#             height: auto;
#         }
#         .content {
#             margin-top: 100px; 
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display the logo at the top
# st.markdown(f'''
#     <div class="header">
#         <img src="data:image/png;base64,{base64_image}" alt="Logo">
#     </div>
#     <div class="content">
# ''', unsafe_allow_html=True)

# st.markdown("""
#     <style>
#         .sidebar-header {
#             font-size: 29px;
#             font-weight: bold;
#             height: 66px;
#             margin: 0;
#             padding: 0;
#             display: flex;
#             align-items: center;
#         }
#         .sidebar-button button {
#             font-size: 24px;
#             margin: 0;
#             padding: 0;
#             height: 100%; 
#             display: flex;
#             align-items: center;
#             justify-content: center;
#         }
#         .prompt-bar {
#             position: sticky;
#             top: 0;
#             z-index: 1;
#             background-color: #ffffff;
#             padding-top: 20px;
#             padding-bottom: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar logic
# with st.sidebar:
#     col1, col2 = st.columns([4, 1])
#     with col1:
#         st.markdown('<p class="sidebar-header">Chat History</p>', unsafe_allow_html=True)
#         unique_chats = list(dict.fromkeys(fq for fq, _ in st.session_state.chat_sessions))
        
#         # Show all threads in the same session
#         for i, first_question in enumerate(unique_chats):
#             if st.button(first_question, key=f"chat_{i}"):
#                 st.session_state.selected_chat = first_question
                
#     with col2:
#         st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
#         if st.button(":heavy_plus_sign:", key="new_chat", help="Start a new chat"):
#             new_chat()
#         st.markdown('</div>', unsafe_allow_html=True)


# # Prompt bar
# st.markdown('<div class="prompt-bar">', unsafe_allow_html=True)
# main_col, mic_col = st.columns([7, 1])

# with main_col:
#     if not st.session_state.input_disabled:
#         inp = st.chat_input("How can I help you .........?")
#         if inp:
#             process_input(inp)
#     else:
#         inp = st.chat_input("How can I help you .........?", disabled=True)

# with mic_col:
#     if st.button(":material/mic:", key="voiceLogo"):
#         audio_data = record_audio()
#         audio = sr.AudioData(audio_data.tobytes(), fs, 2)
        
#         try:
#             text = recognizer.recognize_google(audio, language='en-US')
#             with main_col:
#                 process_input(text)
#         except sr.UnknownValueError:
#             st.write("Sorry, I did not understand that.")
#         except sr.RequestError:
#             st.write("Sorry, my speech service is down.")
# st.markdown('</div>', unsafe_allow_html=True)

# # Display chat messages
# display_messages(st.session_state.chat_sessions)


# Code working fine for both fixed image logo and mice image showing when recording.

# import streamlit as st
# from PIL import Image
# import json
# import http.cookies
# import numpy as np
# import sounddevice as sd
# import speech_recognition as sr
# import time

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Sampling rate for voice input
# fs = 44100

# def record_audio(duration=5):
#     image_path = "C:/Sakshi Miniorange Work/Voice-Recoder-icon.png"
    
#     def image_to_base64(image_path):
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode("utf-8")
    
#     base64_image = image_to_base64(image_path)
    
#     placeholder = st.empty()
#     with placeholder.container():
#         st.markdown(f"""
#         <style>
#             .centered {{
#                 position: fixed;
#                 top: 50%;
#                 left: 50%;
#                 transform: translate(-50%, -50%);
#                 z-index: 9999;
#             }}
#             .centered img {{
#                 max-width: 150px;
#                 height: auto;
#             }}
#             .recording-text {{
#                 font-size: 24px;
#                 font-weight: bold;
#                 margin-top: 10px;
#                 color: red;
#             }}
#         </style>
#         <div class="centered">
#             <img src="data:image/png;base64,{base64_image}" alt="Recording...">
#             <div class="recording-text">Recording ...</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()

#     placeholder.empty()

#     return np.array(audio, dtype=np.int16)


# def process_input(inp):
#     if inp and not st.session_state.input_disabled:
#         st.session_state.input_disabled = True
#         st.session_state.user_input = inp

#         # Dummy chain.invoke function for processing
#         op = chain.invoke(inp)

#         # If no chat is selected, start a new chat
#         if st.session_state.selected_chat is None:
#             st.session_state.first_question = inp
#             st.session_state.chat_sessions.append((inp, [("You", inp), ("Assistant", op)]))
#             st.session_state.selected_chat = inp
#         else:
#             # Find the current chat session
#             for idx, (fq, messages) in enumerate(st.session_state.chat_sessions):
#                 if fq == st.session_state.selected_chat:
#                     st.session_state.chat_sessions[idx] = (fq, messages + [("You", inp), ("Assistant", op)])
#                     break

#         # Save chat history to cookies after every input
#         save_chat_history_to_cookies()

#         # Reset input
#         st.session_state.input_disabled = False
#         st.session_state.user_input = ""

#         # Display the current messages on the main screen
#         display_messages(st.session_state.chat_sessions)

# def new_chat():
#     st.session_state.messages = []
#     st.session_state.first_question = None
#     st.session_state.selected_chat = None

# def display_messages(chat_sessions):
#     if st.session_state.selected_chat:
#         # Find the messages for the selected chat
#         selected_messages = next((messages for fq, messages in chat_sessions if fq == st.session_state.selected_chat), [])

#         # Calculate the number of recent exchanges (each exchange is a pair of messages)
#         num_exchanges = 5
#         recent_exchanges = []
#         for i in range(len(selected_messages) - 1, 0, -2):
#             if len(recent_exchanges) >= num_exchanges * 2:
#                 break
#             recent_exchanges.insert(0, selected_messages[i-1])
#             recent_exchanges.insert(1, selected_messages[i])

#         st.write("### Chat History")
#         for sender, message in recent_exchanges:
#             st.write(f"**{sender}:** {message}")
#             st.write("===========================================================================")
#     else:
#         st.write("No chat selected.")

# def load_chat_history_from_cookies():
#     params = st.query_params
#     cookie_str = params.get('cookie', '')
#     cookie = http.cookies.SimpleCookie(cookie_str)
#     chat_sessions = []
#     selected_chat = None

#     if "chat_sessions" in cookie:
#         try:
#             chat_sessions = json.loads(cookie["chat_sessions"].value)
#         except json.JSONDecodeError:
#             chat_sessions = []

#     if "selected_chat" in cookie:
#         selected_chat = cookie["selected_chat"].value

#     return chat_sessions, selected_chat

# def save_chat_history_to_cookies():
#     chat_data = {"chat_sessions": st.session_state.chat_sessions}
    
#     cookie = http.cookies.SimpleCookie()
#     cookie["chat_sessions"] = json.dumps(chat_data["chat_sessions"])
#     cookie["chat_sessions"]["expires"] = 60 * 3   # Cookie valid for 3 minutes

#     # Save selected chat
#     if st.session_state.selected_chat:
#         cookie["selected_chat"] = st.session_state.selected_chat
#         cookie["selected_chat"]["expires"] = 60 * 3  # Cookie valid for 3 minutes
    
#     cookie_str = cookie.output(header='', sep='')
#     st.query_params.cookie = cookie_str

# # Initialize session state variables
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'input_disabled' not in st.session_state:
#     st.session_state.input_disabled = False
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""
# if 'first_question' not in st.session_state:
#     st.session_state.first_question = None
# if 'chat_sessions' not in st.session_state:
#     st.session_state.chat_sessions, st.session_state.selected_chat = load_chat_history_from_cookies()

# if 'selected_chat' not in st.session_state:
#     st.session_state.selected_chat = None

# import base64
# img = "C:/Sakshi Miniorange Work/miniorange-logo.png"
# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
#     return encoded_string
# base64_image = image_to_base64(img)

# # CSS to fix the image at the top
# st.markdown("""
#     <style>
#         .header {
#             position: fixed;
#             top: 50px;
#             left: 0;
#             width: 100%;
#             background-color: #ffffff;
#             padding: 10px;
#             border-bottom: 2px solid #ddd;
#             z-index: 1000;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }
#         .header img {
#             max-width: 300px;
#             height: auto;
#         }
#         .content {
#             margin-top: 100px; 
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display the logo at the top
# st.markdown(f'''
#     <div class="header">
#         <img src="data:image/png;base64,{base64_image}" alt="Logo">
#     </div>
#     <div class="content">
# ''', unsafe_allow_html=True)

# st.markdown("""
#     <style>
#         .sidebar-header {
#             font-size: 29px;
#             font-weight: bold;
#             height: 66px;
#             margin: 0;
#             padding: 0;
#             display: flex;
#             align-items: center;
#         }
#         .sidebar-button button {
#             font-size: 24px;
#             margin: 0;
#             padding: 0;
#             height: 100%; 
#             display: flex;
#             align-items: center;
#             justify-content: center;
#         }
#         .prompt-bar {
#             position: sticky;
#             top: 0;
#             z-index: 1;
#             background-color: #ffffff;
#             padding-top: 20px;
#             padding-bottom: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar logic
# with st.sidebar:
#     col1, col2 = st.columns([4, 1])
#     with col1:
#         st.markdown('<p class="sidebar-header">Chat History</p>', unsafe_allow_html=True)
#         unique_chats = list(dict.fromkeys(fq for fq, _ in st.session_state.chat_sessions))
        
#         # Show all threads in the same session
#         for i, first_question in enumerate(unique_chats):
#             if st.button(first_question, key=f"chat_{i}"):
#                 st.session_state.selected_chat = first_question
                
#     with col2:
#         st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
#         if st.button(":heavy_plus_sign:", key="new_chat", help="Start a new chat"):
#             new_chat()
#         st.markdown('</div>', unsafe_allow_html=True)


# # Prompt bar
# st.markdown('<div class="prompt-bar">', unsafe_allow_html=True)
# main_col, mic_col = st.columns([7, 1])

# with main_col:
#     if not st.session_state.input_disabled:
#         inp = st.chat_input("How can I help you .........?")
#         if inp:
#             process_input(inp)
#     else:
#         inp = st.chat_input("How can I help you .........?", disabled=True)

# with mic_col:
#     if st.button(":material/mic:", key="voiceLogo"):
#         audio_data = record_audio()
#         audio = sr.AudioData(audio_data.tobytes(), fs, 2)
        
#         try:
#             text = recognizer.recognize_google(audio, language='en-US')
#             with main_col:
#                 process_input(text)
#         except sr.UnknownValueError:
#             st.write("Sorry, I could not understand your voice input.")
#         except sr.RequestError:
#             st.write("Error with the recognition service. Please try again.")

# st.markdown('</div>', unsafe_allow_html=True)

# display_messages(st.session_state.chat_sessions)




import streamlit as st
from PIL import Image
import json
import http.cookies
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import streamlit.components.v1 as components

# Initialize the recognizer
recognizer = sr.Recognizer()

# Sampling rate for voice input
fs = 44100

def record_audio(duration=5):
    image_path = "C:/Sakshi Miniorange Work/Voice-Recoder-icon.png"
    
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    base64_image = image_to_base64(image_path)
    
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
        <style>
            .centered {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 9999;
            }}
            .centered img {{
                max-width: 150px;
                height: auto;
            }}
            .recording-text {{
                font-size: 24px;
                font-weight: bold;
                margin-top: 10px;
                color: red;
            }}
        </style>
        <div class="centered">
            <img src="data:image/png;base64,{base64_image}" alt="Recording...">
            <div class="recording-text">Recording ...</div>
        </div>
        """, unsafe_allow_html=True)
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    placeholder.empty()

    return np.array(audio, dtype=np.int16)


def process_input(inp):
    if inp and not st.session_state.input_disabled:
        st.session_state.input_disabled = True
        st.session_state.user_input = inp

        # Dummy chain.invoke function for processing
        op = chain.invoke(inp)

        # If no chat is selected, start a new chat
        if st.session_state.selected_chat is None:
            st.session_state.first_question = inp
            st.session_state.chat_sessions.append((inp, [("You", inp), ("Assistant", op)]))
            st.session_state.selected_chat = inp
        else:
            # Find the current chat session
            for idx, (fq, messages) in enumerate(st.session_state.chat_sessions):
                if fq == st.session_state.selected_chat:
                    st.session_state.chat_sessions[idx] = (fq, messages + [("You", inp), ("Assistant", op)])
                    break

        # Save chat history to cookies after every input
        save_chat_history_to_cookies()

        # Reset input
        st.session_state.input_disabled = False
        st.session_state.user_input = ""

def new_chat():
    st.session_state.messages = []
    st.session_state.first_question = None
    st.session_state.selected_chat = None

def display_messages(chat_sessions):
    if st.session_state.selected_chat:
        # Find the messages for the selected chat
        selected_messages = next((messages for fq, messages in chat_sessions if fq == st.session_state.selected_chat), [])

        # Calculate the number of recent exchanges (each exchange is a pair of messages)
        num_exchanges = 5
        recent_exchanges = []
        for i in range(len(selected_messages) - 1, 0, -2):
            if len(recent_exchanges) >= num_exchanges * 2:
                break
            recent_exchanges.insert(0, selected_messages[i-1])
            recent_exchanges.insert(1, selected_messages[i])

        st.write("### Chat History")
        for sender, message in recent_exchanges:
            st.write(f"**{sender}:** {message}")
            st.write("===========================================================================")
    else:
        st.write("No chat selected.")

def load_chat_history_from_cookies():
    params = st.query_params
    cookie_str = params.get('cookie', '')
    cookie = http.cookies.SimpleCookie(cookie_str)
    chat_sessions = []
    selected_chat = None

    if "chat_sessions" in cookie:
        try:
            chat_sessions = json.loads(cookie["chat_sessions"].value)
        except json.JSONDecodeError:
            chat_sessions = []

    if "selected_chat" in cookie:
        selected_chat = cookie["selected_chat"].value

    return chat_sessions, selected_chat

def save_chat_history_to_cookies():
    chat_data = {"chat_sessions": st.session_state.chat_sessions}
    
    cookie = http.cookies.SimpleCookie()
    cookie["chat_sessions"] = json.dumps(chat_data["chat_sessions"])
    cookie["chat_sessions"]["expires"] = 60 * 3   # Cookie valid for 3 minutes

    # Save selected chat
    if st.session_state.selected_chat:
        cookie["selected_chat"] = st.session_state.selected_chat
        cookie["selected_chat"]["expires"] = 60 * 3  # Cookie valid for 3 minutes
    
    cookie_str = cookie.output(header='', sep='')
    st.query_params.cookie = cookie_str

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'input_disabled' not in st.session_state:
    st.session_state.input_disabled = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'first_question' not in st.session_state:
    st.session_state.first_question = None
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions, st.session_state.selected_chat = load_chat_history_from_cookies()

if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = None

import base64
img = "C:/Sakshi Miniorange Work/miniorange-logo.png"
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string
base64_image = image_to_base64(img)

# CSS to fix the image at the top
st.markdown("""
    <style>
        .header {
            position: fixed;
            top: 50px;
            left: 0;
            width: 100%;
            background-color: #ffffff;
            padding: 10px;
            border-bottom: 2px solid #ddd;
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .header img {
            max-width: 300px;
            height: auto;
        }
        .content {
            margin-top: 100px; 
        }
    </style>
""", unsafe_allow_html=True)

# Display the logo at the top
st.markdown(f'''
    <div class="header">
        <img src="data:image/png;base64,{base64_image}" alt="Logo">
    </div>
    <div class="content">
''', unsafe_allow_html=True)

st.markdown("""
    <style>
        .sidebar-header {
            font-size: 29px;
            font-weight: bold;
            height: 66px;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
        }
        .sidebar-button button {
            font-size: 24px;
            margin: 0;
            padding: 0;
            height: 100%; 
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .prompt-bar {
            position: sticky;
            top: 0;
            z-index: 1;
            background-color: #ffffff;
        }
            
    </style>
""", unsafe_allow_html=True)

# Sidebar logic
with st.sidebar:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<p class="sidebar-header">Chat History</p>', unsafe_allow_html=True)
        unique_chats = list(dict.fromkeys(fq for fq, _ in st.session_state.chat_sessions))
        
        # Show all threads in the same session
        for i, first_question in enumerate(unique_chats):
            if st.button(first_question, key=f"chat_{i}"):
                st.session_state.selected_chat = first_question
                
    with col2:
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        if st.button(":heavy_plus_sign:", key="new_chat", help="Start a new chat"):
            new_chat()
        st.markdown('</div>', unsafe_allow_html=True)

def show_dialog(message):
    # JavaScript to show an alert dialog
    dialog_html = f"""
    <html>
    <body>
    <script>
    alert("{message}");
    </script>
    </body>
    </html>
    """
    components.html(dialog_html, height=0)

# Prompt bar
st.markdown('<div class="prompt-bar">', unsafe_allow_html=True)
main_col, mic_col = st.columns([7, 1])

with main_col:
    if not st.session_state.input_disabled:
        inp = st.chat_input("How can I help you .........?")
        if inp:
            process_input(inp)
    else:
        inp = st.chat_input("How can I help you .........?", disabled=True)

with mic_col:
    if st.button(":material/mic:", key="voiceLogo"):
        audio_data = record_audio()
        audio = sr.AudioData(audio_data.tobytes(), fs, 2)
        
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            with main_col:
                process_input(text)
        except sr.UnknownValueError:
            show_dialog("Sorry, I could not understand your voice input.")
        except sr.RequestError:
            show_dialog("Error with the recognition service. Please try again.")

st.markdown('</div>', unsafe_allow_html=True)

display_messages(st.session_state.chat_sessions)