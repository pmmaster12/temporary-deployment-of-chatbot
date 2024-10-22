import retriver
import streamlit as st
import chain
import time
import speech_recognition as sr
import pyttsx3
import csv
import sys
import warnings
import huggingface_pipeline_integration
import streamlit as st
from PIL import Image
import json
import numpy as np
import streamlit as st
from PIL import Image
import json
import http.cookies
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import streamlit.components.v1 as components
import speech_recognition as sr
import sentiment_analysis
import time
import pygsheets 
import google_sheet_connector
# import TTS
# Streamlit App
# st.title("miniOrange Support : 24 X 7 Guide")

# Initialize retriever and chain only once
warnings.filterwarnings("ignore")
retrieval = retriver.retrieval()

chain = chain.chain1(retrieval[0])

# Importing required library 

  
# Create the Client 
# client = pygsheets.authorize(service_account_file="chatbot-responses-3fbcd29275d3.json") 
  
# # opens a spreadsheet by its name/title 
# spreadsht = client.open("chatbot responses")
# spreadsht.cell("A1").set_text_format("bold", True).value = "Query"
# spreadsht.cell("A2").set_text_format("bold", True).value = "Response"
# spreadsht.cell("A3").set_text_format("bold", True).value = "Time Taken"

# Input mode selection
# input_mode = st.radio("Choose input mode:", ("Voice", "Text"))


# if input_mode == "Voice":
#     if st.button("Submit"):
#         recognizer = sr.Recognizer()
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             recognizer.adjust_for_ambient_noise(source, duration=0.3)
#             audio = recognizer.listen(source)
#             try:
#                 query = recognizer.recognize_google(audio).lower()
#                 st.write(f"You said: {query}")
#                 engine = pyttsx3.init()
#                 engine.say(query)
#                 engine.runAndWait()
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")
#             except sr.UnknownValueError:
#                 st.write("Could not understand audio")
# else:
# while (True):
 

# If the query is not empty, process it
 
    # temp=sentiment_analysis.sentiment(result)
    # print(temp)
    # if('does not' in result or 'Unfortunately'  in result):
        
    #   result1=chain[1].invoke(query)
    #   print("response:",result1)
    #   response_time = time.time() - t
    #   print(f"**Time taken:** {response_time:.2f} seconds")

    # else:
   #  print("**Response:**", result)
   #  response_time = time.time() - t
   #  print(f"**Time taken:** {response_time:.2f} seconds")
   #  try:
   #    with open('predicted_text.csv','a') as file:
      
   #          writer=csv.writer(file)
   #          writer.writerow([query,result,response_time])
   #  except Exception as e:
   #     print(f"exception occure due to {e}")
     
           

    # Calculate the response timea47
   
    
    # Display the result and response time
recognizer = sr.Recognizer()

# Sampling rate for voice input
fs = 44100

def record_audio(duration=5):
    image_path = "Voice-Recoder-icon.png"
    
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
        time1=time.time()
        op = chain[0].invoke({'question':inp,'context':retrieval[0]})
      #   print(type(op))
        flag=sentiment_analysis.sentiment(op)
    if(flag==False):
        op=chain[1].invoke(inp)
        op=op['output']
        
    time2=time.time()-time1
    time3=f"**Time taken:** {time2:.2f} seconds"
    with open('predicted_text.csv','a') as file:
      
             writer=csv.writer(file)
             writer.writerow([inp,op,time3])

    google_sheet_connector.google_sheet_connector(inp,op,time3)
    
        # voice output for the project
        # op1=TTS.TTS(op)
        
        # try:
        #  result1=chain[1].invoke(inp)
        #  print(type(result1['output']))
        #  temp=result1['output']
        #  print(temp)
        # except Exception as e:
        #     print(f"exception occur due to {e}")
      #   op = chain.invoke(inp)
         
        # If no chat is selected, start a new chat
    if st.session_state.selected_chat is None:
            st.session_state.first_question = inp
            st.session_state.chat_sessions.append((inp, [("You", inp), ("Bot", op)]))
             #  try:
            
   #  except Exception as e:
   #     print(f"exception occure due to {e}")
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

      #   st.write("### Chat History")
        for sender, message in recent_exchanges:
            st.write(f"**{sender}:** {message}")
            st.write("===========================================================================")
    else:
        pass

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
img = "miniorange-logo.png"
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
            alignment:center;
        }
        .header img {
            max-width: 300px;
            height: auto;
            alignment:center;
        }
        .content {
            margin-top: 100px; 
            alignment:center;
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
        inp = st.chat_input("How can I help you ?")
        if inp:
            process_input(inp)
    else:
        inp = st.chat_input("How can I help you ?", disabled=True)

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
    
    

    
    
    