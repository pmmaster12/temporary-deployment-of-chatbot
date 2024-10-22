# Import the required module for text 
# to speech conversion
from gtts import gTTS

# This module is imported so that we can 
# play the converted audio
import os
def TTS(response):
# The text that you want to convert to audio
# mytext = 'Welcome to geeksforgeeks Joe!'

# Language in which you want to convert
 language = 'en'

# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
 myobj = gTTS(text=response, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome 
 myobj.save("welcome.mp3")
# Playing the converted file
 return os.system("start welcome.mp3")

# x=TTS("tu chutiya hai")
