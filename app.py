import streamlit as st
import io
import azure.cognitiveservices.speech as speechsdk
import joblib
import numpy as np
import os





speech_key = os.getenv("")
speech_region = os.getenv("")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

#import model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

#emojyes
emotions_emoji_dict = {"anger": "__😠__", "disgust": "__🤮__", "fear": "__😨😱__", "happy": "__🤗__", "joy": "__😂__", "neutral": "__😐__", "sad": "__😔__",
                       "sadness": "__😔__", "shame": "__😳__", "surprise": "__😮__"}


# Streamlit UI
st.title("Audio Recorder")

# Record the audio
audio_data = st.audio_input("Record a voice message")

# Check if the audio data is available
if audio_data:
    st.audio(audio_data)  # Plays the recorded audio in the app
    
    # Convert the audio to a byte-like object
    audio_bytes = audio_data.getvalue()
    
    # Save the audio to a local file
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_bytes)



    # Specify the audio file to transcribe
    audio_file_path = "recorded_audio.wav"
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    # Create a speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Transcribe the audio file
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        st.write(result.text)
        results = pipe_lr.predict([result.text])
        
        emojy = emotions_emoji_dict[results[0]]
        st.write("{}:{}".format(results[0], emojy))
        st.write("Confidence:{}".format(np.max(pipe_lr.predict_proba([result.text]))))




    elif result.reason == speechsdk.ResultReason.NoMatch:
        st.write("No speech could be recognized.")





    


