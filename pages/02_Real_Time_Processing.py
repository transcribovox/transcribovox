"""
Contact Page using formsubmit.co API
"""
import streamlit as st
from utils import *


st.set_page_config(
        page_title="AI Audio Transciber",
        page_icon="./assets/favicon.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        'Get Help': 'https://github.com/smaranjitghose/AIAudioTranscriber',
        'Report a bug': "https://github.com/smaranjitghose/AIAudioTranscriber/issues",
        'About': "## A minimalistic application to generate transcriptions for audio built using Python"
        }
)


st.title("Transcribo Vox - Real-Time Processing")

# https://towardsdatascience.com/developing-web-based-real-time-video-audio-processing-apps-quickly-with-streamlit-7c7bcd0bc5a8
# https://github.com/whitphx/streamlit-webrtc



hide_footer()


