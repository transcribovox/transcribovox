import streamlit as st
import streamlit_lottie
import streamlit_scrollable_textbox as stx
import pathlib
import requests
import json
import whisper
from whisper.utils import get_writer
from pytube import YouTube

from utils import *


def main():
    """
    Main Function
    """
    st.set_page_config(
        page_title="Transcribo Vox",
        page_icon="./assets/favicon.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/smaranjitghose/AIAudioTranscriber',
            'Report a bug': "https://github.com/smaranjitghose/AIAudioTranscriber/issues",
            'About': "## A minimalistic application to generate transcriptions for audio built using Python"
        }
    )
    
    st.title("Transcribo Vox")
    hide_footer()
    # Load and display animation
    placeholder = st.empty()
    with placeholder.container():
        anim = lottie_local("assets/animations/transcriber.json")
        st_lottie(
            anim,
            speed=1,
            reverse=False,
            loop=True,
            quality="medium",  # low; medium ; high
            # renderer="svg",  # canvas
            height=400,
            width=400,
            key=None
        )

    # Initialize Session State Variables
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
        st.session_state["model_path"] = ""
        st.session_state["input_mode"] = ""
        st.session_state["file_path"] = ""
        st.session_state["transcript"] = ""
        st.session_state["lang"] = ""
        st.session_state["segments"] = []

    model_list = {
        "Tiny": r"./assets/models/tiny.pt",
        "Base": r"./assets/models/base.pt",
        "Small": r"./assets/models/small.pt",
        "Medium": r"./assets/models/medium.pt",
        "Large": r"./assets/models/large-v2.pt"
    }

    # Create Input Form Component
    input_mode = st.sidebar.selectbox(
        label="Input Mode",
        options=["Youtube Video URL", "Upload Audio File"]
    )
    st.session_state["input_mode"] = input_mode

    # Create a Form Component on the Sidebar for accepting input data and parameters
    with st.sidebar.form(key="input_form", clear_on_submit=False):

        # Nested Component to take user input for audio file as per selected mode
        if input_mode == "Upload Audio File":
            uploaded_file = st.file_uploader(
                label="Upload your audio 📁", type=["wav", "mp3", "m4a"], accept_multiple_files=False
            )
        elif input_mode == "Youtube Video URL":
            yt_url = st.text_input(label="Paste URL for Youtube Video 🔗")
        
        # Nested Component for model size selection
        model_choice = st.radio(
            label="Choose Your Transcriber 🪖", options=list(model_list.keys())
        )
        st.session_state["model_path"] = model_list[model_choice]

        # Nested Optional Component to select segment of the clip to be used for transcription
        extra_configs = st.expander("Choose Segment ✂")
        with extra_configs:
            start = st.number_input("Start time for the media (sec)", min_value=0, step=1)
            duration = st.number_input("Duration (sec) - negative implies till the end", min_value=-1, max_value=30, step=1)
        submitted = st.form_submit_button(label="Generate Transcripts✨")

        if submitted:
            # Create input and output sub-directories
            APP_DIR = pathlib.Path(__file__).parent.absolute()
            INPUT_DIR = APP_DIR / "input"
            INPUT_DIR.mkdir(exist_ok=True)

            # Load Audio from selected Input Source
            if input_mode == "Upload Audio File":
                if uploaded_file is not None:
                    grab_uploaded_file(uploaded_file, INPUT_DIR)
                    get_transcripts()
                else:
                    st.warning("Please🙏 upload a relevant audio file")
            elif input_mode == "Youtube Video URL":
                if yt_url and validate_YT_link(yt_url):
                    grab_youtube_video(yt_url, INPUT_DIR)
                    get_transcripts()
                else:
                    st.warning("Please🙏 enter a valid URL for Youtube video")

    if st.session_state["transcript"] != "" and st.session_state["lang"] != "":
        col1, col2 = st.columns([3, 1], gap="medium")
        placeholder.empty()
            
        # Display the generated Transcripts
        with col1:
            if st.session_state["input_mode"] == "Youtube Video URL":
                st.markdown("### Youtube Video w/subtitles ▶️")
                st.video(yt_url)
            st.markdown("### Detected language 🌐:")
            st.markdown(f"{st.session_state['lang']}")
            st.markdown("### Generated Transcripts 📃: ")
            # st.markdown(st.session_state["transcript"])
            stx.scrollableTextbox(st.session_state["transcript"]["text"], height=600)
                    # Display the original Audio
        with col2:
            if st.session_state["input_mode"] == "Youtube Video URL":
                st.markdown("### Original Youtube Video ▶️")
                st.video(yt_url)
            st.markdown("### Original Audio 🎵")
            with open(st.session_state["file_path"], "rb") as f:
                st.audio(f.read())
            # Download button
            st.markdown("### Save Transcripts📥")
            out_format = st.radio(label="Choose Format", options=["Text File", "SRT File", "VTT File"])
            transcript_download(out_format)


def grab_uploaded_file(uploaded_file, INPUT_DIR: pathlib.Path):
    """
    Method to store the uploaded audio file to server
    """
    try:
        print("--------------------------------------------")
        print("Attempting to load uploaded audio file ...")
        # Extract file format
        upload_name = uploaded_file.name
        upload_format = upload_name.split(".")[-1]
        # Create file name
        input_name = f"audio.{upload_format}"
        st.session_state["file_path"] = INPUT_DIR / input_name
        # Save the input audio file to server
        with open(st.session_state["file_path"], "wb") as f:
            f.write(uploaded_file.read())
        print("Successfully loaded uploaded audio")

    except Exception as e:
        print(repr(e))
        st.error("😿 Failed to load uploaded audio file")


def grab_youtube_video(url: str, INPUT_DIR: pathlib.Path):
    """
    Method to fetch the audio codec of a Youtube video and save it to server
    """
    try:
        print("--------------------------------------------")
        print("Attempting to fetch audio from Youtube ...")
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        video = yt.streams.get_by_itag(140)
        video.download(INPUT_DIR, filename="audio.mp3")

        print("Successfully fetched audio from Youtube")
        st.session_state["file_path"] = INPUT_DIR / "audio.mp3"

    except Exception as e:
        print(repr(e))
        st.error("😿 Failed to fetch audio from YouTube")


def grab_online_video(url: str, INPUT_DIR: pathlib.Path):
    """
    Method to fetch an online audio file and save it to server
    """
    try:
        print("--------------------------------------------")
        print("Attempting to fetch remote audio file ...")
        # Fetch file
        r = requests.get(url, allow_redirects=True)
        # Extract file format
        file_name = url.split("/")[-1]
        file_format = url.split(".")[-1]
        # Create file name
        input_name = f"audio.{file_format}"
        st.session_state["file_path"] = INPUT_DIR / input_name
        # Save to server storage
        with open(st.session_state["file_path"], "wb") as f:
            f.write(r.content)
        print("Successfully fetched remote audio")

    except Exception as e:
        print(repr(e))
        st.error("😿 Failed to fetch audio file")


# @st.cache
def get_model(model_type: str = 'tiny'):
    """
    Method to load Whisper model to disk
    """
    print("Attempting to download Whisper")
    with open("get_model_weights.py") as f:
        exec(f.read())
        
    try:
        print("--------------------------------------------")
        print("Attempting to load Whisper ...")
        model = whisper.load_model(model_type)
        print("Successfully loaded Whisper")
        return model

    except Exception as e:
        print(repr(e))
        st.error("😿 Failed to load model")


def get_transcripts():
    """
    Method to generate transcripts for the desired audio file
    """
    try:
        # Load Whisper
        model = get_model()
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(st.session_state["file_path"])
        # audio = whisper.pad_or_trim(audio)
        # Pass the audio file to the model and generate transcripts
        print("--------------------------------------------")
        print("Attempting to generate transcripts ...")
        result = model.transcribe(audio)
        print("Successfully generated transcripts")
        # Grab the text and update it in session state for the app
        st.session_state["transcript"] = result["text"]
        st.session_state["lang"] = match_language(result["language"])
        st.session_state["segments"] = result["segments"]
        st.session_state["transcript"] = result
        # Save Transcripts:
        #st.balloons()

    except Exception as e:
        print(repr(e))
        st.error("😿 Model Failed to generate transcripts")


def match_language(lang_code: str) -> str:
    """
    Method to match the language code detected by Whisper to full name of the language
    """
    with open("./language.json","rb") as f:
        lang_data = json.load(f)
    
    return lang_data[lang_code].capitalize()


def transcript_download(out_format: str):
    """
    Method to save transcripts in TXT, SRT or VTT format
    """

    # Create Output sub-directory if it does not exist already
    APP_DIR = pathlib.Path(__file__).parent.absolute()
    OUTPUT_DIR = APP_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create a dict of out_format and the file type
    file_type_dict = {"Text File": "txt", "SRT File": "srt", "VTT File": "vtt"}

    # Select the file type
    file_type = file_type_dict[out_format]

    if out_format in file_type_dict.keys():
        # Generate Transcript file as per choice
        writer = get_writer(file_type, OUTPUT_DIR)
        writer(
            st.session_state["transcript"],
            st.session_state["file_path"],
            {
                "max_line_width": None,
                "max_line_count": None,
                "highlight_words": True
            }
        )

        # Generate SRT File for Transcript  
        with open(OUTPUT_DIR/f'audio.{file_type}', "r", encoding="utf-8") as f:
            st.download_button(
                label="Click to download 🔽",
                data=f,
                file_name=f"transcripts.{file_type}",
            )


if __name__ == "__main__":
    main()
     
