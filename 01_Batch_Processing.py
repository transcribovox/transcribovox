import streamlit as st
from streamlit.logger import get_logger
import streamlit_scrollable_textbox as stx
import pathlib
import requests
import json
import whisper
from whisper.utils import get_writer
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from pytube import YouTube
from transformers import (
    pipeline, GenerationConfig,
    WhisperForConditionalGeneration,
    AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
)
import librosa
import os

from utils import *

logger = get_logger(__name__)

language_mapping = {
    "English": 'en',
    "French": 'fr',
    "German": 'de',
    "Luxembourgish": 'lb'
}

models_path = {
    "en": r"./assets/models/whisper-small/",
    "fr": r"./assets/models/whisper-small/",
    "de": r"./assets/models/whisper-small/",
    "lb": r"./assets/models/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm/"
}


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
    
    st.title("Transcribo Vox - Batch Processing")
    hide_footer()
    # Load and display animation
    # placeholder = st.empty()
    # with placeholder.container():
    #     anim = lottie_local("assets/animations/transcriber.json")
    #     st_lottie(
    #         anim,
    #         speed=1,
    #         reverse=False,
    #         loop=True,
    #         quality="medium",  # low; medium ; high
    #         # renderer="svg",  # canvas
    #         height=400,
    #         width=400,
    #         key=None
    #     )

    # Initialize Session State Variables
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
        st.session_state["models_path"] = {}
        st.session_state["input_mode"] = ""
        st.session_state["audio_file_path"] = ""
        st.session_state["audio_chunks"] = {}
        st.session_state["video_file_path"] = ""
        st.session_state["sub_video_file_path"] = ""
        st.session_state["transcript"] = ""
        st.session_state["segments"] = []
        st.session_state["srt_file_path"] = ""

    # Create Input Form Component
    input_mode = st.sidebar.selectbox(
        label="Input Mode",
        options=["Upload Video File", "Upload Audio File", "Youtube Video URL"]
    )

    chunks_number = st.sidebar.number_input("Number of language chunks", min_value=1, value=1, step=1)

    st.session_state["input_mode"] = input_mode

    # Create a Form Component on the Sidebar for accepting input data and parameters
    with st.sidebar.form(key="input_form", clear_on_submit=False):

        # Nested Component to take user input for audio file as per selected mode
        if input_mode == "Upload Video File":
            uploaded_file = st.file_uploader(
                label="Upload your video 📁", type=["mp4"], accept_multiple_files=False
            )
        elif input_mode == "Upload Audio File":
            uploaded_file = st.file_uploader(
                label="Upload your audio 📁", type=["wav", "mp3", "m4a"], accept_multiple_files=False
            )
        elif input_mode == "Youtube Video URL":
            yt_url = st.text_input(label="Paste URL for Youtube Video 🔗")
        
        # Nested Component for model size selection
        # model_choice = st.radio(
        #     label="Choose Your Transcriber 🪖", options=list(model_list.keys())
        # )
        # st.session_state["model_path"] = model_list[model_choice]
        st.session_state["models_path"] = models_path

        st.session_state["audio_chunks"] = {}

        for i in range(chunks_number):

            extra_configs = st.expander("Language Segment #{}".format(i+1))
            with extra_configs:

                col1, col2 = st.columns(2)
                with col1:
                    start = st.number_input("Start", min_value=0, value=i, step=0, key=i)
                with col2:
                    end = st.number_input("End", min_value=-1, value=-1, step=0, key=i+50)

                lang = st.selectbox(
                    'Language',
                    ('English', 'French', 'German', 'Luxembourgish'),
                    key=i+100
                )
                sub_audio_key = '{}_{}'.format(start, end)
                if sub_audio_key != '0_0':
                    st.session_state["audio_chunks"][sub_audio_key] = {
                        'lang': language_mapping[lang]
                    }

        submitted = st.form_submit_button(label="Generate Transcripts ✨")

        if submitted:
            # Create input and output sub-directories
            app_dir = pathlib.Path(__file__).parent.absolute()
            input_dir = app_dir / "input"
            input_dir.mkdir(exist_ok=True)

            # Load Audio from selected Input Source
            if input_mode == "Upload Video File":
                if uploaded_file is not None:
                    grab_uploaded_file(uploaded_file, input_dir)
                    video_to_audio(input_dir)
                    cut_audio()
                    get_transcripts()
                    insert_subtitles()
                else:
                    st.warning("Please🙏 upload a relevant video file")
            elif input_mode == "Upload Audio File":
                if uploaded_file is not None:
                    grab_uploaded_file(uploaded_file, input_dir)
                    cut_audio()
                    get_transcripts()
                else:
                    st.warning("Please🙏 upload a relevant audio file")
            elif input_mode == "Youtube Video URL":
                if yt_url and validate_YT_link(yt_url):
                    grab_youtube_video(yt_url, input_dir)
                    video_to_audio(input_dir)
                    cut_audio()
                    get_transcripts()
                    insert_subtitles()
                else:
                    st.warning("Please🙏 enter a valid URL for Youtube video")

    if st.session_state["transcript"] != "" and st.session_state["audio_chunks"] != {}:
        col1, col2 = st.columns([2, 2], gap="medium")
        # placeholder.empty()
            
        # Display the generated Transcripts
        with col1:
            st.markdown("### Detected language 🌐:")

            st.markdown(get_str_segments())
            st.markdown("### Generated Transcripts 📃: ")
            # st.markdown(st.session_state["transcript"])
            stx.scrollableTextbox(st.session_state["transcript"]['text'], height=600)
        # Display the original Audio
        with col2:
            if st.session_state["input_mode"] == "Upload Audio File":
                st.markdown("### Original Audio 🎵")
                with open(st.session_state["audio_file_path"], "rb") as f:
                    st.audio(f.read())
            else:
                st.markdown("### Video w/subtitles ▶️")
                if st.session_state["sub_video_file_path"] != '':
                    st.video(st.session_state["sub_video_file_path"])
                else:
                    st.video(st.session_state["video_file_path"])

            # Download button
            st.markdown("### Save Transcripts📥")
            out_format = st.radio(label="Choose Format", options=["Text File", "SRT File", "VTT File"])
            transcript_download(out_format)


def get_str_segments():

    text = ''

    for chunk_range, chunk_dict in st.session_state["audio_chunks"].items():
        start, end = chunk_range.split('_')

        if end == -1:
            end = 'end'

        lang_str = [k for k, v in language_mapping.items() if v == chunk_dict['lang']][0]
        text = '{} \n- From {} s to {} s : {}'.format(text, start, end, lang_str)

    return text


def grab_uploaded_file(uploaded_file, input_dir: pathlib.Path):
    """
    Method to store the uploaded file to server
    """
    try:
        logger.info("--------------------------------------------")
        logger.info("Attempting to load uploaded file ...")
        # Extract file format
        upload_name = uploaded_file.name
        upload_format = upload_name.split(".")[-1]

        if upload_format == 'mp4':
            input_name = f"video.{upload_format}"
            file_path_key = "video_file_path"

        else:
            input_name = f"audio.{upload_format}"
            file_path_key = "audio_file_path"

        st.session_state[file_path_key] = os.path.join(input_dir, input_name)

        # Save the input audio file to server
        with open(st.session_state[file_path_key], "wb") as f:
            f.write(uploaded_file.read())

        logger.info("Successfully loaded uploaded file")

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Failed to load uploaded file")


def video_to_audio(input_dir: pathlib.Path):
    """
    Method to convert video into audio
    """
    try:
        logger.info("--------------------------------------------")
        logger.info("Attempting to convert video into audio ...")

        audio = AudioFileClip(st.session_state["video_file_path"])

        # cut audio if needed
        # audio = audio.subclip(0, 60)

        st.session_state["audio_file_path"] = os.path.join(input_dir, 'audio.mp3')

        audio.write_audiofile(st.session_state["audio_file_path"])

        # duration = audio.duration

        logger.info("Successfully converted video into audio")

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Failed to load uploaded file")


def grab_youtube_video(url: str, input_dir: pathlib.Path):
    """
    Method to fetch the audio codec of a Youtube video and save it to server
    """
    try:
        logger.info("--------------------------------------------")
        logger.info("Attempting to fetch video from Youtube ...")
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        video = yt.streams.get_highest_resolution()
        video.download(input_dir, filename="video.mp4")

        logger.info("Successfully fetched video from Youtube")
        st.session_state["video_file_path"] = os.path.join(input_dir, "video.mp4")

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Failed to fetch video from YouTube")


def cut_audio():
    try:
        logger.info("--------------------------------------------")
        logger.info("Attempting to cut audio into chunks ...")
        audio = AudioFileClip(st.session_state["audio_file_path"])

        audio_duration = audio.duration

        for chunk_range, chunk_dict in st.session_state["audio_chunks"].items():

            start, end = chunk_range.split('_')
            if end == '-1':
                end = audio_duration
            sub_audio = audio.subclip(start, end)

            sub_audio_name = '{}_{}_audio.mp3'.format(start, end)
            sub_audio_path = os.path.join(os.path.dirname(st.session_state["audio_file_path"]), sub_audio_name)
            sub_audio.write_audiofile(sub_audio_path)
            st.session_state["audio_chunks"][chunk_range]['audio_path'] = sub_audio_path

        logger.info("Successfully cut audio into chunks")

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Failed to cut audio into chunks")


# @st.cache
def load_models():

    try:

        models = {}
        chunk_langs = list(set([chunk_dict['lang'] for chunk_dict in st.session_state["audio_chunks"].values()]))

        for lang, model_path in st.session_state["models_path"].items():

            if lang in chunk_langs:

                logger.info("--------------------------------------------")
                logger.info("Attempting to load {} model ...".format(lang.upper()))

                # load tokenizer
                tokenizer_path = os.path.join(model_path, "tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                # load feature_extractor
                feature_extractor_path = os.path.join(model_path, "feature_extractor")
                feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

                # load model
                actual_model_path = os.path.join(model_path, "model")
                if lang == 'lb':
                    model = AutoModelForCTC.from_pretrained(actual_model_path)
                else:
                    # used for language detection as first step
                    model = WhisperForConditionalGeneration.from_pretrained(actual_model_path)

                generation_config = GenerationConfig.from_pretrained("openai/whisper-small")
                generation_config.task_to_id = {"transcribe": 50359}
                generation_config.is_multilingual = False
                generation_config.lang_to_id = {
                    "<|de|>": 50261,
                    "<|en|>": 50259,
                    "<|fr|>": 50265,
                    "<|lb|>": 50345,
                }
                model.generation_config = generation_config
                # "no_timestamps_token_id": 50363,

                models[lang] = {
                    "tokenizer": tokenizer,
                    "feature_extractor": feature_extractor,
                    "model": model
                }

        logger.info("Successfully loaded models")

        # load decoder
        # decoder_path = os.path.join(model_path, "decoder")
        # if not os.path.isdir(decoder_path):
        #     logger.info("\tDownloading decoder..")
        #     decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name)
        #     decoder.save_to_dir(decoder_path)
        # else:
        #     logger.info("\tDecoder already present locally. Just loading..")
        #     decoder = BeamSearchDecoderCTC.load_from_dir(decoder_path)

        # Punctuation Capitalization model
        # punct_model = None
        # if 'whisper' not in dir_name:
        #     punct_model_path = os.path.join(model_path, "punctuation_model")
        #     if not os.path.isdir(punct_model_path):
        #         logger.info("\tDownloading punctuation model..")
        #     else:
        #         logger.info("\tPunctuation model already present locally. Just loading..")
        # punct_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

        return models

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Failed to load model")


def get_transcripts():
    """
    Method to generate transcripts for the desired audio file
    """
    try:
        models = load_models()

        logger.info("--------------------------------------------")
        logger.info("Attempting to generate transcripts ...")

        transcript = {'text': '', 'segments': []}

        last_timestamp = 0

        for chunk_dict in st.session_state["audio_chunks"].values():

            audio_path = chunk_dict['audio_path']
            lang = chunk_dict['lang']

            input_audio, sr = librosa.load(audio_path, sr=16000)

            model = models[lang]

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model['model'],
                tokenizer=model['tokenizer'],
                feature_extractor=model['feature_extractor'],
                device=0,
                # decoder=model["decoder"]
            )

            generate_kwargs = {"task": "transcribe", "language": "<|{}|>".format(lang)}

            result = pipe(
                # (https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
                input_audio,
                return_timestamps='word',
                chunk_length_s=30,  # useful for long transcriptions (https://huggingface.co/blog/asr-chunking)
                stride_length_s=(6, 4),
                max_new_tokens=448,
                # batch_size=32,  # (https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching)
                generate_kwargs=generate_kwargs
                # generate_kwargs={
                #     "task": "transcribe",
                #     # "task_to_id": {"transcribe": 50359},
                #     # "no_timestamps_token_id": 50363,
                #     # "language": "fr"
                # }  # otherwise it will transcribe AND translate in english
                # generate_kwargs={"task": "transcribe", "language": "fr"}
            )

            transcript['text'] = transcript['text'] + ' ' + result['text']

            number_of_chunks = len(result["chunks"])  # 102
            words_per_segment = 10
            number_of_segments = int(number_of_chunks / words_per_segment)  # 10
            if number_of_segments > 0:
                for i in range(number_of_segments):
                    if (i + 1) * words_per_segment <= number_of_chunks:
                        current_chunks = result["chunks"][i * words_per_segment: (i + 1) * words_per_segment]
                    else:
                        if i * words_per_segment < number_of_chunks:
                            current_chunks = result["chunks"][i * words_per_segment:]
                        else:
                            current_chunks = [result["chunks"][i * words_per_segment]]

                    start = current_chunks[0]['timestamp'][0]
                    end = current_chunks[-1]['timestamp'][1]
                    text = ' '.join([chunk['text'] for chunk in current_chunks])
            else:
                current_chunks = result["chunks"]
                start = current_chunks[0]['timestamp'][0]
                end = current_chunks[-1]['timestamp'][1]
                text = ' '.join([chunk['text'] for chunk in current_chunks])

            transcript['segments'].append({
                'start': start + last_timestamp,
                'end': end + last_timestamp,
                'text': text
            })

            last_timestamp = last_timestamp + end

        # audio = whisper.pad_or_trim()

        # logger.info("--------------------------------------------")
        # logger.info("Attempting to generate transcripts ...")
        # result = model.transcribe(audio)
        logger.info("Successfully generated transcripts")
        # Grab the text and update it in session state for the app
        st.session_state["transcript"] = transcript
        # st.session_state["segments"] = segments

        # generate subtitles:
        create_transcript_files()
        st.session_state["srt_file_path"] = os.path.join('output', 'audio.srt')

    except Exception as e:
        logger.info(repr(e))
        st.error("😿 Model Failed to generate transcripts")


# def match_language(lang_code: str) -> str:
#     """
#     Method to match the language code detected by Whisper to full name of the language
#     """
#     with open("./language.json", "rb") as f:
#         lang_data = json.load(f)
#
#     return lang_data[lang_code].capitalize()


def create_transcript_files():
    """
        Method to save transcripts in TXT, SRT or VTT format
    """
    # Create Output sub-directory if it does not exist already
    app_dir = pathlib.Path(__file__).parent.absolute()
    output_dir = app_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Create a dict of out_format and the file type
    file_type_dict = {"Text File": "txt", "SRT File": "srt", "VTT File": "vtt"}

    for out_format, file_type in file_type_dict.items():
        # Generate Transcript file as per choice
        writer = get_writer(file_type, output_dir)
        writer(
            st.session_state["transcript"],
            st.session_state["audio_file_path"],
            {
                "max_line_width": None,
                "max_line_count": None,
                "highlight_words": True
            }
        )


def insert_subtitles():
    logger.info("--------------------------------------------")
    logger.info("Attempting to insert subtitles ...")
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=30, color='white')
    subs = SubtitlesClip(st.session_state["srt_file_path"], generator)
    subtitles = SubtitlesClip(subs, generator)

    video = VideoFileClip(st.session_state["video_file_path"])
    result = CompositeVideoClip([video, subtitles.set_position(('center', 600))])

    output_dir = 'output'
    st.session_state["sub_video_file_path"] = os.path.join(output_dir, 'sub_video.mp4')
    result.write_videofile(st.session_state["sub_video_file_path"])
    logger.info("Successfully inserted subtitles")
    st.balloons()


def transcript_download(out_format: str):
    """
    Method to save transcripts in TXT, SRT or VTT format
    """
    # Create Output sub-directory if it does not exist already
    app_dir = pathlib.Path(__file__).parent.absolute()
    output_dir = app_dir / "output"

    # Create a dict of out_format and the file type
    file_type_dict = {"Text File": "txt", "SRT File": "srt", "VTT File": "vtt"}

    # Select the file type
    file_type = file_type_dict[out_format]

    if out_format in file_type_dict.keys():

        # Generate SRT File for Transcript  
        with open(output_dir/f'audio.{file_type}', "r", encoding="utf-8") as f:
            st.download_button(
                label="Click to download 🔽",
                data=f,
                file_name=f"transcripts.{file_type}",
            )


if __name__ == "__main__":
    main()
     
