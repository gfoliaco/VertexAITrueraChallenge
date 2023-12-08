import streamlit as st
import pandas as pd
import os
import json
#import utils
import openai
import subprocess
import webbrowser
from io import StringIO
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
import time
import vertexai
from vertexai.language_models import TextGenerationModel
from google.cloud import speech
from pydub import AudioSegment
from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from trulens_eval import Tru
import glob
from utils import get_prebuilt_trulens_recorder
from utils import build_sentence_window_index
from utils import get_sentence_window_query_engine

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_accountZ.json"
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
ruta_guardado = "/home/german/audios/"

def run_trulens_eval():
    # Ejecuta el comando trulens-eval
    subprocess.run(["trulens-eval"])

def save_uploaded_file(uploaded_file, file_path):
    with open(file_path, 'wb') as resumenTxt:  # Use 'wb' for binary write mode
        resumenTxt.write(uploaded_file.getvalue())

# GET A TEXT SUMMARY USING TEXT-BISON
def run_text_summary(file_path):
    # Read the content of the file into the 'text' variable
    with open(file_path, "rb") as file:
        # Use 'latin-1' encoding to handle non-UTF-8 encoded content
        content_bytes = file.read()
        text = content_bytes.decode("latin-1")

    vertexai.init(project="unified-era-355307", location="us-central1")
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(
        f"""Text generation using the content of the following file:
        "{text}"
        """,
        **parameters
    )
    return response

# MEETING MANAGEMENT SPEECH TO TEXT AND AUDIO DIARIZATION FROM VERTEXAI
def run_Audio_Diarization(audio_file_path):
    client = speech.SpeechClient()

    speaker_diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=10,
    )

    # Configure request to enable Speaker diarization
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        sample_rate_hertz=16000,
        diarization_config=speaker_diarization_config,
        model='latest_long'
    )

    # Set the remote path for the audio file
    audio = speech.RecognitionAudio(uri=audio_file_path)

    # Use non-blocking call for getting file transcription
    response = client.long_running_recognize(
        config=recognition_config, audio=audio
    ).result(timeout=300000)

    # The transcript within each result is separate and sequential per result.
    # However, the words list within an alternative includes all the words
    # from all the results thus far. Thus, to get all the words with speaker
    # tags, you only have to take the words list from the last result
    result = response.results[-1]
    words_info = result.alternatives[0].words

    time.sleep(2)
    audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    Transcripcion =f'/home/german/artifacts/{audio_file_name}_transcript.txt'

    # JSON PROCESSING BEFORE LLM CALL
    with open(Transcripcion, 'w') as f:
        current_speaker_tag = None  # Inicializamos la etiqueta del hablante actual
        for word_info in words_info:
            if word_info.speaker_tag != current_speaker_tag:
                if current_speaker_tag is not None:
                    f.write('\n')  # Agregamos un salto de línea si el speaker_tag cambia
                f.write(f'VOICE:{word_info.speaker_tag}\n')  # Escribimos la nueva etiqueta del hablante
                current_speaker_tag = word_info.speaker_tag

            f.write(f"{word_info.word} ")  # Agregamos la palabra
    return

def main():
    region_name = 'us-east-1'
    now = datetime.now()
    resumenTxt = 'resumenTxt.txt'
    string_data = "a resumir"
    audio_name = "default_name"
    carpeta_con_archivos = "/home/german/artifacts/"
    # Utiliza glob para obtener la lista de archivos en la carpeta
    archivos_en_carpeta = glob.glob(carpeta_con_archivos + "*.txt")
    # Utiliza SimpleDirectoryReader con la lista de archivos como input_files
    documents = SimpleDirectoryReader(input_files=archivos_en_carpeta).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
    )
    index = VectorStoreIndex.from_documents([document], service_context=service_context)
    sentence_index = build_sentence_window_index(
    	document,
    	llm,
    	embed_model="local:BAAI/bge-small-en-v1.5",
    	save_dir="sentence_index"
    ) 
    # Aplicacion Streamlit
    st.write("""# Milestone Master""")
    ajuste = st.text_input("What do you want to know today about Project X?")
    st.write(ajuste)
    
    if ajuste:  # Ejecutar solo si se proporciona un texto en el input
        query_engine = index.as_query_engine()
        response = query_engine.query(ajuste)
        st.write(str(response))
        
        sentence_window_engine = get_sentence_window_query_engine(sentence_index)
        window_response = sentence_window_engine.query(ajuste)
        st.write(str(window_response))


        tru = Tru()
        tru.reset_database()
        tru_recorder = get_prebuilt_trulens_recorder(query_engine, app_id="Direct Query Engine")
        tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
            sentence_window_engine,
            app_id = "Sentence Window Query Engine"
        )

    
    if st.button("Ejecutar trulens-eval"):

       with tru_recorder as recording:
           response = query_engine.query(ajuste)

       records, feedback = tru.get_records_and_feedback(app_ids=[])
       #records.head()
       #st.write(records.head())
     
       with tru_recorder_sentence_window as recording:
           response = sentence_window_engine.query(ajuste)

       run_trulens_eval()   
       webbrowser.open("http://34.42.30.236:8501/")
   
 # SIDEBAR MEETING AND DOCUMENTS UPLOAD
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a document")
    if uploaded_file is not None:
        try:
            # Get the filename and file extension
            filename, file_extension = os.path.splitext(uploaded_file.name)
            # Save the uploaded file with its original filename in the specified directory
            file_path = os.path.join("artifacts/", uploaded_file.name)
            save_uploaded_file(uploaded_file, file_path)
            index = VectorStoreIndex.from_documents([document], service_context=service_context)
            # Run text summary using the uploaded file
            response = run_text_summary(file_path)
            st.write(f"Response from Model: {response.text}")

        except Exception as e:
            st.error(f"Error processing the uploaded meeting file: {str(e)}")

    with st.sidebar:
        uploaded_meeting = st.file_uploader("Upload a meeting audio or video")
    if uploaded_meeting is not None:
        try:
            # Obtiene el nombre original del archivo cargado
            audio_name = uploaded_meeting.name.replace(' ', '_')
            
            # Procesa el archivo de audio o video
            audio_segment = AudioSegment.from_file(uploaded_meeting)
            converted_audio = audio_segment.set_channels(1).set_frame_rate(16000)

            # Guarda el archivo con el mismo nombre que se cargó
            ruta_salida = os.path.join(ruta_guardado, f"{audio_name}.wav")
            converted_audio.export(ruta_salida, format="wav")
            # Enviamos archivo a bucket en GCP
            client = storage.Client()
            # Nombre del archivo en Google Cloud Storage
            bucket_name = "pruebasvarias1"
            blob_name = f"{audio_name}.wav"
            # Carga el archivo en GCS
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(ruta_salida)
            st.success("Archivo exitosamente enviado a GCP Speech to text")
            # GET TRANSCRIPT TXT FROM VERTEX AI 
            ResponseSpeech = run_Audio_Diarization(f'gs://pruebasvarias1/{audio_name}.wav')
        
            # Guarda el archivo transcript con el mismo nombre que se cargó
            file_path2 = f'/home/german/artifacts/{audio_name}_transcript.txt'
            index = VectorStoreIndex.from_documents([document], service_context=service_context)
            response = run_text_summary(file_path2)
            st.write(f"Response from Model: {response.text}")

            with open(file_path2, 'r', encoding='utf-8') as file:
                st.success(f"Meeting file '{audio_name}' successfully processed and saved.")
        except Exception as e:
            st.error(f"Error processing the uploaded meeting file: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="MilestoneMaster", page_icon="icono"
    )
    main()
