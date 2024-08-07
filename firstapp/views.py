import os
import json
import yt_dlp as youtube_dl
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
from django.shortcuts import render
from django.http import JsonResponse

MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
MODEL_REVISION = 'a4f8f3e'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

def index(request):
    return render(request, 'firstapp/index.html')

def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
        'postprocessor_args': [
            '-movflags',
            'faststart'
        ],
        'prefer_ffmpeg': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def convert_audio_to_wav():
    sound = AudioSegment.from_mp3("audio.mp3")
    sound.export("audio.wav", format="wav")

def transcribe_audio(audio_file, language):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_wav(audio_file)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return "", language

    chunk_length_ms = 60000  # 1 minute chunks (60000 ms)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = ""
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        print(f"Exported chunk {i} to {chunk_filename}")

        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                print(f"Transcribed chunk {i}: {text}")
                transcription += text + " "
            except sr.RequestError as e:
                print(f"RequestError: Could not request results from Google Speech Recognition service; {e}")
            except sr.UnknownValueError:
                print(f"UnknownValueError: Google Speech Recognition could not understand audio in chunk {i}")
            except Exception as e:
                print(f"Error transcribing chunk {i}: {e}")

        os.remove(chunk_filename)
        print(f"Removed chunk file {chunk_filename}")

    return transcription.strip(), language

def summarize_text(text):
    summarizer = pipeline("summarization", model=MODEL_NAME, revision=MODEL_REVISION)
    max_chunk_length = 1024

    # Tokenize the text
    tokens = tokenizer(text, return_tensors='pt', truncation=False).input_ids[0]

    # Split tokens into chunks
    chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

def detect_language(text):
    translator = Translator()
    lang = translator.detect(text).lang
    return lang

def transcribe(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        url = data['url']
        language = data.get('language', 'en')

        try:
            # Step 1: Download YouTube video and convert to WAV
            download_youtube_video(url)
            convert_audio_to_wav()

            # Step 2: Transcribe audio
            transcription, detected_language = transcribe_audio("audio.wav", language)
            if not transcription:
                return JsonResponse({'error': "Transcription failed or produced no output."})

            # Step 3: Summarize transcription
            summary = summarize_text(transcription)

            return JsonResponse({
                'transcription': transcription.strip(),
                'summary': summary,
                'detected_language': detected_language
            })

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'}, status=400)
