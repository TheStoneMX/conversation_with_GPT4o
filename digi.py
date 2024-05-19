import openai
from openai import OpenAI
import base64
import os
import time
import datetime
import pyautogui
from pathlib import Path
import tempfile
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import wave
from faster_whisper import WhisperModel
import audioop

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

openai.api_key = 'x'
client = OpenAI(api_key=openai.api_key)

def gpt4o_chat(user_query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Software Dev with Python Expertise. HELP THE USER WITH CODE ERRORS AND EXPLAINATIONS. KEEP IT VERY SHORT CONCISE AND CONVERSATIONAL"},
            {"role": "user", "content": user_query}
            
        ],
        temperature=0.4,
        max_tokens=200,  # Reduced max_tokens for faster response
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0
    )
    return response.choices[0].message.content.strip()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path):
    base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract ONLY code-related text and data from the image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
        max_tokens=200  # Reduced max_tokens for faster response
    )
    return response.choices[0].message.content.strip()

def take_screenshot(filepath):
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)

def text_to_speech_and_play(text):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpfile:
        speech_file_path = tmpfile.name

    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format='mp3'
    )

    with open(speech_file_path, 'wb') as f:
        f.write(response.content)

    audio = AudioSegment.from_mp3(speech_file_path)
    play(audio)

    Path(speech_file_path).unlink()

# Set up the faster-whisper model
model_size = "medium.en"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Function to transcribe the recorded audio using faster-whisper
def transcribe_with_whisper(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# Function to record audio from the microphone and save to a file
def record_audio(file_path, silence_threshold=1000, speech_threshold=1000, chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    frames = []
    silence_count = 0
    silence_frames = int(rate / chunk_size * 1.5)  # 1.5 seconds of silence
    speech_frames = int(rate / chunk_size * 0.3)  # 0.3 seconds of speech

    print("Waiting for speech...")

    while True:
        data = stream.read(chunk_size)
        rms = audioop.rms(data, 2)

        if rms > speech_threshold:
            print("Recording...")
            break

    frames.append(data)

    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        rms = audioop.rms(data, 2)
        if rms < silence_threshold:
            silence_count += 1
            if silence_count > silence_frames:
                break
        else:
            silence_count = 0

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    folder_path = "PATH/images/"
    os.makedirs(folder_path, exist_ok=True)

    print(f"{CYAN}Continuous Code Assistant is now observing...{RESET_COLOR}")

    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if "exit" in user_input.lower():
            break

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        screenshot_path = os.path.join(folder_path, f"screenshot_{timestamp}.png")

        take_screenshot(screenshot_path)
        code_related_text = analyze_image(screenshot_path)

        if code_related_text:
            feedback = gpt4o_chat(f"Code Related Text = {code_related_text}\n\nUser Question: {user_input}\n\nAnswer the {{User Question}} with any fluff or verbose rants:")
            text_to_speech_and_play(feedback)
        else:
            print(f"{YELLOW}No code-related text found in the screenshot.{RESET_COLOR}")

        os.remove(screenshot_path)

if __name__ == "__main__":
    main()