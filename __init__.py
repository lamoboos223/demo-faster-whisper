import pyaudio
import wave
import speech_recognition as sr
import pyttsx3  # Import pyttsx3 for text-to-speech
from faster_whisper import WhisperModel
import os

model_size = "large-v3"
WAKEUP_WORD = "hello"

# Initialize Whisper model

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Initialize pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 165)  # Adjust the speed, default is 200
engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")  # Set female voice

# Set up audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust this based on your preference
WAVE_OUTPUT_FILENAME = "audio.wav"

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_for_wakeup_word():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("Listening for wakeup word...")
    # speak("Listening for wakeup word")  # Speak the instruction
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            audio = recognizer.listen(source)
            
            try:
                text = recognizer.recognize_sphinx(audio).lower()
                print("Heard:", text)
                
                if WAKEUP_WORD in text:
                    print("Wakeup word detected!")
                    speak("Yes, Master!")  # Speak the confirmation
                    return True
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Error:", e)


def capture_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Listening...")
    # speak("Please start speaking")  # Speak the instruction

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    # speak("Finished recording")  # Speak the confirmation

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save captured audio to a file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio():
    segments, info = model.transcribe(WAVE_OUTPUT_FILENAME, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    # speak("Detected language %s with probability %f" % (info.language, info.language_probability))  # Speak the language detection result

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        speak("Starting the task " + segment.text)  # Speak the transcribed text
        
    # you can make any logic here based on the segment.txt (the spoken word)
        
    # Delete the audio file after processing
    os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    while True:
        if listen_for_wakeup_word():
            capture_audio()
            transcribe_audio()
