
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        print("Listening to audio...")
        audio_data = recognizer.record(source)
        print("Transcribing...")
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription:")
            print(text)
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Request failed; {e}")