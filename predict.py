import warnings
import whisper
import joblib
warnings.filterwarnings("ignore")

def predict_emotion(audio_path, input_text=None):
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Loading emotion detection model...")
    emotion_model = joblib.load("emotion_model.joblib")

    print(f"Transcribing {audio_path} ...")
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    print(f"Transcribed Text (from audio): {transcribed_text}")

    audio_emotion = emotion_model.predict([transcribed_text])[0]
    print(f"Audio Output Emotion: {audio_emotion}")

    if input_text:
        print(f"Input Text (provided directly): {input_text}")
        text_emotion = emotion_model.predict([input_text])[0]
        print(f"Text Output Emotion: {text_emotion}")
    else:
        print("No direct text input provided.")
    
if __name__ == "__main__":
    predict_emotion("audio.wav", input_text="I am not okay.")