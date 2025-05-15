
# Speech & Text Emotion Detection

## Project Overview
- Detect emotions from both **text** and **audio** inputs using machine learning and speech-to-text transcription.
- Dataset: Twitter-based CSV with `content` and `sentiment` labels.

## Approach
- Use **OpenAI Whisper** to transcribe audio to text.
- Train a **Naive Bayes** classifier on the labeled text data.
- Predict emotions from transcribed audio and direct text input.

## Tools & Technologies
- Python
- OpenAI Whisper
- scikit-learn
- pandas
- pyttsx3 (text-to-speech)
- joblib (model saving/loading)
- FFmpeg (audio processing)

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model and generate audio sample:
   ```bash
   python train.py
   ```
3. Predict emotion from audio and text:
   ```bash
   python predict.py
   ```

## Results
- Achieved approximately **30% accuracy** on multi-class emotion classification.
- Better performance on common emotions like **happiness** and **worry**.

## Future Improvements
- Use advanced models such as **transformers** for better accuracy.
- Apply data balancing and augmentation techniques.
- Incorporate audio features for direct speech emotion recognition.
- Develop a user-friendly interface for real-time emotion detection.
