import os
import tempfile
import yt_dlp
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
from sklearn.preprocessing import StandardScaler
import json
import gc

app = Flask(__name__)

# Initialize accent classifier as None - will be loaded on first use
accent_classifier = None

def get_accent_classifier():
    global accent_classifier
    if accent_classifier is None:
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        accent_classifier = pipeline(
            "audio-classification",
            model="microsoft/wavlm-base",
            device=0 if torch.cuda.is_available() else -1,
            top_k=4
        )
    return accent_classifier

# Define accent categories and their features
ACCENT_CATEGORIES = {
    "British": {
        "features": ["Received Pronunciation", "Cockney", "Scottish", "Welsh"],
        "description": "Characterized by non-rhotic pronunciation and distinct vowel sounds"
    },
    "American": {
        "features": ["General American", "Southern", "New York", "Midwest"],
        "description": "Known for rhotic pronunciation and flat vowel sounds"
    },
    "Australian": {
        "features": ["Broad", "General", "Cultivated"],
        "description": "Features rising intonation and unique vowel shifts"
    },
    "Other": {
        "features": ["International", "Mixed"],
        "description": "Non-native or mixed accent patterns"
    }
}

def download_video(url):
    """Download video and extract audio."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': temp_file.name.replace('.mp3', ''),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return temp_file.name

def analyze_accent(audio_path):
    """Analyze the accent from audio file."""
    try:
        print(f"Starting accent analysis for {audio_path}")
        
        # Load and preprocess audio
        print("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=16000)
        print(f"Audio loaded successfully. Shape: {y.shape}, Sample rate: {sr}")
        
        # Limit audio to first 5 seconds for inference
        max_duration = 5  # seconds
        if len(y) > sr * max_duration:
            print(f"Trimming audio from {len(y)/sr:.2f}s to {max_duration}s for inference.")
            y = y[:sr * max_duration]
        else:
            print(f"Audio duration is {len(y)/sr:.2f}s, no trimming needed.")
        
        # Extract audio features
        print("Extracting audio features...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        print("Audio features extracted successfully")
        
        # Combine features - ensure all are 1D arrays
        print("Combining features...")
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.array([np.mean(spectral_centroid)]),
            np.array([np.mean(spectral_rolloff)])
        ])
        print(f"Features combined. Shape: {features.shape}")
        
        # Normalize features
        print("Normalizing features...")
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features.reshape(1, -1))
        print("Features normalized successfully")
        
        # Get model predictions
        print("Calling accent_classifier...")
        try:
            # Get the classifier (will load if not already loaded)
            classifier = get_accent_classifier()
            
            # Convert audio to the format expected by the model
            audio_input = {
                "array": y,
                "sampling_rate": sr
            }
            predictions = classifier(audio_input)
            print(f"Model predictions: {predictions}")
            
            # Clear memory after prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as model_error:
            print(f"Error in model prediction: {str(model_error)}")
            # If the first approach fails, try with a different audio format
            try:
                # Resample to 16kHz if needed
                if sr != 16000:
                    y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                # Ensure audio is mono
                if len(y.shape) > 1:
                    y = librosa.to_mono(y)
                
                # Normalize audio
                y = librosa.util.normalize(y)
                
                audio_input = {
                    "array": y,
                    "sampling_rate": sr
                }
                print("Calling accent_classifier (second attempt)...")
                predictions = classifier(audio_input)
                print(f"Alternative model predictions: {predictions}")
                
                # Clear memory after prediction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as second_error:
                print(f"Second attempt failed: {str(second_error)}")
                return {
                    "accent": "Unknown",
                    "confidence": 0,
                    "explanation": "Unable to process audio for accent detection",
                    "features": []
                }
        
        if not predictions:
            print("No predictions returned from model")
            return {
                "accent": "Unknown",
                "confidence": 0,
                "explanation": "Unable to detect accent",
                "features": []
            }
        
        # Process predictions and determine accent
        print("Processing predictions...")
        accent_scores = {}
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            print(f"Processing prediction - Label: {label}, Score: {score}")
            
            # Map labels to our categories
            if any(term in label for term in ['british', 'uk', 'england']):
                accent_scores['British'] = accent_scores.get('British', 0) + score
            elif any(term in label for term in ['american', 'us', 'usa']):
                accent_scores['American'] = accent_scores.get('American', 0) + score
            elif any(term in label for term in ['australian', 'aussie']):
                accent_scores['Australian'] = accent_scores.get('Australian', 0) + score
            else:
                accent_scores['Other'] = accent_scores.get('Other', 0) + score
        
        print(f"Accent scores: {accent_scores}")
        
        # Determine the most likely accent
        if not accent_scores:
            print("No accent scores calculated")
            return {
                "accent": "Unknown",
                "confidence": 0,
                "explanation": "Unable to detect accent",
                "features": []
            }
            
        max_accent = max(accent_scores.items(), key=lambda x: x[1])
        confidence = max_accent[1] * 100
        print(f"Selected accent: {max_accent[0]} with confidence: {confidence}")
        
        return {
            "accent": max_accent[0],
            "confidence": round(confidence, 2),
            "explanation": ACCENT_CATEGORIES[max_accent[0]]["description"],
            "features": ACCENT_CATEGORIES[max_accent[0]]["features"]
        }
    except Exception as e:
        print(f"Error in analyze_accent: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "accent": "Error",
            "confidence": 0,
            "explanation": f"Error during analysis: {str(e)}",
            "features": []
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.json['url']
        print(f"Received URL for analysis: {url}")
        
        # Download and process video
        print("Downloading video...")
        audio_path = download_video(url)
        print(f"Video downloaded to: {audio_path}")
        
        # Analyze accent
        print("Starting accent analysis...")
        result = analyze_accent(audio_path)
        print(f"Analysis result: {result}")
        
        # Cleanup
        print("Cleaning up temporary files...")
        os.unlink(audio_path)
        print("Cleanup complete")
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 