# Accent Detection Tool

This tool analyzes video URLs to detect English accents and provide confidence scores for accent classification.

## Features

- Accepts video URLs (YouTube, Loom, direct MP4 links)
- Extracts audio from videos
- Analyzes speaker's accent
- Provides accent classification (British, American, Australian, etc.)
- Generates confidence scores
- Includes brief explanation of the analysis

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter a video URL in the input field
2. Click "Analyze"
3. Wait for the analysis to complete
4. View the results including:
   - Accent classification
   - Confidence score
   - Brief explanation

## Technical Details

The application uses:

- Flask for the web interface
- yt-dlp for video downloading
- librosa for audio processing
- Transformers for accent classification
- scikit-learn for confidence scoring

## Note

This is a proof-of-concept tool and may not be 100% accurate in all cases. It's designed to work with clear English speech and may have limitations with:

- Poor audio quality
- Multiple speakers
- Non-English speech
- Background noise
