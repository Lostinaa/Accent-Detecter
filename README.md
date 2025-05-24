# Accent Detection Tool

A web application that analyzes accents from YouTube videos using machine learning.

## Live Demo

The application is deployed and available at: [https://accent-detecter-production.up.railway.app/](https://accent-detecter-production.up.railway.app/)

## Features

- Upload YouTube video URLs for accent analysis
- Real-time accent detection using ML models
- Detailed analysis of accent features
- Confidence scoring
- Support for multiple accent categories

## Technologies Used

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- Librosa
- Railway (Deployment)

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## API Endpoints

- `/` - Main interface
- `/analyze` - POST endpoint for accent analysis
- `/health` - Health check endpoint

## License

MIT License
