import os
from datetime import timedelta

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', '1234567812345678')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///voice_cloning.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', '1234567812345678')
    MAX_CONTENT_LENGTH = 36 * 1024 * 1024  # 36 MB

    JWT_TOKEN_LOCATION = ['cookies']
    JWT_ACCESS_COOKIE_NAME = 'access_token_cookie'
    JWT_COOKIE_CSRF_PROTECT = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=10)

    # Folders
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    GENERATED_FOLDER = os.path.join(BASE_DIR, 'generated')
    PRETRAINED_FOLDER = os.path.join(BASE_DIR, 'pretrained_models')
    ASR_MODELS_FOLDER = os.path.join(BASE_DIR, 'asr_models')

    # Training settings
    NUM_EPOCHS = 100
    SAVE_INTERVAL = 1
    BASE_ASR_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"

    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:5555",
        "http://127.0.0.1:5555",
        "http://192.168.2.71:5555"
    ]

    # Audio processing
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
    ALLOWED_MIME_TYPES = {
        'audio/wav', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/flac',
        'audio/ogg', 'audio/x-ogg'
    }