import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from config import Config

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
cors = CORS()

def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize Flask extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/*": {"origins": Config.ALLOWED_ORIGINS}})

    # Create necessary folders
    for folder in [
        app.config['UPLOAD_FOLDER'],
        app.config['PROCESSED_FOLDER'],
        app.config['GENERATED_FOLDER'],
        app.config['PRETRAINED_FOLDER'],
        app.config['ASR_MODELS_FOLDER']
    ]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Register Blueprints/Routes
    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        from . import models
        from .core.model_loader import initialize_whisper_model
        from .core.asr_brain import get_asr_brain
        import asyncio

        app.config['WHISPER_MODEL'], \
        app.config['WHISPER_PROCESSOR'], \
        app.config['WHISPER_DEVICE'] = initialize_whisper_model()

        app.config['ASR_BRAIN_INSTANCE'] = get_asr_brain(app.config)

        app.config['TRAINING_PROGRESS'] = {}
        app.config['PAUSE_FLAGS'] = {}
        app.config['TRAINING_SEMAPHORE'] = asyncio.Semaphore(1)

    # Configure logging
    if not app.debug and not app.testing:
        # File logger
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/voice_cloning.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Voice Cloning startup')

    return app