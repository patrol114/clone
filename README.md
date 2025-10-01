# Voice Cloning and Text-to-Speech (TTS) Application

This is a Flask-based web application for voice cloning and text-to-speech synthesis. It allows users to register, upload voice samples, train custom Automatic Speech Recognition (ASR) models, and generate speech in the cloned voice.

## Project Structure

The project is structured as a modular Flask application:

- `run.py`: The main entry point to start the application.
- `config.py`: Contains all configuration settings for the application.
- `requirements.txt`: Lists all Python dependencies for the project.
- `app/`: The main application package.
  - `__init__.py`: Initializes the Flask application and its extensions using the factory pattern.
  - `models.py`: Defines the SQLAlchemy database models (User, VoiceProfile, ASRModel).
  - `routes.py`: Contains all the application's routes (endpoints).
  - `core/`: A package for the core business logic.
    - `audio_processing.py`: Functions for audio processing, augmentation, and feature extraction.
    - `asr_brain.py`: The `ASRBrain` class (based on SpeechBrain) for model training.
    - `model_loader.py`: Functions for loading and managing AI models.
  - `utils.py`: Utility functions.
  - `static/`: Static files (CSS, JavaScript, images).
  - `templates/`: HTML templates for the user interface.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` for package installation
- For GPU acceleration (recommended for training):
  - An NVIDIA GPU with CUDA support
  - CUDA Toolkit and cuDNN installed

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Initialize the database:**
    The application uses Flask-Migrate to manage database schemas. The first time you run the app, the tables will be created automatically. For subsequent changes to the models, you'll need to create migrations:
    ```bash
    flask db init  # Only needed once
    flask db migrate -m "Initial migration."
    flask db upgrade
    ```
    *(Note: You might need to set the `FLASK_APP` environment variable: `export FLASK_APP=run.py`)*

2.  **Start the Flask development server:**
    ```bash
    python run.py
    ```

3.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

1.  **Register:** Create a new user account.
2.  **Login:** Log in with your credentials.
3.  **Upload Voice:** Go to the "Upload Voice" page, provide a name for the voice profile, and upload a `.wav` or `.mp3` file.
4.  **Analyze Audio:** After uploading, you will be redirected to an analysis page showing metrics of your audio file.
5.  **Train Model:** From your profile page, you can start the ASR model training for a specific voice profile.
6.  **Synthesize Speech (TTS):** Once a model is trained, go to the "TTS" page to generate speech using your custom voice.