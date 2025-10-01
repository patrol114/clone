import magic
import uuid
import os
from functools import wraps
from flask import request, flash, redirect, url_for
from pydub import AudioSegment
from config import Config

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_mime_type(file_stream) -> bool:
    try:
        mime = magic.from_buffer(file_stream.read(1024), mime=True)
        file_stream.seek(0)
        return mime in Config.ALLOWED_MIME_TYPES
    except Exception as e:
        # Można dodać logowanie błędu
        return False

def validate_form(*required_fields: str):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method == 'POST':
                data = request.form
                missing = [field for field in required_fields if not data.get(field)]
                if missing:
                    flash(f"Brak wymaganych pól: {', '.join(missing)}.", 'danger')
                    return redirect(request.url)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def split_audio_into_segments(audio_path: str, segment_length: int = 10) -> list:
    """Splits an audio file into segments of specified length in seconds."""
    try:
        audio = AudioSegment.from_file(audio_path)
        total_length = len(audio)  # Duration in milliseconds
        segments = []

        for start_ms in range(0, total_length, segment_length * 1000):
            end_ms = min(start_ms + segment_length * 1000, total_length)
            segment = audio[start_ms:end_ms]
            segment_filename = f"{uuid.uuid4().hex}.wav"
            segment_path = os.path.join(Config.PROCESSED_FOLDER, segment_filename)

            segment.export(segment_path, format="wav")
            segments.append(segment_path)

        return segments
    except Exception as e:
        # Można dodać logowanie błędu
        return []