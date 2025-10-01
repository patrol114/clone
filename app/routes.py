import os
import uuid
import json
import time
import asyncio
from flask import (
    request, jsonify, send_from_directory, render_template, redirect,
    url_for, flash, make_response, stream_with_context, Response, current_app, Blueprint
)
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token, set_access_cookies, unset_jwt_cookies
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

from app import db, jwt
from app.models import User, VoiceProfile, ASRModel
from app.utils import allowed_file, validate_mime_type, validate_form
from app.core.audio_processing import process_audio, augment_audio, process_audio_to_dataset, evaluate_audio_suitability, audio_signal_function
from app.core.model_loader import initialize_whisper_model, transcribe_with_whisper, load_asrtts_model, load_asr_model
from app.core.asr_brain import get_asr_brain, setup_dataio

bp = Blueprint('main', __name__)

executor = ThreadPoolExecutor(max_workers=4)

@bp.route('/')
def home():
    return redirect(url_for('login'))

@bp.route('/register', methods=['GET', 'POST'])
@validate_form('username', 'email', 'password')
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Użytkownik z tym nazwiskiem lub adresem email już istnieje.", 'danger')
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        try:
            db.session.commit()
            flash("Rejestracja zakończona sukcesem. Proszę się zalogować.", 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash("Wystąpił błąd podczas rejestracji. Spróbuj ponownie.", 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@bp.route('/login', methods=['GET', 'POST'])
@validate_form('username_or_email', 'password')
def login():
    if request.method == 'POST':
        data = request.form
        username_or_email = data.get('username_or_email', '').strip()
        password = data.get('password', '').strip()
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email.lower())
        ).first()
        if user and user.check_password(password):
            access_token = create_access_token(identity=user.id)
            response = make_response(redirect(url_for('dashboard')))
            set_access_cookies(response, access_token, max_age=36000)
            flash("Logowanie zakończone sukcesem.", 'success')
            return response
        else:
            flash("Nieprawidłowe dane logowania.", 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@bp.route('/logout')
def logout():
    response = redirect(url_for('login'))
    unset_jwt_cookies(response)
    flash("Zostałeś wylogowany.", 'success')
    return response

@bp.route('/dashboard')
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        flash("Użytkownik nie został znaleziony.", 'danger')
        return redirect(url_for('login'))
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('dashboard.html', username=user.username, current_time=current_time)

@bp.route('/upload_voice', methods=['GET', 'POST'])
@jwt_required()
def upload_voice():
    user_id = get_jwt_identity()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Brak pliku w żądaniu.", 'danger')
            return redirect(request.url)
        file = request.files['file']
        name = request.form.get('name', '').strip() or file.filename
        language = request.form.get('language', 'pl').strip()
        if file.filename == '':
            flash("Nie wybrano pliku.", 'danger')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Nieobsługiwany format pliku audio.", 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}.wav"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            from pydub import AudioSegment
            if filename.rsplit('.', 1)[1].lower() != 'wav':
                audio = AudioSegment.from_file(file)
                audio.export(upload_path, format='wav')
            else:
                file.save(upload_path)
        except Exception as e:
            flash("Nie udało się zapisać lub przekonwertować pliku.", 'danger')
            return redirect(request.url)

        try:
            augment_options = request.form.getlist('augment_options')
            processed_filename = unique_filename
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            process_audio(upload_path, processed_path, augment_options=augment_options)
            transcription = transcribe_with_whisper(
                processed_path,
                current_app.config['WHISPER_MODEL'],
                current_app.config['WHISPER_PROCESSOR'],
                current_app.config['WHISPER_DEVICE']
            )
            voice_profile = VoiceProfile(
                user_id=user_id, name=name, audio_file=processed_filename, transcription=transcription, language=language
            )
            db.session.add(voice_profile)
            db.session.commit()
            flash("Profil głosowy został utworzony, przetworzony i transkrybowany.", 'success')
            return redirect(url_for('analyze_audio', profile_id=voice_profile.id))
        except Exception as e:
            db.session.rollback()
            flash("Wystąpił błąd podczas przetwarzania audio.", 'danger')
            return redirect(request.url)
    return render_template('upload_voice.html')

@bp.route('/profile')
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    profiles = VoiceProfile.query.filter_by(user_id=user_id).all()
    profiles_with_training_status = []
    for profile in profiles:
        asr_trained = ASRModel.query.filter_by(voice_profile_id=profile.id).first() is not None
        profiles_with_training_status.append({
            'profile': profile,
            'asr_trained': asr_trained
        })
    return render_template('profile.html', profiles=profiles_with_training_status)

@bp.route('/train_asr_model/<int:profile_id>', methods=['POST'])
@jwt_required()
def train_asr_model_route(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil głosowy nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    augmented_audio_paths = []
    augmented_transcriptions = []
    try:
        for i in range(3):
            unique_output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"augmented_{uuid.uuid4().hex}.wav")
            augmented_path = augment_audio(audio_path, unique_output_path, augmentation_type="noise")
            if augmented_path:
                augmented_audio_paths.append(augmented_path)
                augmented_transcriptions.append(profile.transcription)
    except Exception as e:
        flash("Błąd podczas augmentacji plików audio.", 'danger')
        return redirect(url_for('profile'))

    all_audio_files = [audio_path] + augmented_audio_paths
    all_transcriptions = [profile.transcription] + augmented_transcriptions

    training_progress = current_app.config['TRAINING_PROGRESS']
    if profile_id in training_progress:
        flash("Trening dla tego profilu jest już w toku.", 'warning')
        return jsonify({"error": "Trening już w toku."}), 400

    training_progress[profile_id] = {"status": "Rozpoczęcie treningu...", "progress": 0}

    def train():
        with app.app_context():
            asyncio.run(train_asr_on_voice_profile(profile_id, all_audio_files, all_transcriptions, current_app._get_current_object()))

    executor.submit(train)
    flash("Trening został rozpoczęty.", 'success')
    return jsonify({"message": "Trening został rozpoczęty."}), 200

async def train_asr_on_voice_profile(profile_id, audio_files, transcriptions, app_context):
    training_semaphore = app_context.config['TRAINING_SEMAPHORE']
    async with training_semaphore:
        try:
            asr_brain_instance = app_context.config['ASR_BRAIN_INSTANCE']
            update_training_progress(profile_id, status="Przygotowywanie danych...", progress=10, app_context=app_context)
            train_dataset, valid_dataset, collate_fn = setup_dataio(asr_brain_instance, audio_files, transcriptions)

            update_training_progress(profile_id, status="Rozpoczynanie treningu...", progress=20, app_context=app_context)
            asr_brain_instance.fit(
                epoch_counter=range(1, app_context.config['NUM_EPOCHS'] + 1),
                train_set=train_dataset,
                valid_set=valid_dataset,
                train_loader_kwargs={"batch_size": 8, "collate_fn": collate_fn},
                valid_loader_kwargs={"batch_size": 8, "collate_fn": collate_fn}
            )

            profile_folder = os.path.join(app_context.config['ASR_MODELS_FOLDER'], f"profile_{profile_id}")
            os.makedirs(profile_folder, exist_ok=True)
            model_path = os.path.join(profile_folder, "pytorch_model.bin")
            import torch
            torch.save(asr_brain_instance.modules['model'].state_dict(), model_path)

            profile = VoiceProfile.query.get(profile_id)
            asr_model_entry = ASRModel(
                user_id=profile.user_id, voice_profile_id=profile.id, name=f"ASR_Model_Profile_{profile.id}",
                model_file="pytorch_model.bin", language=profile.language
            )
            db.session.add(asr_model_entry)
            db.session.commit()
            update_training_progress(profile_id, status="Trening zakończony pomyślnie.", progress=100, app_context=app_context)
        except Exception as e:
            update_training_progress(profile_id, status=f"Błąd: {str(e)}", progress=0, app_context=app_context)
        finally:
            training_progress = app_context.config['TRAINING_PROGRESS']
            if profile_id in training_progress:
                del training_progress[profile_id]

def update_training_progress(profile_id, status=None, progress=None, app_context=None):
    app = app_context or current_app
    training_progress = app.config['TRAINING_PROGRESS']
    if profile_id not in training_progress:
        training_progress[profile_id] = {}
    if status:
        training_progress[profile_id]["status"] = status
    if progress is not None:
        training_progress[profile_id]["progress"] = progress

@bp.route('/training_status/<int:profile_id>')
@jwt_required()
def training_status(profile_id):
    training_progress = current_app.config['TRAINING_PROGRESS']
    progress = training_progress.get(profile_id, {"status": "Nie rozpoczęto", "progress": 0})
    return jsonify(progress)

@bp.route('/tts', methods=['GET', 'POST'])
@jwt_required()
def tts():
    user_id = get_jwt_identity()
    profiles = VoiceProfile.query.filter_by(user_id=user_id).all()
    if request.method == 'POST':
        try:
            text = request.form.get('text', '').strip()
            voice_id = request.form.get('voice_id')
            if not text or not voice_id:
                flash("Wypełnij wszystkie pola.", 'danger')
                return redirect(request.url)

            profile = VoiceProfile.query.filter_by(id=voice_id, user_id=user_id).first()
            if not profile:
                flash("Profil głosowy nie został znaleziony.", 'danger')
                return redirect(request.url)

            model, processor, device = load_asrtts_model(user_id, profile.id, app.config['ASR_MODELS_FOLDER'])

            # This is a placeholder for actual speech generation
            # The original code had a call to generate_speech, but it was async and incomplete
            # For now, we just return a success message
            flash("Generowanie mowy (symulacja).", 'success')
            return redirect(url_for('tts'))

        except Exception as e:
            flash(f"Błąd podczas generowania mowy: {e}", 'danger')
            return redirect(request.url)
    return render_template('tts.html', profiles=profiles)


@bp.route('/analyze_audio/<int:profile_id>')
@jwt_required()
def analyze_audio(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil głosowy nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))
    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))
    try:
        mel_tensor, processing_info, save_status = process_audio_to_dataset(audio_path=audio_path)
        if mel_tensor is None:
            flash("Wystąpił błąd podczas analizy audio.", 'danger')
            return redirect(url_for('profile'))
        suitability = evaluate_audio_suitability(processing_info)
        audio_signal = audio_signal_function(audio_path)
        processing_info['audio_signal'] = audio_signal
        return render_template('audio_analysis.html',
                               profile=profile,
                               processing_info=processing_info,
                               mel_spectrogram=mel_tensor.tolist(),
                               suitability=suitability,
                               save_status=save_status,
                               transcription=profile.transcription)
    except Exception as e:
        flash(f"Wystąpił błąd podczas analizy audio: {str(e)}", 'danger')
        return redirect(url_for('profile'))

@bp.route('/static/generated/<filename>')
def serve_generated_audio(filename: str):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

@bp.route('/static/processed/<filename>')
@jwt_required()
def serve_processed_audio(filename: str):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(audio_file=filename, user_id=user_id).first()
    if not profile:
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@bp.route('/edit_profile/<int:profile_id>', methods=['GET', 'POST'])
@jwt_required()
def edit_profile(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil g³osowy nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    if request.method == 'POST':
        new_name = request.form.get('name', '').strip()
        new_language = request.form.get('language', '').strip()
        if new_name and new_language:
            profile.name = new_name
            profile.language = new_language
            try:
                db.session.commit()
                flash("Profil g³osowy zosta³ zaktualizowany.", 'success')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                flash("Wyst¹pi³ b³¹d podczas aktualizacji profilu.", 'danger')
                return redirect(request.url)
        else:
            flash("Nazwa profilu i jêzyk nie mog¹ byæ puste.", 'danger')
            return redirect(request.url)

    return render_template('edit_profile.html', profile=profile)

@bp.route('/delete_profile/<int:profile_id>', methods=['POST'])
@jwt_required()
def delete_profile(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil g³osowy nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    try:
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)

        asr_model = ASRModel.query.filter_by(voice_profile_id=profile_id).first()
        if asr_model:
            model_path = os.path.join(app.config['ASR_MODELS_FOLDER'], asr_model.model_file)
            if os.path.exists(model_path):
                os.remove(model_path)
            db.session.delete(asr_model)

        db.session.delete(profile)
        db.session.commit()
        flash("Profil g³osowy zosta³ usuniêty.", 'success')
    except Exception as e:
        db.session.rollback()
        flash("Wyst¹pi³ b³¹d podczas usuwania profilu.", 'danger')

    return redirect(url_for('profile'))