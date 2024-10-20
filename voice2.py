# voice2.py
import warnings
import random
import numpy as np
from pydub import AudioSegment
import librosa
import tempfile
import traceback
import noisereduce as nr
import pynvml
import psutil
# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')
warnings.filterwarnings("ignore", category=FutureWarning, module='speechbrain')
from speechbrain.utils.data_pipeline import takes, provides
from tqdm import tqdm  # Importowanie tqdm
from datetime import datetime, timedelta
import os
from multiprocessing import Pool, cpu_count
import time
import logging
import uuid
import threading
from functools import wraps
from flask import (
    Flask, request, jsonify, send_from_directory, render_template, redirect,
    url_for, flash, make_response, stream_with_context, Response
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_cors import CORS
import soundfile as sf
import torch
# Inicjalizacja semafora z liczb¹ dostêpnych rdzeni CPU
training_semaphore = threading.Semaphore(1)
import gc  # do zarz¹dzania pamiêci¹
from torch.utils.data import DataLoader  # do obs³ugi DataLoadera

# Ustawienia PyTorch dla optymalizacji CPU
torch.set_num_threads(os.cpu_count())  # U¿yj wszystkich rdzeni CPU
torch.backends.cudnn.benchmark = True  # Optymalizuj dla sprzêtu
torch.set_num_interop_threads(os.cpu_count())
import magic
import json
import asyncio

# Importy Wav2Vec2 z Hugging Face
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# U¿ywamy klas z transformers zamiast z SpeechBrain
from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.dataio.batch import PaddedBatch
from torch.amp import autocast  # Updated import
from torch.cuda.amp import GradScaler

from speechbrain.utils.autocast import fwd_default_precision
import speechbrain as sb
from speechbrain.core import Brain
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from concurrent.futures import ThreadPoolExecutor, as_completed
from speechbrain.utils.checkpoints import Checkpointer

from speechbrain.nnet.losses import ctc_loss

import types

asr_brain_instance = None
asr_brain_lock_singleton = threading.Lock()

def get_asr_brain():
    global asr_brain_instance
    with asr_brain_lock_singleton:
        if asr_brain_instance is None:
            logger.info("Ładowanie modelu ASR (Singleton)...")
            model, processor, device = load_asr_model()
            if not model or not processor:
                logger.critical("Nie udało się załadować modelu ASR.")
                raise RuntimeError("Nie udało się załadować modelu ASR.")
            
            # Dodaj inicjalizację Checkpointera
            checkpointer = Checkpointer(
                checkpoints_dir=app.config['ASR_MODELS_FOLDER'],  # Ustaw ścieżkę do zapisu modeli
                recoverables={"model": model, "optimizer": torch.optim.AdamW(model.parameters(), lr=0.001)}  # Dodanie modelu i optymalizatora

            )
            
            asr_brain_instance = ASRBrain(
                modules={"model": model},
                opt_class=lambda params: torch.optim.AdamW(params, lr=0.001),
                hparams={
                    "compute_cost": sb.nnet.losses.ctc_loss,  # Przykładowa funkcja straty
                    "processor": processor,
                    "sample_rate": 16000,
                    "target_sampling_rate": 16000,
                    "use_augmentation": True,
                    "convert_to_mono": "average",
                    "noise_reduction": True,
                    "normalize_audio": True,
                    "blank_index": 0,
                    "max_epochs": 10,
                    "downsample_factor": 320
                },
                run_opts={
                    "device": device.type,
                    "precision": "bf16" if device.type == "cpu" else "fp16"
                },
                checkpointer=checkpointer  # Dodanie Checkpointera do ASRBrain
            )
            logger.info("ASRBrain Singleton został zainicjalizowany.")
        return asr_brain_instance
        
# ------------------- Application Configuration -------------------

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '1234567812345678')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///voice_cloning.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', '1234567812345678')
app.config['MAX_CONTENT_LENGTH'] = 36 * 1024 * 1024  # 36 MB

app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_NAME'] = 'access_token_cookie'
app.config['JWT_COOKIE_CSRF_PROTECT'] = False
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=10)  # Token valid for 10 hours

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.getcwd(), 'processed')
app.config['GENERATED_FOLDER'] = os.path.join(os.getcwd(), 'generated')
app.config['PRETRAINED_FOLDER'] = os.path.join(os.getcwd(), 'pretrained_models')
app.config['ASR_MODELS_FOLDER'] = os.path.join(os.getcwd(), 'asr_models')

# Create folders if they don't exist
for folder in [
    app.config['UPLOAD_FOLDER'],
    app.config['PROCESSED_FOLDER'],
    app.config['GENERATED_FOLDER'],
    app.config['PRETRAINED_FOLDER'],
    app.config['ASR_MODELS_FOLDER']
]:
    if not os.path.exists(folder):
        os.makedirs(folder)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

NUM_EPOCHS = 100
SAVE_INTERVAL = 1

allowed_origins = [
    "http://localhost:5555",
    "http://127.0.0.1:5555",
    "http://192.168.2.71:5555"
]

CORS(app)

# ------------------- Logging Configuration -------------------

import sys
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

# Ustawienie poziomu logowania
app.logger.setLevel(logging.DEBUG)

# Usuniêcie domylnych handlerów
for handler in app.logger.handlers[:]:
    app.logger.removeHandler(handler)

# Handler do logowania do pliku
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
file_handler.setFormatter(file_formatter)

# Handler do logowania w konsoli
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Niestandardowy formatter do kolorowania komunikatów
class CustomFormatter(logging.Formatter):
    """Custom logging formatter with colors for different log levels."""

    # Mapowanie poziomów logowania na kolory
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Pobranie koloru dla danego poziomu logowania
        color = self.LEVEL_COLORS.get(record.levelno, '')
        # Formatowanie wiadomoci z u¿yciem koloru
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

console_formatter = CustomFormatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
console_handler.setFormatter(console_formatter)

# Dodanie handlerów do loggera aplikacji
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
logger = app.logger

# ------------------- Definicje Pipeline DataIO -------------------

# Globalny processor
processor = None

#@takes("audio_path")
#@provides("sig")
def audio_pipeline(audio_path):
    try:
        logger.info(f"Loading audio file: {audio_path}")
        # Load audio with librosa or any other method
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        logger.debug(f"Audio loaded. Sample rate: {sample_rate} Hz, length: {len(audio)} samples")
        if len(audio) == 0:
            logger.warning(f"Loaded audio is empty: {audio_path}")
            empty_signal = torch.zeros(16000, dtype=torch.float32)
            return empty_signal
        waveform = torch.tensor(audio, dtype=torch.float32)
        return waveform
    except Exception as e:
        logger.error(f"Error in audio_pipeline: {e}", exc_info=True)
        raise

#@takes("transcription")
#@provides("tokens_encoded", "tokens_lens")
def token_pipeline(transcription, processor):
    try:
        tokens_encoded = processor.tokenizer(
            transcription,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        tokens_lens = torch.tensor([tokens_encoded.size(0)], dtype=torch.long)
        return tokens_encoded, tokens_lens
    except Exception as e:
        logger.error(f"Error in token_pipeline: {e}", exc_info=True)
        raise
# ------------------- Model Configuration -------------------

BASE_ASR_MODEL_ID = "badrex/xlsr-polish"  # Sta³a zmienna dla modelu bazowego

# ThreadPoolExecutor for handling asynchronous training tasks
executor = ThreadPoolExecutor(max_workers=1)

# Thread lock for thread-safe operations on model progress and cache
asr_model_lock = threading.Lock()

# Cache to hold loaded models
asr_model_cache = {}

# Global progress tracking for model loading
model_loading_progress = {}
asr_model_cache_lock = threading.Lock()

def load_asr_model(model_id=BASE_ASR_MODEL_ID, cache_dir=".cache", use_gradient_checkpointing=True):
    """
    Loads the Wav2Vec2 model and processor from Hugging Face with gradient checkpointing.
    Ensures thread-safe access to the model cache.

    Args:
        model_id (str): Identifier for the Hugging Face model.
        cache_dir (str): Directory to cache the model.
        use_gradient_checkpointing (bool): Whether to enable gradient checkpointing.

    Returns:
        tuple: (model, processor, device) if successful, else (None, None, None).
    """
    global asr_model_cache
    global processor  # Declare processor as global

    with asr_model_lock:
        # Check if the model is already loaded and cached
        if model_id in asr_model_cache:
            logger.info(f"Model '{model_id}' found in cache, reusing...")
            model, processor, device = asr_model_cache[model_id]
            return model, processor, device

    try:
        logger.info(f"Loading ASR model '{model_id}'...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=cache_dir)

        if not model or not processor:
            raise RuntimeError("Failed to load ASR model or processor.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Enable gradient checkpointing if desired (to save memory at the cost of speed)
        model.config.gradient_checkpointing = use_gradient_checkpointing
        logger.info(f"ASR model '{model_id}' loaded successfully on {device}.")

        with asr_model_lock:
            # Cache the model to avoid reloading it
            asr_model_cache[model_id] = (model, processor, device)

        return model, processor, device

    except Exception as e:
        logger.error(f"Error loading ASR model '{model_id}': {e}")
        return None, None, None

from speechbrain.utils.data_pipeline import DataPipeline
from functools import partial

def setup_dataio(asr_brain, voice_profiles):
    """
    Sets up the DataIO pipeline for training and validation datasets using the ASR model's processor.

    Args:
        asr_brain (object): The ASR model instance.
        voice_profiles (list): List of voice profiles containing audio paths and transcriptions.

    Returns:
        tuple: The training and validation datasets.
    """
    try:
        # Przygotowanie danych treningowych i walidacyjnych
        train_data, valid_data = prepare_training_data(voice_profiles)

        # Inicjalizacja datasetów
        datasets = {
            "train": DynamicItemDataset(train_data),
            "valid": DynamicItemDataset(valid_data)
        }

        # Uzyskanie procesora z modelu ASR
        processor = asr_brain.hparams.processor

        # Funkcja czêciowa dla pipeline tokenów z procesorem
        token_pipeline_with_processor = partial(token_pipeline, processor=processor)

        # Dynamiczne dodawanie elementów dla obu zestawów (train i valid)
        for split in ["train", "valid"]:
            datasets[split].add_dynamic_item(
                func=audio_pipeline,
                takes=["audio_path"],
                provides=["sig"]
            )
            datasets[split].add_dynamic_item(
                func=token_pipeline_with_processor,
                takes=["transcription"],
                provides=["tokens_encoded", "tokens_lens"]
            )
            datasets[split].set_output_keys(["id", "sig", "tokens_encoded", "tokens_lens"])

        # Tworzenie pipeline'a dla przetwarzania danych
        pipeline = DataPipeline(
            static_data_keys=["id"],  # Dodano brakuj¹cy argument 'static_data_keys'
            dynamic_items=[
                {"func": audio_pipeline, "takes": "audio_path", "provides": "sig"},
                {"func": token_pipeline_with_processor, "takes": "transcription", "provides": ["tokens_encoded", "tokens_lens"]},
            ],
            output_keys=["id", "sig", "tokens_encoded", "tokens_lens"]
        )

        logger.info("DataIO setup completed successfully.")
        return datasets["train"], datasets["valid"]

    except Exception as e:
        logger.error(f"B³¹d podczas ustawiania DataIO: {e}", exc_info=True)
        raise

def background_system_memory_monitor(interval=240):
    """
    Funkcja monitoruj¹ca zu¿ycie pamiêci RAM, CPU i GPU w tle. Zamyka proces tylko wtedy, gdy zu¿ycie RAM przekroczy 100%.

    Args:
        interval (int): Czas oczekiwania miêdzy kolejnymi pomiarami w sekundach.
    """
    while True:
        try:
            # Monitorowanie pamiêci GPU
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)  # Zu¿ycie GPU w MB
            gpu_total = info_gpu.total // (1024 ** 2)  # Ca³kowita pamiêæ GPU w MB
            gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Nie uda³o siê uzyskaæ informacji o pamiêci GPU: {e}")
            gpu_used, gpu_total, gpu_percent = 0, 0, 0

        try:
            # Monitorowanie pamiêci RAM i CPU
            ram_info = psutil.virtual_memory()
            ram_used = ram_info.used // (1024 ** 2)  # Zu¿ycie RAM w MB
            ram_total = ram_info.total // (1024 ** 2)  # Ca³kowita pamiêæ RAM w MB
            ram_percent = ram_info.percent  # Zu¿ycie RAM w procentach

            cpu_percent = psutil.cpu_percent(interval=1)  # Zu¿ycie CPU w procentach
        except Exception as e:
            logger.warning(f"Nie uda³o siê uzyskaæ informacji o pamiêci RAM/CPU: {e}")
            ram_used, ram_total, ram_percent, cpu_percent = 0, 0, 0, 0

        # Logowanie informacji o systemie
        logger.info(f"Zu¿ycie Systemowe - GPU: {gpu_used} MB / {gpu_total} MB ({gpu_percent:.2f}%) | RAM: {ram_used} MB / {ram_total} MB ({ram_percent}%) | CPU: {cpu_percent}%")

        # Sprawdzenie, czy zu¿ycie RAM osi¹gnê³o 100%
        if ram_percent >= 100:
            logger.critical(f"Zu¿ycie RAM przekroczy³o 100% ({ram_percent}%). Zamykanie aplikacji.")
            os._exit(1)

        # Czekanie przed kolejnym pomiarem
        time.sleep(interval)


# ------------------- Global dictionaries to monitor training progress -------------------
training_progress = {}
pause_flags = {}

# ------------------- Database Models -------------------

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    voice_profiles = db.relationship('VoiceProfile', backref='user', lazy=True)
    asr_models = db.relationship('ASRModel', backref='user', lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class VoiceProfile(db.Model):
    __tablename__ = 'voice_profile'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    audio_file = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())
    transcription = db.Column(db.Text, nullable=True)  # Nowa kolumna na transkrypcjê
    sample_rate = db.Column(db.Integer, nullable=True)  # Dodanie czêstotliwoci próbkowania
    num_samples = db.Column(db.Integer, nullable=True)  # Liczba próbek
    rms_db = db.Column(db.Float, nullable=True)  # RMS w dB
    zcr = db.Column(db.Float, nullable=True)  # Zero-Crossing Rate
    snr_db = db.Column(db.Float, nullable=True)  # Signal-to-Noise Ratio

    asr_model = db.relationship('ASRModel', uselist=False, backref='voice_profile')


class ASRModel(db.Model):
    __tablename__ = 'asr_model'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    voice_profile_id = db.Column(db.Integer, db.ForeignKey('voice_profile.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    model_file = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(50), nullable=True, default='pl')
    created_at = db.Column(db.DateTime, default=db.func.now())

# ------------------- Audio Processing Functions -------------------

def reduce_noise_sample(args):
    """
    Funkcja pomocnicza do redukcji szumów dla pojedynczej próbki audio.

    Args:
        args (tuple): Krotka zawieraj¹ca próbkê audio, jej indeks oraz czêstotliwoæ próbkowania.

    Returns:
        tuple: Indeks próbki oraz przetworzona próbka audio.
    """
    sample, sample_idx, sample_rate = args
    try:
        # Redukcja szumów tylko jeli próbka jest wystarczaj¹co g³ona
        if np.max(np.abs(sample)) > 1e-5:
            max_length = 16000  # 1 sekunda przy próbkowaniu 16kHz
            segments = []

            # Podziel próbkê na segmenty, jeli jej d³ugoæ przekracza max_length
            for j in range(0, len(sample), max_length):
                segment = sample[j:j + max_length]
                reduced_segment = nr.reduce_noise(y=segment, sr=sample_rate)
                segments.append(reduced_segment)

            # Sklej zredukowane segmenty w jedn¹ ca³oæ
            reduced_sample = np.concatenate(segments, axis=0)
            return (sample_idx, reduced_sample)
        else:
            logger.warning(f"Sygna³ zbyt cichy, pomijanie redukcji szumów dla próbki {sample_idx}")
            return (sample_idx, sample)
    except Exception as e:
        logger.error(f"B³¹d podczas redukcji szumów dla próbki {sample_idx}: {e}", exc_info=True)
        return (sample_idx, sample)

def collate_fn(batch):
    """
    Custom collate function to prepare batches for training.
    """
    try:
        inputs = []
        tokens = []
        tokens_lens = []
        for idx, item in enumerate(batch):
            if 'sig' not in item or item['sig'] is None:
                logger.warning(f"Missing or None 'sig' key in batch item at index {idx}. Skipping this item.")
                continue
            inputs.append(item['sig'])
            tokens.append(item['tokens_encoded'])
            tokens_lens.append(item['tokens_lens'])

        if not inputs:
            logger.warning("No valid items in batch to collate.")
            return None  # Return None to indicate an empty batch

        # Pad sequences
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        # Use the tokenizer's padding token ID if available; otherwise, default to 0
        padding_value = 0  # Replace with tokenizer.pad_token_id if using a tokenizer
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=padding_value)
        tokens_lens_tensor = torch.tensor(tokens_lens, dtype=torch.long)
        input_lengths = torch.tensor([inp.size(0) for inp in inputs], dtype=torch.long)

        return {
            'inputs': inputs_padded,
            'tokens_encoded': tokens_padded,
            'tokens_lens': tokens_lens_tensor,
            'input_lengths': input_lengths
        }
    except Exception as e:
        logger.error(f"B³¹d w collate_fn: {e}", exc_info=True)
        return None

import torch.nn.functional as F

# ------------------- Definicja Klasy ASRBrain -------------------
class ASRBrain(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts=None, checkpointer=None, use_amp=True):
        super().__init__(modules, opt_class, hparams, run_opts=run_opts, checkpointer=checkpointer)

        # Store run_opts as an instance attribute
        self.run_opts = run_opts or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modules['model'].to(self.device)
        logger.info(f"ASRBrain urz¹dzenie: {self.device} (Typ: {type(self.device)})")
        self.checkpointer = checkpointer
        if self.checkpointer is None:
            logger.error("Checkpointer nie został prawidłowo zainicjalizowany.")
        
        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)
        self.configure_optimizers()

        self.hparams.batch_size = getattr(hparams, 'batch_size', 8)

        self.enable_gradient_checkpointing = getattr(self.hparams, "enable_gradient_checkpointing", False)
        if self.enable_gradient_checkpointing:
            if hasattr(self.modules['model'], 'config'):
                self.modules['model'].config.gradient_checkpointing = True
                logger.info("Gradient checkpointing w³¹czony.")
            else:
                logger.warning("Model nie posiada atrybutu 'config'. Gradient checkpointing nie mo¿e byæ w³¹czony.")

        self.compile_using_fullgraph = getattr(run_opts, 'compile_using_fullgraph', False)
        self.compile_using_dynamic_shape_tracing = getattr(run_opts, 'compile_using_dynamic_shape_tracing', False)

        if self.compile_using_fullgraph or self.compile_using_dynamic_shape_tracing:
            try:
                self.modules['model'] = torch.compile(
                    self.modules['model'],
                    fullgraph=self.compile_using_fullgraph,
                    dynamic=True
                )
                logger.info("Model skompilowany przy u¿yciu ustawieñ dynamicznych.")
            except Exception as e:
                logger.error(f"B³¹d podczas kompilacji modelu: {e}", exc_info=True)

        # Inicjalizacja GradScaler dla mixed precision
        precision = self.run_opts.get("precision", "fp16")
        if self.device.type == "cuda" and precision == "fp16":
            self.scaler = GradScaler() if use_amp else None
            logger.info("GradScaler zainicjalizowany dla mixed precision.")
        else:
            self.scaler = None
            logger.info("GradScaler nie jest u¿ywany.")

        # Inicjalizacja pynvml dla monitorowania GPU
        self.gpu_initialized = False
        self.init_pynvml()
        self.current_profile_id = None

    def init_pynvml(self):
        """
        Inicjalizuje pynvml i pobiera uchwyt do GPU.
        """
        try:
            if not self.gpu_initialized:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_initialized = True
                logger.info("pynvml zainicjalizowany i uchwyt GPU pobrany.")
        except pynvml.NVMLError as e:
            logger.warning(f"Nie uda³o siê zainicjalizowaæ pynvml: {e}")
            self.handle = None

    def monitor_memory_usage(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)  # Zu¿ycie GPU w MB
            gpu_total = info_gpu.total // (1024 ** 2)  # Ca³kowita pamiêæ GPU w MB
            gpu_percent = (gpu_used / gpu_total) * 100
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Nie uda³o siê uzyskaæ informacji o pamiêci GPU: {e}")
            gpu_used, gpu_total, gpu_percent = 0, 0, 0

        ram_info = psutil.virtual_memory()
        ram_used = ram_info.used // (1024 ** 2)  # Zu¿ycie RAM w MB
        ram_total = ram_info.total // (1024 ** 2)  # Ca³kowita pamiêæ RAM w MB
        ram_percent = ram_info.percent

        cpu_percent = psutil.cpu_percent(interval=1)

        logger.info(f"Zu¿ycie GPU: {gpu_used}/{gpu_total} MB ({gpu_percent:.2f}%)")
        logger.info(f"Zu¿ycie RAM: {ram_used}/{ram_total} MB ({ram_percent}%)")
        logger.info(f"Zu¿ycie CPU: {cpu_percent:.2f}%")

        # Automatyczne czyszczenie pamiêci przy wysokim zu¿yciu
        if ram_percent > 90 or gpu_percent > 90:
            logger.warning("Zu¿ycie pamiêci przekracza 90%, zwalnianie pamiêci...")
            torch.cuda.empty_cache()
            gc.collect()

        return {
            "gpu_used": gpu_used,
            "gpu_total": gpu_total,
            "gpu_percent": gpu_percent,
            "ram_used": ram_used,
            "ram_total": ram_total,
            "ram_percent": ram_percent,
            "cpu_percent": cpu_percent
        }

    def adjust_model_parameters(self, memory_info, max_ram=None, max_gpu=None):
        """
        Dostosowuje parametry modelu na podstawie zu¿ycia pamiêci RAM i GPU.

        Args:
            memory_info (dict): S³ownik ze statystykami pamiêci (wynik monitor_memory_usage).
            max_ram (int): Maksymalne dopuszczalne zu¿ycie RAM w MB. Jeli None, u¿ywa dostêpnej pamiêci RAM.
            max_gpu (int): Maksymalne dopuszczalne zu¿ycie GPU w MB. Jeli None, u¿ywa dostêpnej pamiêci GPU.
        """
        # Ustaw domylne limity, jeli nie zosta³y podane
        if max_ram is None:
            max_ram = memory_info['ram_total'] * 0.9  # Limit 80% dostêpnej pamiêci RAM
        if max_gpu is None:
            max_gpu = memory_info['gpu_total'] * 0.9  # Limit 80% dostêpnej pamiêci GPU

        # Sprawdzenie, czy pamiêæ przekracza limity
        if memory_info['ram_used'] > max_ram or memory_info['gpu_used'] > max_gpu:
            logger.warning("Zu¿ycie pamiêci przekracza limit, zmniejszamy rozmiar batcha.")
            current_batch_size = max(1, getattr(self.hparams, "batch_size", 16) // 2)
            self.hparams["batch_size"] = current_batch_size
            logger.info(f"Nowy rozmiar batcha: {current_batch_size}")

            if getattr(self.hparams, "use_augmentation", False):
                self.hparams["use_augmentation"] = False
                logger.info("Wy³¹czono augmentacjê danych audio.")
        else:
            logger.info("Zu¿ycie pamiêci w normie. Parametry modelu pozostaj¹ bez zmian.")

    def compute_objectives(self, predictions, batch, stage):
        try:
            # Pobranie wartoci docelowych i ich d³ugoci
            tokens_eos = batch["tokens_encoded"].to(self.device)  # [batch, target_length]
            tokens_eos_lens = batch["tokens_lens"].to(self.device)  # [batch]

            # Przekszta³cenie predykcji na logarytmiczne prawdopodobieñstwa
            log_probs = torch.log_softmax(predictions.logits, dim=-1)

            # Przetwarzanie d³ugoci wejæ
            input_lengths = batch["input_lengths"].to(self.device)
            downsample_factor = getattr(self.hparams, "downsample_factor", 1)
            input_lengths = (input_lengths / downsample_factor).long()

            # Konwersja d³ugoci na typ int64
            input_lengths = input_lengths.to(dtype=torch.int64)
            tokens_eos_lens = tokens_eos_lens.to(dtype=torch.int64)

            # Ustawienia indeksu pustego znaku
            blank_index = getattr(self.hparams, "blank_index", 0)

            # Konwersja predykcji na float32 do obliczania CTC loss
            log_probs = log_probs.to(torch.float32)

            # Obliczanie straty CTC
            loss = F.ctc_loss(
                log_probs=log_probs.permute(1, 0, 2),  # [time, batch, num_classes]
                targets=tokens_eos.contiguous().view(-1),  # [batch * target_length]
                input_lengths=input_lengths,  # [batch]
                target_lengths=tokens_eos_lens,  # [batch]
                blank=blank_index,
                reduction='mean'
            )

            return loss

        except KeyError as ke:
            logger.error(f"Brakuj¹cy klucz w batchu podczas compute_objectives: {ke}")
            raise
        except Exception as e:
            logger.error(f"B³¹d w compute_objectives: {e}", exc_info=True)
            raise

    def configure_optimizers(self):
        """
        Konfiguracja optymalizatora dla modelu.
        """
        try:
            if self.opt_class is not None:
                self.optimizer = self.opt_class(self.modules['model'].parameters())
                logger.info("Optymalizator skonfigurowany dla modelu.")
            else:
                logger.error("Brak klasy optymalizatora. Nie mo¿na skonfigurowaæ optymalizatora.")
                raise ValueError("Optimizer class is not provided.")
        except Exception as e:
            logger.error(f"B³¹d podczas konfigurowania optymalizatora: {e}", exc_info=True)
            raise

    def fit(self, epoch_counter, train_set, valid_set=None, progressbar=True, train_loader_kwargs={}, valid_loader_kwargs={}):
        logger.info("Rozpoczęcie trenowania modelu.")
        try:
            train_dataloader = sb.dataio.dataloader.make_dataloader(train_set, **train_loader_kwargs)
            total_steps = len(train_dataloader)
            epoch_pbar = tqdm(total=len(epoch_counter), desc="Epoki treningowe", unit="epoch") if progressbar else None

            for epoch in epoch_counter:
                logger.info(f"Rozpoczęcie epoki {epoch}")
                epoch_loss = 0.0
                batch_pbar = tqdm(total=total_steps, desc=f"Batch epoka {epoch}", unit="batch") if progressbar else None

                for batch in train_dataloader:
                    if batch is None:
                        logger.warning("Otrzymano pusty batch. Pomijanie.")
                        continue
                    loss = self.train_step(batch)
                    epoch_loss += loss.item()
                    if batch_pbar:
                        batch_pbar.update(1)

                if batch_pbar:
                    batch_pbar.close()

                avg_loss = epoch_loss / total_steps
                logger.info(f"Epoka {epoch} zakończona. Średnia strata: {avg_loss:.4f}")

                # Zapisz model po każdej epoce
                if self.checkpointer is not None:
                    if epoch % SAVE_INTERVAL == 0:
                        logger.info(f"Zapisywanie modelu po epoce {epoch}.")
                        self.checkpointer.save_checkpoint()

                if epoch_pbar:
                    epoch_pbar.update(1)

            if epoch_pbar:
                epoch_pbar.close()

            logger.info("Trenowanie modelu zakończone.")
        
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}", exc_info=True)
            raise

    def train_step(self, batch):
        optimizer = self.optimizer
        optimizer.zero_grad()

        try:
            # Wybór precyzji
            precision = self.run_opts.get("precision", "fp16")
            dtype = torch.float16 if precision == "fp16" else torch.float32

            # Zmiana dtype dla batch inputs na odpowiednią precyzję
            batch['inputs'] = batch['inputs'].to(self.device, dtype=dtype, non_blocking=True)

            # Przeprowadzanie operacji forward w trybie autocast dla mieszanej precyzji
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=(precision == "fp16")):
                predictions = self.modules['model'](batch['inputs'])

            # Obliczanie straty
            loss = self.compute_objectives(predictions, batch, stage=sb.Stage.TRAIN)

            # Mixed precision gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            return loss
        except Exception as e:
            logger.error(f"Błąd podczas train_step: {e}", exc_info=True)
            raise

    def adjust_batch_size_based_on_lengths(self, input_lengths, max_memory_usage=14000):
        """
        Dostosowuje rozmiar batcha na podstawie redniej d³ugoci sekwencji i dostêpnej pamiêci.

        Args:
            input_lengths (torch.Tensor): D³ugoci sekwencji w bie¿¹cej partii danych.
            max_memory_usage (int): Maksymalne zu¿ycie pamiêci RAM w MB.
        """
        try:
            avg_length = input_lengths.float().mean().item()
            estimated_memory = avg_length * self.hparams.batch_size * 4 / (1024 ** 2)  # Przybli¿ona pamiêæ w MB

            if estimated_memory > max_memory_usage:
                new_batch_size = max(1, int(self.hparams.batch_size * (max_memory_usage / estimated_memory)))
                logger.warning(f"Przekroczono limit pamiêci. Zmniejszanie batch_size z {self.hparams.batch_size} na {new_batch_size}.")
                self.hparams.batch_size = new_batch_size
            else:
                logger.info("Batch size jest odpowiedni.")
        except Exception as e:
            logger.error(f"B³¹d podczas dostosowywania batch_size: {e}", exc_info=True)

    def compute_forward(self, batch, stage):
        try:
            memory_info = self.monitor_memory_usage()
            self.adjust_model_parameters(memory_info)

            logger.debug("Wejcie do compute_forward...")
            wavs, wav_lens = batch['inputs'], batch['input_lengths']
            logger.debug(f"Kszta³t wavs: {wavs.shape}, wav_lens: {wav_lens}")

            # Konwersja na mono, jeli to konieczne
            if wavs.dim() == 3 and wavs.shape[1] > 1:
                logger.info(f"Wykryto {wavs.shape[1]} kana³ów, konwertowanie na mono...")
                wavs = wavs.mean(dim=1, keepdim=True)
                logger.debug(f"Konwertowano na mono, nowy kszta³t: {wavs.shape}")
            elif wavs.dim() == 2:
                wavs = wavs.unsqueeze(1)
                logger.debug("Dane audio s¹ ju¿ mono, dodano wymiar kana³u.")

            # Padding do minimalnej d³ugoci
            MIN_INPUT_SIZE = 16000
            current_length = wavs.shape[2]
            if current_length < MIN_INPUT_SIZE:
                padding_size = MIN_INPUT_SIZE - current_length
                wavs = torch.nn.functional.pad(wavs, (0, padding_size), "constant", 0)
                logger.info(f"Sygna³ by³ zbyt krótki. Dodano padding, nowy kszta³t: {wavs.shape}")

            # Redukcja szumów
            if getattr(self.hparams, "noise_reduction", False):
                logger.debug("Wykonywanie redukcji szumów...")
                wavs = self.reduce_noise(wavs)

            # Resampling, jeli wymagane
            target_sampling_rate = getattr(self.hparams, "target_sampling_rate", self.hparams.sample_rate)
            if target_sampling_rate != self.hparams.sample_rate:
                logger.debug(f"Resampling z {self.hparams.sample_rate} Hz na {target_sampling_rate} Hz...")
                wavs = self.resample_audio(wavs, orig_sr=self.hparams.sample_rate, target_sr=target_sampling_rate)

            # Normalizacja
            if getattr(self.hparams, "normalize_audio", False):
                logger.debug("Normalizacja sygna³u audio...")
                wavs = self.normalize_audio(wavs, target_rms_db=-40.0)

            # Przygotowanie listy wavów do przetworzenia przez processor
            wavs_list = [wavs[i, 0, :int(wav_lens[i].item())].cpu().numpy() for i in range(wavs.shape[0])]

            # Przetwarzanie za pomoc¹ processor (CPU)
            inputs = self.hparams.processor(
                wavs_list,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding=True
            )

            # Przeniesienie danych na urz¹dzenie
            input_values = inputs.input_values.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
            logger.debug(f"Kszta³t input_values: {input_values.shape}")

            # Mixed Precision and Model Forward Pass
            precision = self.run_opts.get("precision", "fp16")
            autocast_enabled = (precision == "fp16")
            device_type = self.device.type  # 'cuda' or 'cpu'

            with autocast(device_type=device_type, enabled=autocast_enabled):
                logits = self.modules['model'](input_values, attention_mask=attention_mask).logits

            logger.debug(f"Kszta³t logits: {logits.shape}")

            # Aktualizacja input_lengths po downsamplingu
            input_lengths = attention_mask.sum(-1).long()
            logger.debug(f"D³ugoci wejciowe: {input_lengths}")

            batch['input_lengths'] = input_lengths

            return logits

        except Exception as e:
            logger.error(f"B³¹d w compute_forward: {e}", exc_info=True)
            raise

    def resample_audio(self, wavs, orig_sr, target_sr):
        try:
            wavs_np = wavs.squeeze(1).cpu().numpy()

            # Resampling each audio sample to the target sampling rate
            resampled = [librosa.resample(wavs_np[i], orig_sr=orig_sr, target_sr=target_sr) for i in range(wavs_np.shape[0])]

            # Ensure all samples have the same length by padding to the longest one
            max_length = max(len(audio) for audio in resampled)
            resampled_padded = np.array([np.pad(audio, (0, max_length - len(audio)), 'constant') for audio in resampled])

            # Convert back to PyTorch tensor and return
            wavs_resampled = torch.from_numpy(resampled_padded).unsqueeze(1).to(wavs.device)
            return wavs_resampled

        except Exception as e:
            logger.error(f"B³¹d podczas resamplingu: {e}", exc_info=True)
            return wavs  # Return the original wavs in case of error

    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            self.modules['model'].train()
        else:
            self.modules['model'].eval()

        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)
        logger.info(f"Rozpoczêcie etapu: {stage}, Epoka: {epoch}")

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if isinstance(stage_loss, dict):
            avg_loss = stage_loss["loss"]
        else:
            avg_loss = stage_loss  # Jeśli stage_loss jest floatem, użyj go bezpośrednio
    
        if stage == sb.Stage.TRAIN:
            logger.info(f"Epoka: {epoch} | Strata: {avg_loss:.4f}")
        else:
            try:
                wer = self.wer_metric.summarize("WER")
                cer = self.cer_metric.summarize("CER")
                logger.info(f"Zakończenie etapu: {stage}, Strata: {avg_loss:.4f} | WER: {wer:.2f}% | CER: {cer:.2f}%")
    
                # Zapisz checkpoint tylko jeśli CER się poprawia
                current_cer = cer
                best_cer = getattr(self, 'best_cer', float('inf'))
                if current_cer < best_cer:
                    logger.info(f"Poprawa CER z {best_cer:.2f}% na {current_cer:.2f}%. Zapisuję checkpoint.")
                    self.best_cer = current_cer
                    self.checkpointer.save_checkpoint(f"best_epoch_{epoch}")
            except ZeroDivisionError:
                logger.error("ZeroDivisionError podczas obliczania WER/CER: Brak ocenionych zdań.")
                logger.info(f"Zakończenie etapu: {stage}, Strata: {avg_loss:.4f} | WER: N/A | CER: N/A")
            except Exception as e:
                logger.error(f"Błąd podczas obliczania metryk: {e}", exc_info=True)
                logger.info(f"Zakończenie etapu: {stage}, Strata: {avg_loss:.4f} | WER: N/A | CER: N/A")

    @staticmethod
    def normalize_audio(wavs, target_rms_db=None):
        """
        Normalizuje sygna³ audio do okrelonego RMS w decybelach lub do maksymalnej wartoci.
        """
        try:
            if not isinstance(wavs, (torch.Tensor, np.ndarray)):
                raise TypeError("Oczekiwano typu torch.Tensor lub np.ndarray dla wavs.")

            if isinstance(wavs, np.ndarray):
                wavs = torch.from_numpy(wavs).float()

            rms = wavs.pow(2).mean(dim=-1, keepdim=True).sqrt()

            if target_rms_db is not None:
                rms_db = 20 * torch.log10(rms + 1e-9)
                target_rms = 10 ** (target_rms_db / 20)
                scale_factor = target_rms / (rms + 1e-9)
                wavs_normalized = wavs * scale_factor
            else:
                max_val = wavs.abs().max(dim=-1, keepdim=True)[0]
                wavs_normalized = wavs / (max_val + 1e-9)

            return wavs_normalized

        except Exception as e:
            logger.error(f"B³¹d podczas normalizacji audio: {e}")
            return wavs

# ------------------- Helper Functions -------------------

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
ALLOWED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3',
    'audio/flac',
    'audio/ogg', 'audio/x-ogg'
}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_mime_type(file_stream) -> bool:
    try:
        mime = magic.from_buffer(file_stream.read(1024), mime=True)
        file_stream.seek(0)
        logger.info(f"Detected MIME type: {mime}")
        return mime in ALLOWED_MIME_TYPES
    except Exception as e:
        logger.error(f"Error validating MIME type: {e}")
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

def prepare_training_data(voice_profiles, split_ratio=0.8, processed_folder='processed'):
    try:
        if not voice_profiles or not isinstance(voice_profiles, list):
            raise ValueError("Invalid voice profiles data.")

        data = {}
        for idx, profile in enumerate(voice_profiles):
            if not isinstance(profile, VoiceProfile):
                logger.warning(f"Invalid profile at index {idx}. Skipping.")
                continue

            if not profile.audio_file or not profile.transcription:
                logger.warning(f"Incomplete profile ID {profile.id}. Skipping.")
                continue

            audio_path = os.path.join(processed_folder, os.path.basename(profile.audio_file))
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file {audio_path} does not exist. Skipping.")
                continue

            if len(profile.transcription.strip()) == 0:
                logger.warning(f"Empty transcription for profile ID {profile.id}. Skipping.")
                continue

            sample_id = f"sample_{idx}"
            data[sample_id] = {
                'audio_path': audio_path,
                'transcription': profile.transcription.strip()
            }
            logger.debug(f"Added sample: audio_path={audio_path}, transcription_length={len(profile.transcription)}")

        if not data:
            raise ValueError("No valid data for training.")

        # Split into training and validation sets
        split = int(split_ratio * len(data))
        data_items = list(data.items())
        random.shuffle(data_items)  # Shuffle data before splitting
        train_data = dict(data_items[:split])
        valid_data = dict(data_items[split:])

        logger.info(f"Training data size: {len(train_data)}, Validation data size: {len(valid_data)}")

        return train_data, valid_data

    except Exception as e:
        logger.error(f"Error preparing training data: {e}", exc_info=True)
        raise

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
            segment_path = os.path.join("processed", segment_filename)  # Modify path as needed

            # Export segment and append to the list
            segment.export(segment_path, format="wav")
            segments.append(segment_path)

        logger.info(f"Split audio file {audio_path} into {len(segments)} segments.")
        return segments

    except Exception as e:
        logger.error(f"Error splitting audio file into segments: {e}")
        return []

# ------------------- Processing Functions -------------------

def add_reverb(audio_segment, delay_ms=50, decay_db=6, num_echoes=3):
    try:
        reverberated = audio_segment
        for i in range(1, num_echoes + 1):
            delayed = AudioSegment.silent(duration=delay_ms * i) + (audio_segment - (decay_db * i))
            reverberated = reverberated.overlay(delayed)
        return reverberated
    except Exception as e:
        logger.error(f"Error adding reverb: {e}", exc_info=True)
        raise

def augment_audio(audio_path, output_path, augmentation_type=None):
    try:
        audio = AudioSegment.from_file(audio_path)

        if augmentation_type == "noise":
            # Adding background noise
            noise = AudioSegment.silent(duration=len(audio))
            noise = noise.overlay(audio)
            augmented_audio = audio.overlay(noise, gain_during_overlay=-10)

        elif augmentation_type == "speed_pitch":
            # Modifying speed and pitch
            speed_factor = random.uniform(0.9, 1.1)
            pitch_factor = random.uniform(-2, 2)
            augmented_audio = change_speed(audio, speed_factor)
            augmented_audio = change_pitch(augmented_audio, pitch_factor)

        elif augmentation_type == "reverb":
            # Adding reverb
            augmented_audio = add_reverb(audio, delay_ms=50, decay_db=6, num_echoes=3)

        else:
            # Default augmentation if no type is specified
            augmented_audio = audio  # No modification

        # Export audio
        augmented_audio.export(output_path, format="wav")
        logger.info(f"Audio file augmented and saved: {output_path}")

        return output_path
    except Exception as e:
        logger.error(f"Error during audio augmentation: {e}", exc_info=True)
        raise



def change_speed(audio_segment, speed_factor):
    try:
        new_frame_rate = int(audio_segment.frame_rate * speed_factor)
        return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(audio_segment.frame_rate)
    except Exception as e:
        logger.error(f"Error changing speed: {e}")
        raise

def change_pitch(audio, semitones=2):
    """Changes the pitch of the audio by a specified number of semitones."""
    try:
        # Konwersja AudioSegment do NumPy array
        y = np.array(audio.get_array_of_samples()).astype(np.float32)
        y /= np.iinfo(audio.array_type).max  # Normalizacja do zakresu [-1, 1]

        # Zmiana wysokoci tonu za pomoc¹ librosa
        y_shifted = librosa.effects.pitch_shift(y, sr=audio.frame_rate, n_steps=semitones)

        # Konwersja z powrotem do AudioSegment
        y_shifted = np.clip(y_shifted, -1.0, 1.0)
        y_shifted_int16 = (y_shifted * 32767).astype(np.int16)
        shifted_audio = AudioSegment(
            y_shifted_int16.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,  # 16-bit audio
            channels=1
        )

        logger.info(f"Pitch shifted by {semitones} semitones.")
        return shifted_audio
    except Exception as e:
        logger.error(f"Error changing pitch: {e}")
        return audio  # Zwraca oryginalne nagranie w przypadku b³êdu

from pydub.silence import split_on_silence

def strip_silence(audio_segment, silence_thresh=-60.0, min_silence_len=500, keep_silence=100):
    """
    Remove silence from the audio file using Pydub.
    """
    try:
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

        # Combine all non-silent chunks
        combined_audio = AudioSegment.empty()
        for chunk in chunks:
            combined_audio += chunk

        return combined_audio
    except Exception as e:
        logger.error(f"Error stripping silence: {e}")
        raise
    
def process_audio(upload_path: str, processed_path: str,
                  trim_silence: bool = True,
                  augment: bool = False,
                  augment_options: dict = None) -> str:
    """
    Processes an audio file by stripping silence and optionally augmenting it.

    Args:
        upload_path (str): Path to the input audio file.
        processed_path (str): Path to save the processed audio file.
        trim_silence (bool, optional): Whether to strip silence. Defaults to True.
        augment (bool, optional): Whether to augment the audio. Defaults to True.
        augment_options (dict, optional): Additional options for augmentation. Defaults to None.

    Returns:
        str: Path to the processed (and possibly augmented) audio file.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(upload_path)

        # Strip silence if enabled
        if trim_silence:
            audio = strip_silence(audio)

        # Augment audio if enabled
        if augment:
            if augment_options is None:
                augment_options = {"augmentation_type": "noise"}  # Domyślna opcja augmentacji

            augmented_path = augment_audio(upload_path, processed_path, **augment_options)
            return augmented_path
        else:
            # Export the processed (silence-stripped) audio
            audio.export(processed_path, format="wav")
            logger.info(f"Plik audio przetworzony i zapisany: {processed_path}")

            return processed_path
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania pliku audio: {e}", exc_info=True)
        raise

def generate_speech(text: str, profile: VoiceProfile, emotion: str = 'neutral', intonation: float = 1.0) -> str:
    try:
        # Pobranie wpisu ASRModel z bazy danych
        asr_model_entry = ASRModel.query.filter_by(voice_profile_id=profile.id).first()
        if not asr_model_entry:
            logger.error("Nie znaleziono wytrenowanego modelu ASR dla tego profilu.")
            raise ValueError("Wytrenowany model ASR nie jest dostêpny dla tego profilu.")

        # cie¿ka do zapisanego modelu
        model_filename = f"asr_model_profile_{profile.id}_{int(time.time())}.pt"
        model_path = os.path.join(app.config['ASR_MODELS_FOLDER'], model_filename)
                
        if not os.path.exists(model_path):
            logger.error(f"Plik modelu ASR nie zosta³ znaleziony: {model_path}")
            raise FileNotFoundError(f"Plik modelu ASR nie zosta³ znaleziony: {model_path}")

        # Sprawdzenie w cache
        with asr_model_cache_lock:
            if model_path in asr_model_cache:
                loaded_asr_brain = asr_model_cache[model_path]
                logger.info(f"Za³adowany model ASR z cache: {model_path}")
            else:
                loaded_asr_brain = ASRBrain.load_from_checkpoint(checkpoint_path=model_path, run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
                asr_model_cache[model_path] = loaded_asr_brain
                logger.info(f"Wytrenowany model ASR za³adowany z {model_path} i dodany do cache")

        # Przyk³ad u¿ycia oryginalnych modeli TTS (Tacotron2 i HIFIGAN)
        if not Tacotron2 or not HIFIGAN:
            logger.error("Modele TTS nie zosta³y za³adowane.")
            raise ValueError("Modele TTS s¹ niedostêpne.")

        # Mo¿esz tutaj dodaæ logikê integracji ASR z TTS, np. personalizacja g³osu
        mel_output, mel_length, alignment = Tacotron2.encode_text(text, emotion=emotion, intonation=intonation)
        waveforms = HIFIGAN.decode_batch(mel_output)
        audio_data = waveforms.squeeze().cpu().numpy()

        output_filename = f"generated_{uuid.uuid4().hex}.wav"
        generated_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)
        sf.write(generated_path, audio_data, 22050)

        logger.info(f"Mowa zosta³a wygenerowana: {generated_path}")
        return output_filename

    except Exception as e:
        logger.error(f"B³¹d podczas generowania mowy: {e}")
        raise

def save_transcription_result(transcription: str, save_path: str):
    """Zapisuje wynik transkrypcji do pliku."""
    try:
        with open(save_path, 'w') as f:
            f.write(transcription)
        logger.info(f"Transcription result saved: {save_path}")
    except Exception as e:
        logger.error(f"Error saving transcription result: {e}")

def fine_tune_asr(train_data, valid_data, user_id, language, app_config, profile_id,
                  num_epochs=10, batch_size=8, learning_rate=0.001):
    """
    Fine-tunes the Automatic Speech Recognition (ASR) model based on the user's voice input.
    """
    global asr_brain_instance

    try:
        logger.info(f"Starting fine-tuning for profile {profile_id} (User: {user_id}, Language: {language})")

        if not train_data or not valid_data:
            raise ValueError("No training or validation data.")

        if asr_brain_instance is None:
            raise RuntimeError("ASR model is not initialized. Load the ASR model before fine-tuning.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler() if torch.cuda.is_available() else None

        # Update model with new hyperparameters
        asr_brain_instance.hparams.lr = learning_rate
        asr_brain_instance.hparams.device = device
        asr_brain_instance.scaler = scaler if scaler else None

        # Setup optimizer
        asr_brain_instance.optimizer = torch.optim.AdamW(asr_brain_instance.modules['model'].parameters(), lr=learning_rate)

        # Move model to the appropriate device
        for mod in asr_brain_instance.modules.values():
            mod.to(device)

        # Prepare data loader arguments
        train_loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 2,
            "collate_fn": PaddedBatch,
            "pin_memory": device.type == "cuda"
        }
        valid_loader_kwargs = train_loader_kwargs.copy()

        logger.info("Starting ASR model training...")
        asr_brain_instance.fit(
            epoch_counter=range(1, num_epochs + 1),
            train_set=train_data,
            valid_set=valid_data,
            train_loader_kwargs=train_loader_kwargs,
            valid_loader_kwargs=valid_loader_kwargs
        )

        logger.info(f"ASR training completed for profile {profile_id}.")
        return "Training completed successfully."

    except Exception as e:
        logger.error(f"Error during ASR training for profile {profile_id}: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}")

def train_asr_on_voice_profile(profile, app_config, audio_files, transcriptions,
                               batch_size=8, num_epochs=10, num_workers=2):
    """
    Trains the ASR model on a voice profile using provided audio files and transcriptions.
    """
    global asr_brain_instance

    try:
        # Validate input data
        if not audio_files or not transcriptions:
            raise ValueError("Audio files and transcriptions lists cannot be empty.")
        if len(audio_files) != len(transcriptions):
            raise ValueError("The number of audio files and transcriptions must be the same.")

        logger.info(f"Starting ASR training for profile: {profile.name}")

        # Ensure the ASR model is loaded
        if asr_brain_instance is None:
            raise RuntimeError("ASR model is not initialized. Load the ASR model before training.")

        # Ustawienie current_profile_id w ASRBrain
        asr_brain_instance.current_profile_id = profile.id
        logger.debug(f"Ustawiono 'current_profile_id' na: {profile.id}")

        # Preprocess audio files (without augmentation)
        preprocessed_audio_files = []
        for audio_file in audio_files:
            try:
                processed_file = process_audio(
                    audio_file,
                    processed_path=os.path.join(app_config['PROCESSED_FOLDER'], f"{uuid.uuid4().hex}.wav"),
                    trim_silence=True
                )
                preprocessed_audio_files.append(processed_file)
            except Exception as e:
                logger.error(f"Failed to process audio file {audio_file}: {e}", exc_info=True)
                continue  # Skip errors during individual audio file processing

        if not preprocessed_audio_files:
            raise ValueError("No successfully preprocessed audio files.")

        # Create VoiceProfile objects
        voice_profiles = []
        for audio_file, transcription in zip(preprocessed_audio_files, transcriptions):
            try:
                profile_obj = VoiceProfile(audio_file=audio_file, transcription=transcription)
                voice_profiles.append(profile_obj)
                logger.debug(f"Created VoiceProfile: audio_file={audio_file}, transcription_length={len(transcription)}")
            except Exception as e:
                logger.error(f"Error creating VoiceProfile object for file {audio_file}: {e}", exc_info=True)

        if not voice_profiles:
            raise ValueError("Voice profiles list is empty or contains invalid data.")

        # Configure DataIO
        try:
            train_dataset, valid_dataset = setup_dataio(asr_brain_instance, voice_profiles)
            if not train_dataset or not valid_dataset:
                raise ValueError("Failed to configure DataIO.")
        except Exception as e:
            logger.error(f"Error configuring DataIO: {e}", exc_info=True)
            raise RuntimeError("Failed to configure DataIO.")

        logger.info(f"Starting model training for {num_epochs} epochs.")

        # Configure device (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        asr_brain_instance.hparams.device = device
        for mod in asr_brain_instance.modules.values():
            mod.to(device)

        # Initialize scaler for mixed precision, if enabled
        if torch.cuda.is_available():
            if not hasattr(asr_brain_instance, 'scaler') or asr_brain_instance.scaler is None:
                asr_brain_instance.scaler = torch.cuda.amp.GradScaler()

        logger.info(f"ASR model running on device: {device}")

        # Ustawienie liczby epok w profilu na podstawie modelu
        profile.epochs = asr_brain_instance.hparams.max_epochs = num_epochs
        logger.info(f"Liczba epok dla profilu {profile.id} ustawiona na {num_epochs}")

        # Train the model
        try:
            asr_brain_instance.fit(
                epoch_counter=range(1, num_epochs + 1),
                train_set=train_dataset,
                valid_set=valid_dataset,
                train_loader_kwargs={
                    "batch_size": batch_size,
                    "collate_fn": collate_fn,
                    "num_workers": num_workers,
                    "pin_memory": device.type == "cuda"
                },
                valid_loader_kwargs={
                    "batch_size": batch_size,
                    "collate_fn": collate_fn,
                    "num_workers": num_workers,
                    "pin_memory": device.type == "cuda"
                }
            )
            logger.info(f"ASR training completed successfully for profile: {profile.name}")
        except Exception as e:
            logger.error(f"Error during ASR model training: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}")

        # Save the trained model
        with app.app_context():
            try:
                model_filename = f"CKPT+asr_model_profile_{profile.id}_{int(time.time())}.pt"
                model_path = os.path.join(app_config['ASR_MODELS_FOLDER'], model_filename)
                torch.save(asr_brain_instance.modules['model'].state_dict(), model_path)
                asr_brain_instance.checkpointer.save_checkpoint(name=model_filename)  # Zapisz checkpoint
                
                logger.info(f"Trenowany model ASR zapisany jako {model_path}")

                # Add entry to the database for ASRModel
                asr_model_entry = ASRModel(
                    user_id=profile.user_id,
                    voice_profile_id=profile.id,
                    name=f"ASR_Model_Profile_{profile.id}",
                    model_file=model_filename,
                    language=profile.language
                )
                db.session.add(asr_model_entry)
                db.session.commit()
                logger.info(f"Wpis ASRModel utworzony w bazie danych dla profilu {profile.id}")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error saving ASR model: {e}", exc_info=True)
                raise RuntimeError(f"Saving model failed: {str(e)}")

        return "Training completed successfully."

    except Exception as e:
        logger.error(f"Error during ASR training for profile {profile.name}: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}")

    
def evaluate_audio_suitability(processing_info,
                               min_duration_sec=5.0,
                               max_duration_sec=400.0,
                               min_snr_db=10.0,
                               max_snr_db=50.0,
                               max_zcr=0.1,
                               min_rms_db=-40.0):
    """Evaluates whether the audio is suitable for ASR training based on specified criteria."""
    try:
        duration = float(processing_info.get("duration_sec", 0))
        mel_tensor_shape = processing_info.get("mel_tensor_shape", [0, 0, 0])
        snr_db = float(processing_info.get("snr_db", 0))
        clipping_detected = processing_info.get("clipping_detected", False)
        rms_db = float(processing_info.get("rms_db", 0))
        zcr = float(processing_info.get("zcr", 0))
        reasons = []
        is_suitable = True

        if duration < min_duration_sec:
            is_suitable = False
            reasons.append(f"Minimalny czas trwania nagrania to {min_duration_sec} sekund. Twoje nagranie ma tylko {duration:.2f} sekund.")
        if duration > max_duration_sec:
            is_suitable = False
            reasons.append(f"Maksymalny czas trwania nagrania to {max_duration_sec} sekund. Twoje nagranie ma {duration:.2f} sekund.")
        if mel_tensor_shape[1] < 40:
            is_suitable = False
            reasons.append(f"Zbyt ma³a liczba Mel Bands ({mel_tensor_shape[1]}). Powinno byæ przynajmniej 40.")
        if snr_db < min_snr_db:
            is_suitable = False
            reasons.append(f"Stosunek sygna³u do szumu (SNR) jest za niski: {snr_db} dB. Minimalny wymagany SNR to {min_snr_db} dB.")
        elif snr_db > max_snr_db:
            is_suitable = False
            reasons.append(f"Stosunek sygna³u do szumu (SNR) jest za wysoki: {snr_db} dB. Maksymalny dozwolony SNR to {max_snr_db} dB.")
        if clipping_detected:
            is_suitable = False
            reasons.append("Nagranie zawiera clipping, co mo¿e wp³ywaæ negatywnie na jakoæ treningu.")
        if zcr > max_zcr:
            is_suitable = False
            reasons.append(f"Zero-Crossing Rate (ZCR) jest za wysoki: {zcr}. Maksymalny dozwolony ZCR to {max_zcr}.")
        if rms_db < min_rms_db:
            is_suitable = False
            reasons.append(f"rednia g³onoæ (RMS Energy) jest za niska: {rms_db} dB. Minimalny wymagany poziom to {min_rms_db} dB.")
        if is_suitable:
            return {
                "is_suitable": True,
                "reason": "Nagranie spe³nia wszystkie wymagane kryteria."
            }
        else:
            return {
                "is_suitable": False,
                "reason": " ".join(reasons)
            }
    except Exception as e:
        logger.error(f"Error evaluating audio suitability: {e}")
        return {
            "is_suitable": False,
            "reason": f"Nie uda³o siê oceniæ przydatnoci nagrania: {str(e)}"
        }


def create_dataloaders(train_set, valid_set, batch_size: int = 8, num_workers: int = 2):
    """
    Creates DataLoader instances for training and validation datasets.

    Args:
        train_set (Dataset): Training dataset.
        valid_set (Dataset): Validation dataset.
        batch_size (int, optional): Number of samples in a batch. Default is 8.
        num_workers (int, optional): Number of worker processes for data loading. Default is 4.

    Returns:
        tuple: A tuple containing (train_loader, valid_loader) or (None, None) in case of an error.
    """
    try:
        logger.info("Creating DataLoaders for training and validation sets.")

        # Training DataLoader
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Optimization for GPU
        )
        logger.info(f"Training DataLoader created: batch_size={batch_size}, num_workers={num_workers}")

        # Validation DataLoader
        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Optimization for GPU
        )
        logger.info(f"Validation DataLoader created: batch_size={batch_size}, num_workers={num_workers}")

        return train_loader, valid_loader

    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}", exc_info=True)
        return None, None


def correct_clipping(audio, threshold=0.99, max_corrections=5):
    """
    Korekuje clipping w nagraniu audio poprzez skalowanie sygna³u.
    """
    try:
        for i in range(max_corrections):
            if np.any(np.abs(audio) > threshold):
                max_val = np.max(np.abs(audio))
                scale = threshold / max_val
                audio = audio * scale
                logger.warning(f"Clipping zosta³ wykryty i skorygowany. Iteracja {i+1}/{max_corrections}.")
            else:
                break
        clipping_detected = detect_clipping(audio, threshold)
        if clipping_detected:
            logger.warning("Clipping nadal wystêpuje po korekcji.")
        return audio
    except Exception as e:
        logger.error(f"Error correcting clipping: {e}")
        return audio

def apply_low_pass_filter(audio, cutoff=3000, sample_rate=16000):
    """
    Stosuje filtr dolnoprzepustowy w celu zmniejszenia iloci przejæ przez zero (ZCR).
    """
    try:
        from scipy.signal import butter, lfilter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        filtered_audio = lfilter(b, a, audio)
        logger.info("Filtr dolnoprzepustowy zosta³ zastosowany w celu redukcji ZCR.")
        return filtered_audio
    except Exception as e:
        logger.error(f"Error applying low-pass filter: {e}")
        return audio

def reduce_noise_audio(audio, sample_rate):
    """
    Redukuje szumy w nagraniu audio.
    """
    try:
        logger.debug("Rozpoczynanie redukcji szumów.")
        # Automatyczna detekcja szumu na pocz¹tku nagrania
        noisy_part = audio[:int(0.5 * sample_rate)]  # Pierwsze 0.5 sekundy
        reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noisy_part, prop_decrease=1.0)
        logger.debug("Redukcja szumów zakoñczona.")
        return reduced_noise
    except Exception as e:
        logger.error(f"Error reducing noise: {e}", exc_info=True)
        return audio  # Zwraca oryginalne nagranie w przypadku b³êdu


def ensure_min_duration(audio_segment, min_duration_sec=5.0):
    """
    Upewnia siê, ¿e nagranie ma minimaln¹ d³ugoæ poprzez dodanie ciszy.
    """
    try:
        current_duration_sec = len(audio_segment) / 1000.0
        if current_duration_sec >= min_duration_sec:
            return audio_segment
        else:
            required_duration_ms = int((min_duration_sec - current_duration_sec) * 1000)
            silence = AudioSegment.silent(duration=required_duration_ms)
            return audio_segment + silence
    except Exception as e:
        logger.error(f"Error ensuring minimum duration: {e}", exc_info=True)
        raise

from scipy.signal import savgol_filter


def smooth_audio(audio, window_length=101, polyorder=2):
    """
    Wyg³adza sygna³ audio za pomoc¹ filtru Savitzky-Golay w celu redukcji przejæ przez zero (ZCR).

    Args:
        audio (np.array): Sygna³ audio do wyg³adzenia.
        window_length (int): D³ugoæ okna filtru. Musi byæ nieparzysta.
        polyorder (int): Rz¹d wielomianu u¿ywany do aproksymacji w filtrze.

    Returns:
        np.array: Wyg³adzony sygna³ audio.
    """
    try:
        # Upewnij siê, ¿e window_length jest mniejsze ni¿ d³ugoæ sygna³u i nieparzyste
        if window_length >= len(audio):
            window_length = len(audio) // 2 * 2 + 1  # Ustaw na najbli¿sz¹ mniejsz¹ wartoæ nieparzyst¹
        if window_length % 2 == 0:
            window_length += 1

        # Zastosowanie filtru Savitzky-Golay do wyg³adzenia sygna³u
        smoothed_audio = savgol_filter(audio, window_length=window_length, polyorder=polyorder)

        return smoothed_audio
    except Exception as e:
        logger.error(f"B³¹d podczas wyg³adzania audio: {e}", exc_info=True)
        return audio  # W przypadku b³êdu zwróæ oryginalny sygna³

def process_audio_to_dataset(audio_path: str, n_mels=80, n_fft=1024, hop_length=256, max_duration_sec=400.0, save_dir="processed_data", max_iterations=3):
    """
    Przetwarza plik audio, oblicza spektrogram Mel i przygotowuje dane do treningu modelu ASR.

    Args:
        audio_path (str): cie¿ka do pliku audio.
        n_mels (int): Liczba pasm Mel w spektrogramie.
        n_fft (int): Wielkoæ FFT.
        hop_length (int): D³ugoæ kroku miêdzy oknami FFT.
        max_duration_sec (float): Maksymalny czas trwania audio (w sekundach).
        save_dir (str): Katalog, w którym zapisane zostan¹ przetworzone dane.
        max_iterations (int): Maksymalna liczba iteracji przetwarzania w celu korekcji jakoci dwiêku.

    Returns:
        tuple: Tensor spektrogramu Mel, informacje o przetwarzaniu, status zapisania.
    """
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Audio path does not exist: {audio_path}")
            return None, {"error": "Plik audio nie istnieje."}, {"success": False, "error": "Plik audio nie istnieje."}

        logger.info(f"Loading audio file: {audio_path}")
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)

        if len(audio) == 0:
            logger.error("Loaded audio is empty.")
            return None, {"error": "Za³adowane audio jest puste."}, {"success": False, "error": "Za³adowane audio jest puste."}

        duration_sec = len(audio) / sample_rate
        if duration_sec < 5.0:
            logger.warning(f"Audio duration {duration_sec} sekund jest poni¿ej minimalnej wartoci (5.0 sekund). Dodajê ciszê.")
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.dtype.itemsize,
                channels=1
            )
            audio_segment = ensure_min_duration(audio_segment, min_duration_sec=5.0)
            audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2**15)
            duration_sec = len(audio) / sample_rate
            logger.info(f"Audio po dodaniu ciszy ma {duration_sec} sekund.")

        # Iteracyjna korekta audio
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Iteration {iteration} of audio processing.")
            audio = reduce_noise_audio(audio, sample_rate)
            audio = correct_clipping(audio, threshold=0.99)
            audio = smooth_audio(audio)

            additional_metrics = compute_additional_metrics(audio, sample_rate)
            rms_db = additional_metrics["rms_db"]
            zcr = additional_metrics["zcr"]

            clipping_detected = detect_clipping(audio)
            logger.debug(f"Iteration {iteration} metrics: RMS={rms_db} dB, ZCR={zcr}, Clipping Detected={clipping_detected}")

            if not clipping_detected and zcr <= 0.1:
                logger.info("Audio meets the quality criteria.")
                break
            else:
                logger.info("Audio does not meet the quality criteria. Applying further corrections.")
                if zcr > 0.1:
                    audio = smooth_audio(audio)
                if clipping_detected:
                    audio = correct_clipping(audio, threshold=0.99)

        processing_info = {
            "duration_sec": round(len(audio) / sample_rate, 2),
            "sample_rate": sample_rate,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "mel_tensor_shape": None,
            "snr_db": None,
            "clipping_detected": clipping_detected,
            "rms_db": round(rms_db, 2),
            "zcr": round(zcr, 4)
        }

        snr = compute_snr(audio, sample_rate)
        processing_info["snr_db"] = round(snr, 2)

        additional_metrics = compute_additional_metrics(audio, sample_rate)
        processing_info["rms_db"] = round(additional_metrics["rms_db"], 2)
        processing_info["zcr"] = round(additional_metrics["zcr"], 4)

        clipping_detected = detect_clipping(audio)
        processing_info["clipping_detected"] = clipping_detected

        if clipping_detected:
            logger.warning("Clipping nadal wystêpuje po korekcji.")

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        mel_tensor = torch.FloatTensor(mel_spectrogram_normalized).unsqueeze(0)
        signal_tensor = torch.tensor(audio, dtype=torch.float32)
        processing_info["mel_tensor_shape"] = mel_tensor.shape

        logger.info(f"Final processing info: {processing_info}")

        if mel_tensor.shape[1] < 40:
            logger.warning(f"Liczba Mel Bands ({mel_tensor.shape[1]}) jest za niska. Próbujê przeliczyæ mel spektrogram.")
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=40,
                n_fft=n_fft,
                hop_length=hop_length,
                power=2.0
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
            mel_tensor = torch.FloatTensor(mel_spectrogram_normalized).unsqueeze(0)
            processing_info["n_mels"] = 40
            processing_info["mel_tensor_shape"] = mel_tensor.shape

        save_status = save_processed_data(mel_tensor, signal_tensor, processing_info, save_dir)
        return mel_tensor, processing_info, save_status
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {str(e)}")
        return None, {"error": str(e)}, {"success": False, "error": str(e)}


def save_processed_data(mel_tensor, signal_tensor, processing_info, save_dir):
    """Saves processed data to a file."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"processed_audio_{int(time.time())}.pt"
        file_path = os.path.join(save_dir, file_name)
        torch.save({
            'mel_spectrogram': mel_tensor,
            'audio_signal': signal_tensor,
            'processing_info': processing_info
        }, file_path)
        logger.info(f"Processed data saved to {file_path}")
        return {"success": True, "path": file_path}
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        return {"success": False, "error": str(e)}

def compute_snr(audio, sample_rate, top_db=30):
    """Oblicza stosunek sygna³u do szumu (SNR) nagrania audio."""
    try:
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) == 0:
            logger.warning("Nie wykryto mowy w nagraniu.")
            return 0.0
        signal_power = np.sum([
            np.sum(audio[start:end] ** 2) for start, end in non_silent_intervals
        ])

        if signal_power == 0:
            logger.warning("Moc sygna³u wynosi zero po usuniêciu ciszy.")
            return 0.0

        noise_intervals = []
        prev_end = 0
        for start, end in non_silent_intervals:
            if prev_end < start:
                noise_intervals.append((prev_end, start))
            prev_end = end
        if prev_end < len(audio):
            noise_intervals.append((prev_end, len(audio)))
        noise_power = np.sum([
            np.sum(audio[start:end] ** 2) for start, end in noise_intervals
        ])
        logger.debug(f"Signal power: {signal_power}, Noise power: {noise_power}")
        if noise_power < 1e-10:
            logger.warning("Nie wykryto szumu w nagraniu. Ustawianie SNR na nieskoñczonoæ.")
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    except Exception as e:
        logger.error(f"B³¹d podczas obliczania SNR: {e}", exc_info=True)
        return 0.0

def detect_clipping(audio, threshold=0.99):
    """Detects clipping in an audio recording."""
    try:
        clipping = np.any(np.abs(audio) > threshold)
        if clipping:
            logger.warning("Clipping detected in the audio.")
        return clipping
    except Exception as e:
        logger.error(f"Error detecting clipping: {e}")
        return False

def compute_additional_metrics(audio, sample_rate):
    """Computes additional metrics for audio quality."""
    try:
        if len(audio) == 0:
            logger.warning("Audio signal is empty. Setting default metrics.")
            return {
                "rms_db": 0.0,
                "zcr": 0.0
            }

        rms_energy = np.sqrt(np.mean(audio ** 2))
        rms_db = librosa.amplitude_to_db(np.array([rms_energy]))[0]
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        return {
            "rms_db": round(rms_db, 2),
            "zcr": round(zcr, 4)
        }
    except Exception as e:
        logger.error(f"Error computing additional metrics: {e}")
        return {
            "rms_db": 0.0,
            "zcr": 0.0
        }

def transcribe_with_whisper(audio_path: str) -> str:
    # Load the processor and model with specified language and task
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="pl",
        task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set forced decoder IDs for language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="pl",
        task="transcribe"
    )

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Process audio to get input features and attention mask
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True  # Ensure attention mask is returned
        )
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate transcription with attention mask
        predicted_ids = model.generate(
            input_features=input_features,  # Use 'input_features' instead of 'inputs'
            attention_mask=attention_mask,  # Pass the attention mask
            forced_decoder_ids=model.config.forced_decoder_ids
        )

        # Decode predictions, skipping special tokens
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return transcription.strip()
    except Exception as e:
        logger.error(f"B³¹d podczas transkrypcji za pomoc¹ Whisper: {e}", exc_info=True)
        return ""

from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.lobes.models.flair.embeddings import FlairEmbeddings

def evaluate_metrics(model: ASRBrain, valid_data: DynamicItemDataset, profile_id: int, app_config: dict):
    """
    Evaluates the fine-tuned ASR model using various metrics.

    Args:
        model (ASRBrain): The ASRBrain instance.
        valid_data (DynamicItemDataset): The validation dataset.
        profile_id (int): ID of the voice profile.
        app_config (dict): Application configuration.
    """
    try:
        logger.info(f"Evaluating metrics for profile {profile_id}...")

        # Prepare references and hypotheses
        refs = []
        hyps = []

        for batch in valid_data:
            logits, input_lengths = model.compute_forward(batch, stage='valid')
            predictions = torch.argmax(logits, dim=-1)
            transcripts = model.hparams.processor.batch_decode(predictions)
            target_transcripts = model.hparams.processor.batch_decode(batch.tokens_encoded)

            refs.extend([ref.lower().split() for ref in target_transcripts])
            hyps.extend([hyp.lower().split() for hyp in transcripts])

        # Compute WER
        wer_stats = ErrorRateStats()
        wer_stats.append(ids=list(range(len(refs))), predict=hyps, target=refs)
        wer_result = wer_stats.summarize()

        # Compute CER
        cer_stats = ErrorRateStats(split_tokens=True)
        cer_stats.append(ids=list(range(len(refs))), predict=hyps, target=refs)
        cer_result = cer_stats.summarize()

        # Log Metrics
        logger.info(f"Metrics for profile {profile_id}:")
        logger.info(f"WER: {wer_result['WER']:.2f}%")
        logger.info(f"CER: {cer_result['WER']:.2f}%")

        # Optionally, store metrics in the database or another storage system
        asr_model_entry = ASRModel.query.filter_by(voice_profile_id=profile_id).first()
        if asr_model_entry:
            asr_model_entry.metrics = json.dumps({
                "WER": wer_result['WER'],
                "CER": cer_result['WER'],
            })
            db.session.commit()
            logger.info(f"Metrics saved to database for ASR model of profile {profile_id}.")

    except Exception as e:
        logger.error(f"Error during metrics evaluation for profile {profile_id}: {e}", exc_info=True)
        raise e

# ------------------- API Routes -------------------

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
@validate_form('username', 'email', 'password')
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("U¿ytkownik z tym nazwiskiem lub adresem email ju¿ istnieje.", 'danger')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        try:
            db.session.commit()
            logger.info(f"New user registered: {username}")
            flash("Rejestracja zakoñczona sukcesem. Proszê siê zalogowaæ.", 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during user registration: {e}")
            flash("Wyst¹pi³ b³¹d podczas rejestracji. Spróbuj ponownie.", 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
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
            logger.info(f"User logged in: {user.username}")
            flash("Logowanie zakoñczone sukcesem.", 'success')
            return response
        else:
            flash("Nieprawid³owe dane logowania.", 'danger')
            logger.warning(f"Failed login attempt for: {username_or_email}")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    unset_jwt_cookies(response)
    flash("Zosta³e wylogowany.", 'success')
    return response

@app.route('/dashboard')
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        flash("U¿ytkownik nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('login'))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('dashboard.html', username=user.username, current_time=current_time)

@app.route('/upload_voice', methods=['GET', 'POST'])
@jwt_required()
def upload_voice():
    user_id = get_jwt_identity()
    if request.method == 'POST':
        # Sprawdzenie, czy plik jest w żądaniu
        if 'file' not in request.files:
            message = "Brak pliku w żądaniu."
            logger.warning(message)
            if request.is_json:
                return jsonify({"success": False, "message": message}), 400
            flash(message, 'danger')
            return redirect(request.url)

        file = request.files['file']
        name = request.form.get('name', '').strip() or file.filename
        language = request.form.get('language', 'pl').strip()

        if file.filename == '':
            message = "Nie wybrano pliku."
            logger.warning(message)
            if request.is_json:
                return jsonify({"success": False, "message": message}), 400
            flash(message, 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            message = "Nieobsługiwany format pliku audio."
            logger.warning(message)
            if request.is_json:
                return jsonify({"success": False, "message": message}), 400
            flash(message, 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}.wav"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            if filename.rsplit('.', 1)[1].lower() != 'wav':
                audio = AudioSegment.from_file(file)
                audio.export(upload_path, format='wav')
                logger.info(f"Plik audio przekonwertowany do WAV: {upload_path}")
            else:
                file.save(upload_path)
                logger.info(f"Plik audio zapisany: {upload_path}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania lub konwertowania pliku: {e}", exc_info=True)
            message = "Nie udało się zapisać lub przekonwertować pliku."
            if request.is_json:
                return jsonify({"success": False, "message": message}), 500
            flash(message, 'danger')
            return redirect(request.url)

        try:
            # Pobranie wybranych opcji augmentacji
            augment_options = request.form.get('augment_options', '')
            augment_options = augment_options.split(',') if augment_options else []

            processed_filename = unique_filename
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            process_audio(upload_path, processed_path, augment_options=augment_options)
            logger.info(f"Plik audio przetworzony: {processed_path}")

            # Transkrypcja za pomocą Whisper
            transcription = transcribe_with_whisper(processed_path)
            logger.info(f"Transkrypcja zakończona: {transcription}")

            # Tworzenie profilu głosu
            voice_profile = VoiceProfile(
                user_id=user_id,
                name=name,
                audio_file=processed_filename,
                transcription=transcription,
                language=language
            )
            db.session.add(voice_profile)
            db.session.commit()
            logger.info(f"Profil głosowy utworzony: ID {voice_profile.id}")

            message = "Profil głosowy został utworzony, przetworzony i transkrybowany."
            if request.is_json:
                return jsonify({"success": True, "profile_id": voice_profile.id, "message": message}), 200
            flash(message, 'success')
            return redirect(url_for('analyze_audio', profile_id=voice_profile.id))

        except Exception as e:
            db.session.rollback()
            logger.error(f"Błąd podczas przetwarzania audio: {e}", exc_info=True)
            message = "Wystąpił błąd podczas przetwarzania audio."
            if request.is_json:
                return jsonify({"success": False, "message": message}), 500
            flash(message, 'danger')
            return redirect(request.url)

    # GET request
    return render_template('upload_voice.html')
    
@app.route('/profile')
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    profiles = VoiceProfile.query.filter_by(user_id=user_id).all()

    profiles_with_training_status = []
    for profile in profiles:
        asr_trained = profile.asr_model is not None
        profiles_with_training_status.append({
            'profile': profile,
            'asr_trained': asr_trained
        })

    return render_template('profile.html', profiles=profiles_with_training_status)
    
@app.route('/train_asr_model/<int:profile_id>', methods=['POST'])
@jwt_required()
def train_asr_model_route(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()

    if not profile:
        flash("Profil g³osowy nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    # Przygotowanie danych augmentowanych
    augmented_audio_paths = []
    augmented_transcriptions = []
    for i in range(3):  # Generowanie trzech ró¿nych augmentacji
        unique_output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"augmented_{uuid.uuid4().hex}.wav")
        augmented_path = augment_audio(audio_path, unique_output_path, augmentation_type="noise")
        if augmented_path:
            augmented_audio_paths.append(augmented_path)
            augmented_transcriptions.append(profile.transcription)  # U¿ycie oryginalnej transkrypcji

    # Po³¹czenie oryginalnego i augmentowanych plików audio oraz transkrypcji
    all_audio_files = [audio_path] + augmented_audio_paths
    all_transcriptions = [profile.transcription] + augmented_transcriptions

    with asr_model_lock:
        if profile_id in training_progress:
            flash("Trening dla tego profilu jest ju¿ w toku.", 'warning')
            return jsonify({"error": "Trening ju¿ w toku."}), 400

        training_progress[profile_id] = {
            "status": "Rozpoczêcie treningu...",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 10,
            "time_elapsed": 0,
            "is_paused": False,
            "metrics": {}
        }
        logger.debug(f"Zainicjowano training_progress dla profilu {profile_id}: {training_progress[profile_id]}")

    def train():
        start_time = time.time()
        # Zabezpieczenie przy u¿yciu semafora
        training_semaphore.acquire()
        try:
            # Aktualizacja statusu na "£adowanie modelu"
            update_training_progress(profile_id, status="Ładowanie modelu...", progress=5)

            # Wywo³anie funkcji treningowej
            train_asr_on_voice_profile(profile, app.config, all_audio_files, all_transcriptions, num_epochs=10, batch_size=8, num_workers=2)

            # Aktualizacja statusu na zakoñczony trening
            update_training_progress(profile_id, status="Trening zakoñczony pomylnie.", progress=100, current_epoch=10)

            logger.info(f"Trening ASR dla profilu ID {profile_id} zakoñczony pomylnie.")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"B³¹d podczas treningu ASR dla profilu ID {profile_id}: {e}")
            update_training_progress(profile_id, status=f"B³¹d: {str(e)}", progress=0)
        finally:
            # Zwolnienie semafora
            training_semaphore.release()
            # Usuniêcie statusu treningu po zakoñczeniu lub b³êdzie
            with asr_model_lock:
                if profile_id in training_progress:
                    del training_progress[profile_id]

    # Uruchomienie treningu w osobnym w¹tku
    try:
        executor.submit(train)
        logger.info(f"Rozpoczêto trening dla profilu ID {profile_id}.")
        flash("Trening zosta³ rozpoczêty.", 'success')
    except Exception as e:
        logger.error(f"Nie uda³o siê rozpocz¹æ treningu: {e}")
        flash("Nie uda³o siê rozpocz¹æ treningu.", 'danger')
        return jsonify({"error": "Nie uda³o siê rozpocz¹æ treningu."}), 500

    return jsonify({"message": "Trening zosta³ rozpoczêty."}), 200

@app.route('/training_status/<int:profile_id>', methods=['GET'])
@jwt_required()
def training_status(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        return jsonify({"error": "Profil g³osowy nie zosta³ znaleziony."}), 404

    if request.headers.get('Accept') == 'text/event-stream':
        # Handle SSE for real-time updates
        def generate():
            while profile_id in training_progress:
                progress_data = training_progress.get(profile_id, {
                    "status": "Nie rozpoczêto",
                    "progress": 0,
                    "current_epoch": 0,
                    "total_epochs": 10,
                    "time_elapsed": 0,
                    "is_paused": False
                })
                yield f"data: {json.dumps(progress_data)}\n\n"
                time.sleep(1)
            yield f"data: {json.dumps({'status': 'Zakoñczono', 'progress': 100, 'current_epoch': 10, 'total_epochs': 10, 'time_elapsed': 0, 'is_paused': False})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        # Return JSON status
        progress = training_progress.get(profile_id, {
            "status": "Nie rozpoczêto",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 10,
            "time_elapsed": 0,
            "is_paused": False
        })
        return jsonify(progress)

# ------------------- Helper Functions for Training Progress -------------------

# Helper function to update training progress
def update_training_progress(profile_id, status=None, progress=None, current_epoch=None, total_epochs=None, time_elapsed=None, metrics=None, loss=None):
    with asr_model_lock:
        if profile_id not in training_progress:
            training_progress[profile_id] = {
                "status": status or "Rozpoczynanie...",
                "progress": progress or 0,
                "current_epoch": current_epoch or 0,
                "total_epochs": total_epochs or 10,
                "time_elapsed": time_elapsed or 0,
                "is_paused": training_progress.get(profile_id, {}).get("is_paused", False),
                "metrics": metrics or {},
                "loss": loss or 0.0
            }
        else:
            if status:
                training_progress[profile_id]["status"] = status
            if progress is not None:
                training_progress[profile_id]["progress"] = progress
            if current_epoch is not None:
                training_progress[profile_id]["current_epoch"] = current_epoch
            if total_epochs is not None:
                training_progress[profile_id]["total_epochs"] = total_epochs
            if time_elapsed is not None:
                training_progress[profile_id]["time_elapsed"] = time_elapsed
            if metrics:
                training_progress[profile_id]["metrics"].update(metrics)
            if loss is not None:
                training_progress[profile_id]["loss"] = loss
                
@app.route('/pause_training/<int:profile_id>', methods=['POST'])
@jwt_required()
def pause_training(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()

    if not profile:
        return jsonify({"error": "Profil g³osowy nie zosta³ znaleziony."}), 404

    if profile_id in training_progress:
        training_progress[profile_id]["status"] = "Trening wstrzymany"
        training_progress[profile_id]["is_paused"] = True
        pause_flags[profile_id] = True
        logger.info(f"Training paused for profile {profile_id}.")
        return jsonify({"message": "Trening zosta³ wstrzymany."}), 200
    return jsonify({"error": "Trening nie zosta³ znaleziony lub ju¿ zakoñczony."}), 400

@app.route('/resume_training/<int:profile_id>', methods=['POST'])
@jwt_required()
def resume_training(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()

    if not profile:
        return jsonify({"error": "Profil g³osowy nie zosta³ znaleziony."}), 404

    if profile_id in training_progress and training_progress[profile_id].get("is_paused", False):
        training_progress[profile_id]["status"] = "Wznowiono trening"
        training_progress[profile_id]["is_paused"] = False
        pause_flags[profile_id] = False
        logger.info(f"Training resumed for profile {profile_id}.")
        return jsonify({"message": "Trening zosta³ wznowiony."}), 200
    return jsonify({"error": "Trening nie zosta³ wstrzymany lub ju¿ zakoñczony."}), 400

@app.route('/tts', methods=['GET', 'POST'])
@jwt_required()
def tts():
    user_id = get_jwt_identity()
    profiles = VoiceProfile.query.filter_by(user_id=user_id).all()

    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        voice_id = request.form.get('voice_id')
        emotion = request.form.get('emotion', 'neutral')
        try:
            intonation = float(request.form.get('intonation', 1.0))
        except ValueError:
            intonation = 1.0  # Wartość domyślna, jeśli nie uda się skonwertować
        
        if not text:
            flash("Nie podano tekstu do syntezy.", 'danger')
            return redirect(request.url)

        if not voice_id:
            flash("Nie wybrano profilu głosowego.", 'danger')
            return redirect(request.url)

        profile = VoiceProfile.query.filter_by(id=voice_id, user_id=user_id).first()
        if not profile:
            flash("Profil głosowy nie został znaleziony.", 'danger')
            return redirect(request.url)

        # Sprawdzenie, czy asr_model istnieje
        if not profile.asr_model or not profile.asr_model.model_file:
            flash("Nie znaleziono przypisanego modelu ASR do profilu głosowego.", 'danger')
            return redirect(request.url)

        model_path = os.path.join(app.config['ASR_MODELS_FOLDER'], profile.asr_model.model_file)

        if not os.path.exists(model_path):
            flash(f"Błąd: Plik modelu ASR nie został znaleziony: {model_path}", 'danger')
            return redirect(request.url)

        try:
            output_filename = generate_speech(text, profile, emotion, intonation)
            flash("Mowa została wygenerowana i jest dostępna do pobrania.", 'success')
            return send_from_directory(app.config['GENERATED_FOLDER'], output_filename, as_attachment=True)
        except Exception as e:
            flash(f"Błąd podczas generowania mowy: {e}", 'danger')
            return redirect(request.url)

    return render_template('tts.html', profiles=profiles)
    
@app.route('/play_audio/<filename>')
@jwt_required()
def play_audio(filename: str):
    return render_template('play_audio.html', filename=filename)

@app.route('/static/generated/<filename>')
def serve_generated_audio(filename: str):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

@app.route('/static/processed/<filename>')
@jwt_required()
def serve_processed_audio(filename: str):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(audio_file=filename, user_id=user_id).first()
    if not profile:
        flash("Plik audio nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    # Check if the file exists
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        logger.error(f"Plik {filename} nie istnieje w katalogu {app.config['UPLOAD_FOLDER']}")
        flash("Plik audio nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

@app.route('/edit_profile/<int:profile_id>', methods=['GET', 'POST'])
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
                logger.error(f"Error updating voice profile: {e}")
                flash("Wyst¹pi³ b³¹d podczas aktualizacji profilu.", 'danger')
                return redirect(request.url)
        else:
            flash("Nazwa profilu i jêzyk nie mog¹ byæ puste.", 'danger')
            return redirect(request.url)

    return render_template('edit_profile.html', profile=profile)

@app.route('/delete_profile/<int:profile_id>', methods=['POST'])
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
            logger.info(f"Deleted audio file: {processed_file_path}")

        # Delete associated ASR model if exists
        if profile.asr_model:
            if os.path.exists(profile.asr_model.model_file):
                os.remove(profile.asr_model.model_file)
                logger.info(f"Deleted ASR model file: {profile.asr_model.model_file}")
            db.session.delete(profile.asr_model)

        db.session.delete(profile)
        db.session.commit()
        flash("Profil g³osowy zosta³ usuniêty.", 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting voice profile: {e}")
        flash("Wyst¹pi³ b³¹d podczas usuwania profilu.", 'danger')

    return redirect(url_for('profile'))

def audio_signal_function(audio_path: str) -> list:
    """
    ?aduje plik audio i zwraca sygna? jako list? warto?ci amplitudy.

    Args:
        audio_path (str): ?cie?ka do pliku audio.

    Returns:
        list: Lista warto?ci amplitudy sygna?u audio. Zwraca pust? list? w przypadku b??du.
    """
    try:
        logger.info(f"?adowanie pliku audio: {audio_path}")
        # Za?aduj audio za pomoc? librosa
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        logger.debug(f"Audio za?adowane. Cz?stotliwo?? próbkowania: {sample_rate} Hz, d?ugo??: {len(audio)} próbek")
        # Konwertuj sygna? na list?
        audio_signal = audio.tolist()
        return audio_signal
    except Exception as e:
        logger.error(f"B??d podczas ?adowania sygna?u audio z pliku {audio_path}: {e}", exc_info=True)
        return []

@app.route('/analyze_audio/<int:profile_id>', methods=['GET'])
@jwt_required()
def analyze_audio(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil głosowy nie został znaleziony.", 'danger')
        logger.warning(f"Profil głosowy ID {profile_id} nie znaleziony dla użytkownika ID {user_id}.")
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie został znaleziony.", 'danger')
        logger.error(f"Plik audio {audio_path} nie istnieje.")
        return redirect(url_for('profile'))

    try:
        # Pobierz instancję ASRBrain, jeżeli potrzebna w dalszej części
        asr_brain = get_asr_brain()

        # Poprawione wywołanie funkcji bez przekazywania asr_brain jako argumentu n_mels
        mel_tensor, processing_info, save_status = process_audio_to_dataset(
            audio_path=audio_path,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
            max_duration_sec=400.0,
            save_dir="processed_data",
            max_iterations=3
        )

        if mel_tensor is None:
            flash("Wystąpił błąd podczas analizy audio.", 'danger')
            logger.error(f"process_audio_to_dataset zwrócił None dla pliku {audio_path}.")
            return redirect(url_for('profile'))

        suitability = evaluate_audio_suitability(processing_info)

        audio_signal = audio_signal_function(audio_path)
        if audio_signal:
            processing_info['audio_signal'] = audio_signal
            logger.info(f"Sygnał audio załadowany poprawnie dla pliku: {audio_path}")
        else:
            processing_info['audio_signal'] = []
            logger.warning(f"Sygnał audio nie został poprawnie załadowany dla pliku: {audio_path}")

        return render_template('audio_analysis.html',
                               profile=profile,
                               processing_info=processing_info,
                               mel_spectrogram=mel_tensor.tolist(),
                               suitability=suitability,
                               save_status=save_status,
                               transcription=profile.transcription)
    except Exception as e:
        logger.error(f"Error during audio analysis: {e}", exc_info=True)
        flash(f"Wystąpił błąd podczas analizy audio: {str(e)}", 'danger')
        return redirect(url_for('profile'))


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


# ------------------- Run Application -------------------
if __name__ == "__main__":
    try:
        with app.app_context():
            db.create_all()
            logger.info("Starting to load ASR model...")
            asr_brain_instance = get_asr_brain()
            if asr_brain_instance:
                logger.info("ASR model loaded successfully.")
            else:
                logger.critical("Failed to load ASR model. Exiting application.")
                sys.exit(1)
        logger.info("Flask application has been started.")

        monitor_thread = threading.Thread(target=background_system_memory_monitor, args=(240,), daemon=True)
        monitor_thread.start()
        logger.info("System memory monitor thread started.")

        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

    except KeyboardInterrupt:
        logger.info("Application has been stopped by the user.")
    except Exception as e:
        logger.critical(f"Failed to start the application: {e}")
    finally:
        logger.info("Application cleanup process completed.")
