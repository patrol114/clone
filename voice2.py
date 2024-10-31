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
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')
warnings.filterwarnings("ignore", category=FutureWarning, module='speechbrain')
from speechbrain.utils.data_pipeline import takes, provides
from tqdm import tqdm
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
training_semaphore = threading.Semaphore(1)
import gc
from torch.utils.data import DataLoader
from safetensors.torch import save_file as safetensors_save_file
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.benchmark = True
torch.set_num_interop_threads(os.cpu_count())
import magic
import json
import asyncio
from typing import List, Dict, Tuple
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from speechbrain.inference import EncoderDecoderASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.dataio.batch import PaddedBatch, PaddedData, BatchsizeGuesser
from torch.amp import autocast
from torch import GradScaler
from speechbrain.utils.autocast import fwd_default_precision
from speechbrain.utils.data_pipeline import DataPipeline
from functools import partial
from speechbrain.dataio.encoder import CTCTextEncoder
import speechbrain as sb
from speechbrain.core import Brain
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from concurrent.futures import ThreadPoolExecutor, as_completed
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.nnet.losses import ctc_loss
import torch.nn.functional as F

asr_brain_instance = None
asr_brain_lock_singleton = threading.Lock()

def get_asr_brain() -> 'ASRBrain':
    global asr_brain_instance
    with asr_brain_lock_singleton:
        if asr_brain_instance is None:
            try:
                logger.info("Ładowanie modelu ASR (Singleton)...")
                model, processor, device = load_asr_model()
                if not model or not processor:
                    logger.critical("Nie udało się załadować modelu ASR.")
                    raise RuntimeError("Nie udało się załadować modelu ASR.")

                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                checkpointer = Checkpointer(
                    checkpoints_dir=app.config['ASR_MODELS_FOLDER'],
                    recoverables={
                        "model": model,
                        "optimizer": optimizer
                    }
                )

                asr_brain_instance = ASRBrain(
                    modules={"model": model},
                    opt_class=lambda params: torch.optim.AdamW(params, lr=0.001),
                    hparams={
                        "compute_cost": ctc_loss,
                        "processor": processor,
                        "sample_rate": 16000,
                        "target_sampling_rate": 16000,
                        "use_augmentation": True,
                        "convert_to_mono": "average",
                        "noise_reduction": True,
                        "normalize_audio": True,
                        "blank_index": 0,
                        "max_epochs": 10,
                        "downsample_factor": 320,
                        "save_interval": 1
                    },
                    run_opts={
                        "device": device.type,
                        "precision": "bf16" if device.type == "cpu" else "fp16"
                    },
                    checkpointer=checkpointer
                )
                logger.info("ASRBrain Singleton został zainicjalizowany.")
            except Exception as e:
                logger.error(f"Błąd inicjalizacji ASRBrain: {e}", exc_info=True)
                raise
        return asr_brain_instance

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '1234567812345678')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///voice_cloning.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', '1234567812345678')
app.config['MAX_CONTENT_LENGTH'] = 36 * 1024 * 1024  # 36 MB

app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_NAME'] = 'access_token_cookie'
app.config['JWT_COOKIE_CSRF_PROTECT'] = False
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=10)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.getcwd(), 'processed')
app.config['GENERATED_FOLDER'] = os.path.join(os.getcwd(), 'generated')
app.config['PRETRAINED_FOLDER'] = os.path.join(os.getcwd(), 'pretrained_models')
app.config['ASR_MODELS_FOLDER'] = os.path.join(os.getcwd(), 'asr_models')

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

import sys
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

app.logger.setLevel(logging.DEBUG)

for handler in app.logger.handlers[:]:
    app.logger.removeHandler(handler)

file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

class CustomFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, '')
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

console_formatter = CustomFormatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
console_handler.setFormatter(console_formatter)

app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
logger = app.logger

processor = None
global_tokenizer = None

BASE_ASR_MODEL_ID = "badrex/xlsr-polish"

executor = ThreadPoolExecutor(max_workers=4)

def initialize_whisper_model():
    global whisper_processor, whisper_model, whisper_device
    try:
        logger.info("Ładowanie WhisperProcessor.")
        whisper_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small",
            language="pl",
            task="transcribe"
        )
        logger.info("Ładowanie WhisperForConditionalGeneration.")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Ustawienie urządzenia (GPU/CPU)
        whisper_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        whisper_model.to(whisper_device)
        logger.info(f"Model Whisper uruchomiony na urządzeniu: {whisper_device}")
        
        # Ustawienie wymuszonych ID dekodera na podstawie języka i zadania
        whisper_model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
            language="pl",
            task="transcribe"
        )
        logger.info("Whisper model and processor initialized successfully.")
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji modelu Whisper: {e}", exc_info=True)
        raise

def audio_pipeline(audio_path):
    """
    Wczytuje i przetwarza plik audio na tensor. Przygotowuje również długość sygnału.
    """
    try:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Plik audio nie istnieje: {audio_path}")
        
        logger.info(f"Wczytywanie pliku audio: {audio_path}")
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        if audio.size == 0:
            raise ValueError("Sygnał audio jest pusty.")

        waveform = torch.tensor(audio, dtype=torch.float32)
        audio_lens = torch.tensor([len(audio)], dtype=torch.long)
        return waveform, audio_lens

    except Exception as e:
        logger.error(f"Błąd w audio_pipeline: {e}", exc_info=True)
        raise


def text_pipeline_with_tokenizer(transcription):
    """
    Tokenizuje tekst przy użyciu globalnego tokenizera.
    """
    global global_tokenizer
    try:
        # Inicjalizacja tokenizera, jeśli nie jest dostępny
        if global_tokenizer is None:
            processor = Wav2Vec2Processor.from_pretrained(BASE_ASR_MODEL_ID)
            global_tokenizer = processor.tokenizer
            logger.info("Tokenizator został zainicjalizowany.")

        tokenizer = global_tokenizer
        tokens = tokenizer.encode(transcription)
        
        tokens_encoded = torch.tensor(tokens, dtype=torch.long)
        tokens_lens = torch.tensor([len(tokens)], dtype=torch.long)

        return tokens_encoded, tokens_lens

    except Exception as e:
        logger.error(f"Błąd w text_pipeline_with_tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Tokenizacja transkrypcji nie powiodła się: {e}")
        
def read_audio_from_path(audio_path):
    try:
        if not audio_path:
            raise ValueError("The 'audio_path' is missing or empty.")

        audio_signal, _ = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio_signal) == 0:
            raise ValueError(f"The loaded audio signal is empty for the path: {audio_path}")

        return audio_signal, len(audio_signal), audio_path
    except Exception as e:
        logger.error(f"Error reading audio file from {audio_path}: {e}", exc_info=True)
        raise

def prepare_training_data(preprocessed_audio_files: list, transcriptions: list, split_ratio: float = 0.8):
    """
    Przygotowuje dane treningowe i walidacyjne na podstawie przetworzonych plików audio i transkrypcji.
    """
    try:
        if len(preprocessed_audio_files) != len(transcriptions):
            error_msg = f"Liczba plików audio ({len(preprocessed_audio_files)}) nie zgadza się z liczbą transkrypcji ({len(transcriptions)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Budowanie zestawu danych
        data = {
            f'sample_{idx}': {
                'audio_path': audio_path,
                'transcription': transcription
            } for idx, (audio_path, transcription) in enumerate(zip(preprocessed_audio_files, transcriptions))
        }

        # Logowanie przykładów danych
        for sample_id, sample_data in list(data.items())[:5]:
            logger.debug(f"Przykład danych - ID: {sample_id}, Dane: {sample_data}")

        # Losowy podział na dane treningowe i walidacyjne
        data_items = list(data.items())
        random.shuffle(data_items)
        split_point = int(len(data_items) * split_ratio)
        
        train_items = dict(data_items[:split_point])
        valid_items = dict(data_items[split_point:])

        logger.info(f"Rozmiar zbioru treningowego: {len(train_items)}, Rozmiar zbioru walidacyjnego: {len(valid_items)}")
        return train_items, valid_items

    except Exception as e:
        logger.error(f"Błąd w prepare_training_data: {e}", exc_info=True)
        raise


asr_model_lock = threading.Lock()

asr_model_cache = {}

model_loading_progress = {}
asr_model_cache_lock = threading.Lock()

def load_asr_model(profile_id=None, base_model_id=BASE_ASR_MODEL_ID, cache_dir=".cache", use_gradient_checkpointing=True):
    global asr_model_cache
    with asr_model_lock:
        if profile_id:
            asr_model_entry = ASRModel.query.filter_by(voice_profile_id=profile_id).first()
            if asr_model_entry and asr_model_entry.model_file.endswith('.pt'):
                model_path = os.path.join(app.config['ASR_MODELS_FOLDER'], asr_model_entry.model_file)

                if model_path in asr_model_cache:
                    logger.info(f"Profile-specific model for profile ID {profile_id} loaded from cache.")
                    return asr_model_cache[model_path]

                if os.path.exists(model_path):
                    logger.info(f"Loading profile-specific ASR model from: {model_path}")
                    processor = Wav2Vec2Processor.from_pretrained(base_model_id, cache_dir=cache_dir)
                    model = Wav2Vec2ForCTC.from_pretrained(base_model_id, cache_dir=cache_dir)
                    state_dict = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    model.load_state_dict(state_dict)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)

                    asr_model_cache[model_path] = (model, processor, device)
                    return model, processor, device
                else:
                    logger.error(f"Model file for profile {profile_id} not found.")
                    raise FileNotFoundError(f"Model file for profile {profile_id} not found.")

        logger.info(f"Falling back to loading base model: {base_model_id}")
        if base_model_id in asr_model_cache:
            logger.info(f"Base ASR model '{base_model_id}' loaded from cache.")
            return asr_model_cache[base_model_id]

        try:
            processor = Wav2Vec2Processor.from_pretrained(base_model_id, cache_dir=cache_dir)
            model = Wav2Vec2ForCTC.from_pretrained(base_model_id, cache_dir=cache_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            if use_gradient_checkpointing and hasattr(model.config, 'gradient_checkpointing'):
                model.config.gradient_checkpointing = True
                logger.info("Gradient checkpointing enabled for the base model.")

            logger.info(f"Base ASR model '{base_model_id}' loaded on {device}.")

            asr_model_cache[base_model_id] = (model, processor, device)
            return model, processor, device

        except Exception as e:
            logger.error(f"Error loading base ASR model '{base_model_id}': {e}")
            raise RuntimeError(f"Error loading ASR model: {e}")
        
def setup_dataio(
    asr_brain: "ASRBrain",
    preprocessed_audio_files: list,
    transcriptions: list,
    split_ratio: float = 0.8
) -> tuple:
    try:
        # Prepare training and validation data
        train_data, valid_data = prepare_training_data(preprocessed_audio_files, transcriptions, split_ratio)

        # Create DynamicItemDatasets using the constructor
        train_dataset = sb.dataio.dataset.DynamicItemDataset(train_data)
        valid_dataset = sb.dataio.dataset.DynamicItemDataset(valid_data)

        # Add dynamic items for audio processing
        sb.dataio.dataset.add_dynamic_item(
            [train_dataset, valid_dataset],
            audio_pipeline,
            takes=["audio_path"],
            provides=["sig", "audio_lens"]
        )

        # Assign tokenizer to the global variable
        global global_tokenizer
        global_tokenizer = asr_brain.hparams.processor.tokenizer

        # Add dynamic items for text processing
        sb.dataio.dataset.add_dynamic_item(
            [train_dataset, valid_dataset],
            text_pipeline_with_tokenizer,
            takes=["transcription"],
            provides=["tokens_encoded", "tokens_lens"]
        )

        # Set output keys (usuń 'id' jeśli jest automatycznie obsługiwane)
        sb.dataio.dataset.set_output_keys(
            [train_dataset, valid_dataset],
            ['sig', 'audio_lens', 'tokens_encoded', 'tokens_lens']
        )

        logger.info("DataIO setup completed successfully.")
        return train_dataset, valid_dataset

    except Exception as e:
        logger.error(f"Error in setup_dataio: {e}", exc_info=True)
        raise


def background_system_memory_monitor(interval=240):
    while True:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)
            gpu_total = info_gpu.total // (1024 ** 2)
            gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Nie udało się uzyskać informacji o pamięci GPU: {e}")
            gpu_used, gpu_total, gpu_percent = 0, 0, 0

        try:
            ram_info = psutil.virtual_memory()
            ram_used = ram_info.used // (1024 ** 2)
            ram_total = ram_info.total // (1024 ** 2)
            ram_percent = ram_info.percent
            cpu_percent = psutil.cpu_percent(interval=1)
        except Exception as e:
            logger.warning(f"Nie udało się uzyskać informacji o pamięci RAM/CPU: {e}")
            ram_used, ram_total, ram_percent, cpu_percent = 0, 0, 0, 0

        logger.info(f"Zużycie Systemowe - GPU: {gpu_used} MB / {gpu_total} MB ({gpu_percent:.2f}%) | RAM: {ram_used} MB / {ram_total} MB ({ram_percent}%) | CPU: {cpu_percent}%")

        if ram_percent >= 100:
            logger.critical(f"Zużycie RAM przekroczyło 100% ({ram_percent}%). Zamykanie aplikacji.")
            os._exit(1)

        time.sleep(interval)

training_progress = {}
pause_flags = {}

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    audio_file = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(50), nullable=False, default='pl')
    created_at = db.Column(db.DateTime, default=db.func.now())
    transcription = db.Column(db.Text, nullable=True)
    sample_rate = db.Column(db.Integer, nullable=True)
    num_samples = db.Column(db.Integer, nullable=True)
    rms_db = db.Column(db.Float, nullable=True)
    zcr = db.Column(db.Float, nullable=True)
    snr_db = db.Column(db.Float, nullable=True)
    asr_model = db.relationship('ASRModel', uselist=True, backref='voice_profile')

class ASRModel(db.Model):
    __tablename__ = 'asr_model'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    voice_profile_id = db.Column(db.Integer, db.ForeignKey('voice_profile.id'), nullable=False, unique=True, index=True)
    name = db.Column(db.String(100), nullable=False)
    model_file = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(50), nullable=True, default='pl')
    created_at = db.Column(db.DateTime, default=db.func.now())


    
def reduce_noise_sample(args):
    sample, sample_idx, sample_rate = args
    try:
        if np.max(np.abs(sample)) > 1e-5:
            max_length = 16000
            segments = []

            for j in range(0, len(sample), max_length):
                segment = sample[j:j + max_length]
                reduced_segment = nr.reduce_noise(y=segment, sr=sample_rate)
                segments.append(reduced_segment)

            reduced_sample = np.concatenate(segments, axis=0)
            return (sample_idx, reduced_sample)
        else:
            logger.warning(f"Sygnał zbyt cichy, pomijanie redukcji szumów dla próbki {sample_idx}")
            return (sample_idx, sample)
    except Exception as e:
        logger.error(f"Błąd podczas redukcji szumów dla próbki {sample_idx}: {e}", exc_info=True)
        return (sample_idx, sample)
    


def collate_fn(batch):

    try:

        logger.debug("Rozpoczęcie przetwarzania batcha.")

        # Filtrujemy próbki None

        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:

            logger.warning("Batch jest pusty po filtrowaniu.")

            return None

            

        logger.debug("Konwersja tokens_encoded do tensorów Long.")

        # Usunięto requires_grad dla tokens_encoded, ponieważ są to indeksy

        batch_tokens_encoded = [

            item['tokens_encoded'].clone().detach() if isinstance(item['tokens_encoded'], torch.Tensor)

            else torch.tensor(item['tokens_encoded'], dtype=torch.long)

            for item in batch

        ]

        

        logger.debug("Padding sekwencji tokenów.")

        tokens_encoded_padded = torch.nn.utils.rnn.pad_sequence(

            batch_tokens_encoded,

            batch_first=True,

            padding_value=global_tokenizer.pad_token_id

        )

        

        # Konwersja sygnałów audio na float32

        for item in batch:

            if isinstance(item['sig'], torch.Tensor):

                item['sig'] = item['sig'].float()  # Konwersja do float32

            else:

                item['sig'] = torch.tensor(item['sig'], dtype=torch.float32)

        

        logger.debug("Tworzenie obiektu PaddedBatch.")

        batched_data = PaddedBatch(

            examples=batch,

            padded_keys=['sig', 'tokens_encoded'],

            device_prep_keys=['sig', 'tokens_encoded'],

            apply_default_convert=False,

            nonpadded_stack=True

        )

        

        logger.debug("Ustalanie długości audio i tokenów.")

        # Długości jako long bez requires_grad

        batched_data.audio_lens = torch.tensor(

            [sample['audio_lens'] for sample in batch], 

            dtype=torch.long

        ).clone().detach()

        

        batched_data.tokens_lens = torch.tensor(

            [sample['tokens_lens'] for sample in batch], 

            dtype=torch.long

        ).clone().detach()

        

        logger.debug("Sprawdzanie typów danych dla sig i tokens_encoded.")

        for sample in batch:

            assert isinstance(sample['sig'], torch.Tensor), "sig nie jest tensorem"

            assert isinstance(sample['tokens_encoded'], torch.Tensor), "tokens_encoded nie jest tensorem"

            # Dodatkowe sprawdzenie typów

            assert sample['sig'].dtype == torch.float32, "sig nie jest typu float32"

            assert sample['tokens_encoded'].dtype == torch.long, "tokens_encoded nie jest typu long"

        

        logger.info("Batch przetworzony pomyślnie.")

        return batched_data

        

    except Exception as e:

        logger.error(f"Błąd podczas paddingu batcha: {str(e)}", exc_info=True)

        raise

def singleton(cls):
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class ASRBrain(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts=None, checkpointer=None, use_amp=True):
        super().__init__(modules, opt_class, hparams, run_opts=run_opts, checkpointer=checkpointer)

        self.run_opts = run_opts or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modules['model'].to(self.device)
        logger.info(f"ASRBrain urządzenie: {self.device} (Typ: {type(self.device)})")

        self.checkpointer = checkpointer
        if self.checkpointer is None:
            logger.error("Checkpointer nie został prawidłowo zainicjalizowany.")

        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)
        self.configure_optimizers()

        # Inicjalizacja hiperparametrów
        self.hparams.batch_size = getattr(hparams, 'batch_size', 8)
        self.hparams.num_workers = getattr(hparams, 'num_workers', 0)
        self.hparams.lr = getattr(hparams, 'lr', 0.001)  # Domyślna wartość learning rate

        self.enable_gradient_checkpointing = getattr(self.hparams, "enable_gradient_checkpointing", False)
        if self.enable_gradient_checkpointing:
            if hasattr(self.modules['model'], 'config'):
                self.modules['model'].config.gradient_checkpointing = True
                logger.info("Gradient checkpointing włączony.")
            else:
                logger.warning("Model nie posiada atrybutu 'config'. Gradient checkpointing nie może być włączony.")

        # Logika kompilacji
        self.compile_using_fullgraph = self.run_opts.get('compile_using_fullgraph', False)
        self.compile_using_dynamic_shape_tracing = self.run_opts.get('compile_using_dynamic_shape_tracing', False)

        if self.compile_using_fullgraph or self.compile_using_dynamic_shape_tracing:
            try:
                self.modules['model'] = torch.compile(
                    self.modules['model'],
                    fullgraph=self.compile_using_fullgraph,
                    dynamic=True
                )
                logger.info("Model skompilowany przy użyciu ustawień dynamicznych.")
            except Exception as e:
                logger.error(f"Błąd podczas kompilacji modelu: {e}", exc_info=True)

        # Obsługa precision
        precision = self.run_opts.get("precision", "fp16")
        if self.device.type == "cuda" and precision == "fp16":
            self.scaler = GradScaler() if use_amp else None
            logger.info("GradScaler zainicjalizowany dla mixed precision.")
        else:
            self.scaler = None
            logger.info("GradScaler nie jest używany.")

        self.gpu_initialized = False
        self.init_pynvml()
        self.current_profile_id = None

        # Inicjalizacja historii statystyk
        self.train_stats_history = {}
        self.valid_stats_history = {}
        
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
            logger.warning(f"Nie udało się zainicjalizować pynvml: {e}")
            self.handle = None

    def load_from_checkpoint(cls, checkpoint_path: str, run_opts: dict):
        """
        Ładuje model ASR z pliku checkpoint.

        Args:
            checkpoint_path (str): Ścieżka do pliku checkpoint.
            run_opts (dict): Opcje uruchomienia, np. urządzenie (CPU/GPU).

        Returns:
            ASRBrain: Załadowana instancja ASRBrain.
        """
        try:
            # Załaduj stan modelu
            state_dict = torch.load(checkpoint_path, map_location=run_opts["device"])

            # Inicjalizacja ASRBrain z odpowiednimi modułami, optymalizatorem i hiperparametrami
            model, processor, device = load_asr_model()  # Upewnij się, że ta funkcja działa poprawnie
            model.to(device)

            checkpointer = Checkpointer(
                checkpoints_dir=os.path.dirname(checkpoint_path),
                recoverables={"model": model, "optimizer": torch.optim.AdamW(model.parameters(), lr=0.001)}
            )

            asr_brain = cls(
                modules={"model": model},
                opt_class=lambda params: torch.optim.AdamW(params, lr=0.001),
                hparams={
                    "compute_cost": ctc_loss,
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
                run_opts=run_opts,
                checkpointer=checkpointer
            )

            # Załaduj stan modelu
            asr_brain.modules['model'].load_state_dict(state_dict)

            logger.info(f"Model ASR został załadowany z {checkpoint_path} na urządzeniu {run_opts['device']}.")
            return asr_brain

        except Exception as e:
            logger.error(f"Błąd podczas ładowania checkpointa ASR: {e}", exc_info=True)
            return None

    def monitor_memory_usage(self):
        try:
            # Monitorowanie GPU za pomocą pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_used = info_gpu.used // (1024 ** 2)  # Zużycie GPU w MB
                gpu_total = info_gpu.total // (1024 ** 2)  # Całkowita pamięć GPU w MB
                gpu_percent = (gpu_used / gpu_total) * 100
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Nie udało się uzyskać informacji o pamięci GPU: {e}")
                gpu_used, gpu_total, gpu_percent = 0, 0, 0

            # Monitorowanie pamięci RAM za pomocą psutil
            import psutil
            ram_info = psutil.virtual_memory()
            ram_used = ram_info.used // (1024 ** 2)  # Zużycie RAM w MB
            ram_total = ram_info.total // (1024 ** 2)  # Całkowita pamięć RAM w MB
            ram_percent = ram_info.percent

            # Monitorowanie użycia CPU
            cpu_percent = psutil.cpu_percent(interval=1)

            logger.info(f"Zużycie GPU: {gpu_used}/{gpu_total} MB ({gpu_percent:.2f}%)")
            logger.info(f"Zużycie RAM: {ram_used}/{ram_total} MB ({ram_percent}%)")
            logger.info(f"Zużycie CPU: {cpu_percent:.2f}%")

            # Automatyczne czyszczenie pamięci przy wysokim zużyciu
            if ram_percent > 90 or gpu_percent > 90:
                logger.warning("Zużycie pamięci przekracza 90%, zwalnianie pamięci...")
                torch.cuda.empty_cache()
                import gc
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

        except Exception as e:
            logger.error(f"Błąd podczas monitorowania zużycia pamięci: {e}", exc_info=True)
            return {
                "gpu_used": 0,
                "gpu_total": 0,
                "gpu_percent": 0,
                "ram_used": 0,
                "ram_total": 0,
                "ram_percent": 0,
                "cpu_percent": 0
            }

    def adjust_model_parameters(self, memory_info, max_ram=None, max_gpu=None):
        """
        Dostosowuje parametry modelu na podstawie zużycia pamięci RAM i GPU.

        Args:
            memory_info (dict): Słownik ze statystykami pamięci (wynik monitor_memory_usage).
            max_ram (int): Maksymalne dopuszczalne zużycie RAM w MB. Jeśli None, używa 90% dostępnej pamięci RAM.
            max_gpu (int): Maksymalne dopuszczalne zużycie GPU w MB. Jeśli None, używa 90% dostępnej pamięci GPU.
        """
        try:
            # Ustaw domyślne limity, jeśli nie zostały podane
            if max_ram is None:
                max_ram = memory_info['ram_total'] * 0.9  # Limit 90% dostępnej pamięci RAM
            if max_gpu is None:
                max_gpu = memory_info['gpu_total'] * 0.9  # Limit 90% dostępnej pamięci GPU

            # Sprawdzenie, czy pamięć przekracza limity
            if memory_info['ram_used'] > max_ram or memory_info['gpu_used'] > max_gpu:
                logger.warning("Zużycie pamięci przekracza limit, zmniejszamy rozmiar batcha.")

                # Redukcja rozmiaru batcha o połowę, ale nie poniżej 1
                current_batch_size = max(1, getattr(self.hparams, "batch_size", 16) // 2)
                self.hparams["batch_size"] = current_batch_size
                logger.info(f"Nowy rozmiar batcha: {current_batch_size}")

                # Wyłączenie augmentacji audio, jeśli jest aktywna
                if getattr(self.hparams, "use_augmentation", False):
                    self.hparams["use_augmentation"] = False
                    logger.info("Wyłączono augmentację danych audio.")

            else:
                logger.info("Zużycie pamięci w normie. Parametry modelu pozostają bez zmian.")

        except Exception as e:
            logger.error(f"Błąd podczas dostosowywania parametrów modelu: {e}", exc_info=True)

    def compute_metrics(self, predictions, batch):
        """
        Oblicza metryki WER i CER dla podanych predykcji.
        """
        try:
            logger.debug(f"self.hparams: {self.hparams}")

            # Obliczanie predykcji i celów
            preds = predictions.argmax(dim=-1)  # [T, batch_size]
            targets = batch['tokens_encoded'].to(self.device)  # [sum(target_lengths)]
            target_lengths = batch['tokens_lens'].to(self.device)  # [batch_size]

            # Walidacja wymiarów predykcji
            if preds.dim() != 2:
                raise ValueError(f"Predykcje po argmax powinny mieć 2 wymiary (T, batch_size), otrzymano: {preds.dim()}")

            batch_size = preds.size(1)

            # Walidacja batch_size
            if batch_size != target_lengths.size(0):
                raise ValueError(f"Batch size predykcji ({batch_size}) nie zgadza się z batch size targetów ({target_lengths.size(0)})")

            # Przygotowanie target_ids_list
            target_ids_list = self.prepare_target_ids(targets, target_lengths, batch_size)

            # Przygotowanie predykcji do dekodowania
            pred_lists = preds.transpose(0, 1).tolist()  # [batch_size, seq_length]

            # Dekodowanie predykcji i celów
            decoded_preds, decoded_targets = self.decode_predictions(pred_lists, target_ids_list)

            # Sprawdzenie, czy istnieją predykcje i cele do oceny
            if not decoded_preds or not decoded_targets:
                logger.warning("Brak predykcji lub celów do obliczenia metryk.")
                return {'wer': float('nan'), 'cer': float('nan')}

            # Obliczanie metryk WER/CER
            wer = self.calculate_wer(decoded_preds, decoded_targets)
            cer = self.calculate_cer(decoded_preds, decoded_targets)

            # Logowanie metryk
            self.metrics['wer'].append(wer)
            self.metrics['cer'].append(cer)

            logger.info("Metryki WER/CER obliczone pomyślnie")
            logger.debug(f"WER: {wer:.4f}, CER: {cer:.4f}")

            return {'wer': wer, 'cer': cer}

        except Exception as e:
            logger.error(f"Błąd podczas obliczania metryk: {e}", exc_info=True)
            raise

    def prepare_target_ids(self, targets, target_lengths, batch_size):
        """
        Przygotowuje listę target_ids na podstawie długości.
        """
        target_ids_list = []
        current_position = 0

        for i in range(batch_size):
            length = target_lengths[i].item()
            target_id = targets[current_position:current_position + length].tolist()
            if not all(isinstance(x, int) for x in target_id):
                raise ValueError(f"Wszystkie elementy target_id muszą być typu int.")
            target_ids_list.append(target_id)
            current_position += length

        return target_ids_list

    def decode_predictions(self, pred_lists, target_ids_list):
        """
        Dekoduje predykcje i cele, zwraca ich listy.
        """
        try:
            decoded_preds = self.hparams.processor.batch_decode(pred_lists, skip_special_tokens=True)
            decoded_targets = self.hparams.processor.batch_decode(target_ids_list, skip_special_tokens=True)

            for i in range(min(len(decoded_preds), len(decoded_targets))):
                logger.debug(f"Sample {i}: decoded_pred = {decoded_preds[i]}, decoded_target = {decoded_targets[i]}")

            return decoded_preds, decoded_targets
        except Exception as e:
            logger.error(f"Błąd podczas dekodowania: {e}", exc_info=True)
            raise


    def configure_optimizers(self):
        """
        Konfiguracja optymalizatora dla modelu.
        """
        try:
            if not hasattr(self, 'opt_class') or self.opt_class is None:
                raise ValueError("Klasa optymalizatora nie jest zdefiniowana")
                
            if not hasattr(self.modules, 'model'):
                raise ValueError("Model nie jest zdefiniowany w self.modules")
                
            parameters = list(self.modules['model'].parameters())
            if not parameters:
                raise ValueError("Model nie ma parametrów do optymalizacji")
                
            self.optimizer = self.opt_class(parameters)
            logger.info(f"Optymalizator {self.optimizer.__class__.__name__} skonfigurowany pomyślnie")
            
            return self.optimizer
            
        except Exception as e:
            logger.error(f"Błąd podczas konfigurowania optymalizatora: {e}", exc_info=True)
            raise

    def compute_forward(self, batch: dict, stage: sb.Stage) -> dict:
        try:
            memory_info = self.monitor_memory_usage()

            logger.debug("Wejście do compute_forward...")
            wavs, wav_lens = batch['sig'].data, batch['audio_lens'].data

            if wavs is None:
                logger.error("Dane wejściowe (wavs) są puste. Nie można kontynuować przetwarzania.")
                raise ValueError("Dane audio (wavs) nie zostały poprawnie załadowane.")

            logger.debug(f"Kształt wavs: {wavs.shape}, wav_lens: {wav_lens}")

            # Sprawdzenie i ewentualne dodanie wymiarów
            if wavs.dim() == 1:
                # Dodajemy wymiar batch_size i kanału
                wavs = wavs.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_length]
                logger.debug("Dodano wymiary batch_size i kanału do sygnału audio.")
            elif wavs.dim() == 2:
                # Dodajemy wymiar kanału
                wavs = wavs.unsqueeze(1)  # [batch_size, 1, seq_length]
                logger.debug("Dodano wymiar kanału do sygnału audio.")
            elif wavs.dim() == 3:
                logger.debug("Wavs ma oczekiwaną liczbę wymiarów.")
            else:
                logger.error(f"Nieoczekiwana liczba wymiarów dla wavs: {wavs.dim()}")
                raise ValueError(f"Nieoczekiwana liczba wymiarów dla wavs: {wavs.dim()}")

            MIN_INPUT_SIZE = 16000
            current_length = wavs.shape[2]
            if current_length < MIN_INPUT_SIZE:
                padding_size = MIN_INPUT_SIZE - current_length
                wavs = torch.nn.functional.pad(wavs, (0, padding_size), "constant", 0)
                logger.info(f"Sygnał był zbyt krótki. Dodano padding, nowy kształt: {wavs.shape}")

            if getattr(self.hparams, "noise_reduction", False):
                logger.debug("Wykonywanie redukcji szumów...")
                wavs = self.reduce_noise_in_audio(wavs)

            target_sampling_rate = getattr(self.hparams, "target_sampling_rate", self.hparams.sample_rate)
            if target_sampling_rate != self.hparams.sample_rate:
                logger.debug(f"Resampling z {self.hparams.sample_rate} Hz na {target_sampling_rate} Hz...")
                wavs = self.resample_audio(wavs, orig_sr=self.hparams.sample_rate, target_sr=target_sampling_rate)

            if getattr(self.hparams, "normalize_audio", False):
                logger.debug("Normalizacja sygnału audio...")
                wavs = self.normalize_audio(wavs, target_rms_db=-40.0)

            wavs_list = [
                wavs[i, 0, :wav_lens[i].item()].cpu().numpy()
                for i in range(wavs.shape[0])
            ]

            inputs = self.hparams.processor(
                wavs_list,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
            logger.debug(f"Kształt input_values: {input_values.shape}")
            logger.info(f"Shape of attention_mask: {attention_mask.shape}")

            input_lengths = (wav_lens // getattr(self.hparams, "downsample_factor", 320)).long().to(self.device)
            logger.info(f"Shape of input_lengths: {input_lengths.shape}, Values: {input_lengths}")

            precision = self.run_opts.get("precision", "fp16")
            autocast_enabled = (precision == "fp16")
            device_type = self.device.type

            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                logits = self.modules['model'](input_values, attention_mask=attention_mask).logits
                logits = logits.transpose(0, 1)

            logger.debug(f"Kształt logits: {logits.shape}")

            log_probs = F.log_softmax(logits, dim=-1)
            logger.debug(f"Kształt log_probs: {log_probs.shape}")

            return {
                "log_probs": log_probs,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask
            }

        except Exception as e:
            logger.error(f"Błąd w compute_forward: {e}", exc_info=True)
            raise
        
    def compute_objectives(self, predictions, batch, stage):
        """
        Oblicza funkcję straty CTC dla podanych predykcji i batcha.
        
        Args:
            predictions (dict): Słownik zawierający log_probs i input_lengths
            batch (dict): Batch danych wejściowych
            stage (sb.Stage): Etap treningu (TRAIN/VALID/TEST)
        
        Returns:
            torch.Tensor: Wartość funkcji straty
        """
        try:
            # Rozpakowanie predykcji
            if isinstance(predictions, dict):
                log_probs = predictions["log_probs"]
                input_lengths = predictions["input_lengths"]
            else:
                logger.error("Predictions must be a dictionary containing 'log_probs' and 'input_lengths'")
                raise ValueError("Invalid predictions format")

            # Pobranie targetów
            targets = batch['tokens_encoded'].data.to(self.device)
            target_lengths = batch['tokens_lens'].to(self.device)

            # Logowanie kształtów dla debugowania
            logger.debug(f"log_probs shape: {log_probs.shape}")
            logger.debug(f"targets shape: {targets.shape}")
            logger.debug(f"input_lengths shape: {input_lengths.shape}")
            logger.debug(f"target_lengths shape: {target_lengths.shape}")

            # Walidacja kształtów tensorów
            assert log_probs.dim() == 3, f"log_probs should have 3 dimensions, but got {log_probs.dim()}"
            assert input_lengths.dim() == 1, "input_lengths should be a tensor of shape [batch_size]"
            assert input_lengths.size(0) == log_probs.size(1), "input_lengths should match batch size in log_probs"
            
            # Poprawka: Usunięcie niepoprawnego sprawdzenia
            # assert target_lengths.size(0) == targets.size(0), "target_lengths should match batch size in targets"
            
            # Poprawione sprawdzenie: Upewnij się, że target_lengths ma rozmiar równy batch_size
            batch_size = log_probs.size(1)
            assert target_lengths.size(0) == batch_size, "target_lengths should match batch size"

            # Obliczanie straty CTC
            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='mean'
            )

            # Logowanie wartości straty
            logger.debug(f"CTC loss value: {loss.item()}")

            return loss

        except Exception as e:
            logger.error(f"Błąd podczas obliczania funkcji straty: {e}", exc_info=True)
            raise


    def train_step(self, batch):
        optimizer = self.optimizer
        optimizer.zero_grad()

        try:
            # Compute forward pass
            outputs = self.compute_forward(batch, stage=sb.Stage.TRAIN)
            
            # Compute loss
            loss = self.compute_objectives(outputs, batch, stage=sb.Stage.TRAIN)

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
    
    def fit(self, epoch_counter, train_set, valid_set=None, progressbar=True, train_loader_kwargs={}, valid_loader_kwargs={}):
        logger.info("Rozpoczęcie trenowania modelu.")
        try:
            # Pobranie batch_size i num_workers z hparams
            batch_size = getattr(self.hparams, 'batch_size', 8)
            num_workers = getattr(self.hparams, 'num_workers', 0)

            # Przygotowanie DataLoaderów z użyciem niestandardowej funkcji collate_fn
            train_loader_kwargs['collate_fn'] = collate_fn
            valid_loader_kwargs['collate_fn'] = collate_fn

            train_dataloader = sb.dataio.dataloader.make_dataloader(
                train_set,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=collate_fn
            )
            total_steps = len(train_dataloader)
            epoch_pbar = tqdm(total=len(epoch_counter), desc="Epoki", unit="epoch", colour="blue") if progressbar else None

            # Pętla po epokach
            for epoch in epoch_counter:
                logger.info(f"--- Epoka {epoch} ---")
                epoch_loss = 0.0
                batch_pbar = tqdm(total=total_steps, desc=f"Batch {epoch}", unit="batch", colour="green") if progressbar else None

                # Wywołanie on_stage_start dla fazy treningowej
                self.on_stage_start(sb.Stage.TRAIN, epoch)

                # Pętla po batchach treningowych
                for batch in train_dataloader:
                    if batch is None:
                        logger.warning("Otrzymano pusty batch. Pomijanie.")
                        continue
                    loss = self.train_step(batch)

                    epoch_loss += loss.item()
                    if batch_pbar:
                        batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
                        batch_pbar.update(1)

                # Zakończenie pętli batchy treningowych
                if batch_pbar:
                    batch_pbar.close()

                avg_loss = epoch_loss / total_steps
                logger.info(f"Epoka {epoch} zakończona. Średnia strata: {avg_loss:.4f}")

                # Wywołanie on_stage_end dla fazy treningowej
                self.on_stage_end(sb.Stage.TRAIN, stage_loss=avg_loss, epoch=epoch)

                # Walidacja
                if valid_set is not None:
                    valid_dataloader = sb.dataio.dataloader.make_dataloader(
                        valid_set,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        collate_fn=collate_fn
                    )
                    valid_loss = 0.0
                    valid_steps = len(valid_dataloader)
                    valid_pbar = tqdm(total=valid_steps, desc=f"Walidacja epoka {epoch}", unit="batch", colour="yellow") if progressbar else None

                    # Wywołanie on_stage_start dla walidacji
                    self.on_stage_start(sb.Stage.VALID, epoch)

                    # Pętla po batchach walidacyjnych
                    for batch in valid_dataloader:
                        if batch is None:
                            logger.warning("Otrzymano pusty batch podczas walidacji. Pomijanie.")
                            continue
                        predictions = self.compute_forward(batch, stage=sb.Stage.VALID)
                        loss = self.compute_objectives(predictions, batch, stage=sb.Stage.VALID)
                        valid_loss += loss.item()
                        if valid_pbar:
                            valid_pbar.set_postfix({"Valid Loss": f"{loss.item():.4f}"})
                            valid_pbar.update(1)

                    if valid_pbar:
                        valid_pbar.close()

                    avg_valid_loss = valid_loss / valid_steps
                    logger.info(f"Walidacja epoka {epoch} zakończona. Średnia strata walidacyjna: {avg_valid_loss:.4f}")

                    # Wywołanie on_stage_end dla walidacji
                    self.on_stage_end(sb.Stage.VALID, stage_loss=avg_valid_loss, epoch=epoch)

                # Zapis modelu po każdej epoce (jeśli włączony checkpointer)
                if self.checkpointer is not None:
                    if epoch % getattr(self.hparams, "save_interval", 1) == 0:

                        logger.info(f"Zapisywanie modelu po epoce {epoch}.")
                        self.checkpointer.save_checkpoint()

                # Aktualizacja paska postępu dla epok
                if epoch_pbar:
                    epoch_pbar.update(1)

            if epoch_pbar:
                epoch_pbar.close()

            logger.info("Trenowanie modelu zakończone.")

        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}", exc_info=True)
            raise

    def adjust_batch_size_based_on_lengths(self, input_lengths, max_memory_usage=15000):
        """
        Dostosowuje rozmiar batcha na podstawie średniej długości sekwencji i dostępnej pamięci.

        Args:
            input_lengths (torch.Tensor): Długości sekwencji w bieżącej partii danych.
            max_memory_usage (int): Maksymalne zużycie pamięci RAM w MB.
        """
        try:
            if not isinstance(input_lengths, torch.Tensor):
                raise TypeError("input_lengths musi być tensorem typu torch.Tensor.")
            if input_lengths.numel() == 0:
                raise ValueError("input_lengths jest pusty. Nie można dostosować batch_size.")

            avg_length = input_lengths.float().mean().item()
            precision = self.run_opts.get("precision", "fp32")
            dtype_factor = 2 if precision == "fp16" else 4

            estimated_memory = avg_length * self.hparams.batch_size * dtype_factor / (1024 ** 2)

            logger.debug(f"Średnia długość sekwencji: {avg_length}")
            logger.debug(f"Szacowane zużycie pamięci (MB): {estimated_memory:.2f}")

            if estimated_memory > max_memory_usage:
                new_batch_size = max(1, int(self.hparams.batch_size * (max_memory_usage / estimated_memory)))
                logger.warning(f"Przekroczono limit pamięci. Zmniejszanie batch_size z {self.hparams.batch_size} na {new_batch_size}.")
                self.hparams.batch_size = new_batch_size
            else:
                logger.info("Batch size jest odpowiedni.")

        except Exception as e:
            logger.error(f"Błąd podczas dostosowywania batch_size: {e}", exc_info=True)

    def reduce_noise_in_audio(self, wavs):
        try:
            # Sprawdź, czy wavs jest poprawnym tensorem
            if isinstance(wavs, torch.Tensor):
                wavs_np = wavs.squeeze(1).cpu().numpy()  # Konwersja tensora na numpy array
                logger.debug(f"Przekonwertowano wavs na numpy array: {wavs_np.shape}")
            else:
                raise ValueError(f"Niepoprawny typ danych dla wavs: {type(wavs)}. Oczekiwano torch.Tensor.")

            # Redukcja szumów
            reduced_noise = nr.reduce_noise(y=wavs_np, sr=16000)
            logger.info("Redukcja szumów zakończona.")

            # Konwersja z powrotem do tensora
            return torch.from_numpy(reduced_noise).unsqueeze(1).to(wavs.device)
        except Exception as e:
            logger.error(f"Błąd podczas redukcji szumów: {e}")
            return wavs  # W przypadku błędu zwróć oryginalne dane audio
        

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
            logger.error(f"Błąd podczas resamplingu: {e}", exc_info=True)
            return wavs  # Return the original wavs in case of error

    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            self.modules['model'].train()
        else:
            self.modules['model'].eval()

        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)
        logger.info(f"Rozpoczęcie etapu: {stage}, Epoka: {epoch}")
        logger.info(f"Hiperparametry na początku etapu: {vars(self.hparams)}")

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_stats = {"epoch": epoch, "loss": stage_loss}
        elif stage == sb.Stage.VALID:
            if hasattr(self.hparams, "lr") and hasattr(self.hparams, "lr_annealing_factor"):
                old_lr = self.hparams.lr
                self.hparams.lr *= self.hparams.lr_annealing_factor
                self.checkpointer.save_checkpoint(name=f"epoch_{epoch}_loss_{stage_loss:.4f}")

                # Aktualizacja najlepszej straty walidacyjnej
                if not hasattr(self, "best_valid_loss") or stage_loss < self.best_valid_loss:
                    self.best_valid_loss = stage_loss
                    self.best_valid_loss_epoch = epoch
                    logger.info(f"Uaktualniono najlepszą stratę walidacyjną: {self.best_valid_loss:.4f} (Epoka {self.best_valid_loss_epoch})")

                # Logowanie statystyk
                logger.info(f"Epoka {epoch}, Strata walidacyjna: {stage_loss:.4f}")
                logger.info(f"Liczba epok bez poprawy: {epoch - self.best_valid_loss_epoch}")
                logger.info(f"Wyniki po epoce {epoch}:")
                logger.info(f"  * Strata treningowa: {self.train_stats['loss']:.4f}")
                logger.info(f"  * Strata walidacyjna: {stage_loss:.4f}")
                logger.info(f"  * Aktualna learning rate: {getattr(self.hparams, 'lr', 'N/A'):.4f}")

                # Porównanie wyników
                if epoch > 0:
                    prev_train_loss = self.train_stats_history[epoch - 1]["loss"]
                    prev_valid_loss = self.valid_stats_history[epoch - 1]["loss"]
                    logger.info(f"Poprawa straty treningowej: {(prev_train_loss - self.train_stats['loss']) * 100:.2f}%")
                    logger.info(f"Poprawa straty walidacyjnej: {(prev_valid_loss - stage_loss) * 100:.2f}%")

                # Zapis wyników do historii
                self.train_stats_history[epoch] = self.train_stats
                self.valid_stats_history[epoch] = {"loss": stage_loss, "epoch": epoch}

                # Przywrócenie starej wartości learning rate
                self.hparams.lr = old_lr

    @staticmethod
    def normalize_audio(wavs, target_rms_db=-40.0):
        try:
            # Sprawdź, czy wavs jest torch.Tensor
            if not isinstance(wavs, torch.Tensor):
                logger.error(f"Oczekiwano typu torch.Tensor, ale otrzymano {type(wavs)}")
                raise TypeError("Nieprawidłowy typ danych dla wavs.")

            # Normalizacja RMS
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
            logger.error(f"Błąd podczas normalizacji audio: {e}")
            return wavs  # Zwróć oryginalne dane w przypadku błędu


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

import torchaudio
import torchaudio.transforms as T

def augment_audio(audio_path, output_path, augmentation_type=None):
    try:
        # Load the audio using torchaudio
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.unsqueeze(0)  # Add batch dimension

        if augmentation_type == "noise":
            # Add noise (as previously defined)
            noise = generate_noise(audio, noise_factor=0.05)
            augmented_audio = audio + noise
            augmented_audio = torch.clamp(augmented_audio, -1.0, 1.0)

        elif augmentation_type == "speed_pitch":
            # Speed and pitch modifications
            speed_factor = random.uniform(0.9, 1.1)
            pitch_factor = random.uniform(-2, 2)
            augmented_audio = change_speed(audio, speed_factor)
            augmented_audio = change_pitch(augmented_audio, pitch_factor)

        elif augmentation_type == "reverb":
            # Adding reverb
            augmented_audio = T.Reverberate(sample_rate)(audio)

        else:
            # No modification
            augmented_audio = audio

        # Normalize the augmented audio for better clarity
        augmented_audio = augmented_audio.squeeze(0)  # Remove batch dimension
        augmented_audio = augmented_audio / augmented_audio.abs().max()  # Normalize the signal

        # Save the augmented audio
        torchaudio.save(output_path, augmented_audio, sample_rate)
        logger.info(f"Audio file augmented and saved: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error during audio augmentation: {e}", exc_info=True)
        raise

# Generate white noise
def generate_noise(audio, noise_factor=0.05):
    """
    Generates white noise to mix with audio.

    Args:
        audio (torch.Tensor): The original audio tensor.
        noise_factor (float): The intensity of the noise relative to the original audio.

    Returns:
        torch.Tensor: Noise tensor of the same shape as the input audio.
    """
    noise = torch.randn_like(audio) * noise_factor
    return noise

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

def strip_silence(audio_segment, silence_thresh=-40.0, min_silence_len=500, keep_silence=100):
    """
    Removes silence from the audio while keeping quiet sections and pauses.
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
    Processes an audio file by stripping silence, augmenting, and normalizing it.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(upload_path)

        # Strip silence if enabled
        if trim_silence:
            audio = strip_silence(audio)

        # Normalize audio to ensure consistent volume
        audio = audio.normalize()

        # Augment audio if enabled
        if augment:
            if augment_options is None:
                augment_options = {"augmentation_type": "noise"}  # Default augmentation option

            augmented_path = augment_audio(upload_path, processed_path, **augment_options)
            return augmented_path
        else:
            # Export the processed (silence-stripped and normalized) audio
            audio.export(processed_path, format="wav")
            logger.info(f"Processed and normalized audio saved: {processed_path}")

            return processed_path
    except Exception as e:
        logger.error(f"Error during audio processing: {e}", exc_info=True)
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
        asr_brain_instance.scaler = scaler

        # Setup optimizer
        asr_brain_instance.optimizer = torch.optim.AdamW(asr_brain_instance.modules['model'].parameters(), lr=learning_rate)

        # Move model to the appropriate device
        for mod in asr_brain_instance.modules.values():
            mod.to(device)

        # Prepare data loader arguments
        train_loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "collate_fn": collate_fn,
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

def train_asr_on_voice_profile(profile_id, app_config, preprocessed_audio_files=None, transcriptions=None, num_epochs=10, batch_size=8, num_workers=0):
    """
    Trains the ASR model on a voice profile using processed audio files and transcriptions.
    The function retrieves data directly from the profile if not provided explicitly.
    If necessary, missing transcriptions are generated using Whisper.

    Args:
        profile_id (int): The ID of the voice profile.
        app_config (dict): Application configuration.
        preprocessed_audio_files (list, optional): List of paths to preprocessed audio files.
        transcriptions (list, optional): List of transcriptions corresponding to the audio files.
        num_epochs (int): Number of training epochs.
        batch_size (int): Size of the training batch.
        num_workers (int): Number of workers for data loading.
    """
    global asr_brain_instance
    profile = None

    try:
        # Pobranie profilu głosowego na podstawie ID
        profile = VoiceProfile.query.filter_by(id=profile_id).first()
        if not profile:
            raise ValueError(f"VoiceProfile with ID {profile_id} not found.")

        logger.info(f"Rozpoczęcie treningu ASR dla profilu: {profile.name} (ID: {profile.id})")

        # Pobranie listy przetworzonych plików audio
        if preprocessed_audio_files is None:
            preprocessed_audio_files = [profile.audio_file] if profile.audio_file else []

        if not preprocessed_audio_files:
            logger.warning("Lista przetworzonych plików audio jest pusta.")
            raise ValueError("Brak przetworzonych plików audio do treningu.")

        # Sprawdzenie i generowanie brakujących transkrypcji
        if transcriptions is None:
            transcriptions = [profile.transcription] if profile.transcription else []

        if len(preprocessed_audio_files) > len(transcriptions):
            logger.info("Brakuje transkrypcji dla niektórych plików audio. Rozpoczęcie automatycznej transkrypcji...")
            for i in range(len(transcriptions), len(preprocessed_audio_files)):
                audio_path = preprocessed_audio_files[i]
                transcription = transcribe_with_whisper(audio_path)
                if transcription:
                    transcriptions.append(transcription)
                    logger.info(f"Transkrypcja dla pliku {audio_path}: {transcription}")
                else:
                    raise ValueError(f"Nie udało się wygenerować transkrypcji dla pliku {audio_path}.")

        if len(preprocessed_audio_files) != len(transcriptions):
            error_msg = f"Liczba plików audio ({len(preprocessed_audio_files)}) nie zgadza się z liczbą transkrypcji ({len(transcriptions)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Preprocessed audio files list (first 5 entries): {preprocessed_audio_files[:5]}")
        logger.debug(f"Transcriptions list (first 5 entries): {transcriptions[:5]}")

        # Inicjalizacja asr_brain_instance, jeśli jeszcze nie jest załadowany
        if asr_brain_instance is None:
            logger.info("Inicjalizacja ASRBrain instance.")
            asr_brain_instance = ASRBrain.load_from_checkpoint(
                checkpoint_path=app_config['DEFAULT_CHECKPOINT'],
                run_opts={'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}
            )
            if asr_brain_instance is None:
                raise RuntimeError("Nie udało się załadować ASRBrain instance.")

        # Przygotowanie datasetu z przetworzonymi plikami audio i transkrypcjami
        try:
            train_dataset, valid_dataset = setup_dataio(asr_brain_instance, preprocessed_audio_files, transcriptions)
            if not train_dataset or not valid_dataset:
                raise ValueError("Nie udało się skonfigurować DataIO.")

            # Debugowanie - wyświetlenie informacji o przygotowanym dataset
            logger.debug(f"Training dataset size: {len(train_dataset)}")
            logger.debug(f"Validation dataset size: {len(valid_dataset)}")

        except Exception as e:
            logger.error(f"Błąd w konfiguracji DataIO: {e}", exc_info=True)
            raise RuntimeError("Nie udało się skonfigurować DataIO.")

        logger.info(f"Rozpoczęcie treningu modelu na {num_epochs} epok dla profilu ID: {profile.id}.")

        # Konfiguracja urządzenia (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        asr_brain_instance.hparams.device = device
        for mod in asr_brain_instance.modules.values():
            mod.to(device)

        # Inicjalizacja skalera dla mieszanej precyzji, jeśli jest dostępna
        if torch.cuda.is_available():
            if not hasattr(asr_brain_instance, 'scaler') or asr_brain_instance.scaler is None:
                asr_brain_instance.scaler = GradScaler()

        logger.info(f"Model ASR uruchomiony na urządzeniu: {device}")

        # Ustawienie liczby epok
        asr_brain_instance.hparams.max_epochs = num_epochs
        logger.info(f"Liczba epok dla profilu {profile.id} ustawiona na {num_epochs}")

        # Trenowanie modelu z dodanym paskiem postępu tqdm
        try:
            asr_brain_instance.fit(
                epoch_counter=range(1, num_epochs + 1),
                train_set=train_dataset,
                valid_set=valid_dataset,
                progressbar=True,  # POPRAWKA
                train_loader_kwargs={
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "collate_fn": collate_fn,
                    "pin_memory": device.type == "cuda"
                },
                valid_loader_kwargs={
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "collate_fn": collate_fn,
                    "pin_memory": device.type == "cuda"
                }
            )
            logger.info(f"Trening ASR zakończony pomyślnie dla profilu: {profile.name}")
        except Exception as e:
            logger.error(f"Błąd podczas treningu modelu ASR: {e}", exc_info=True)
            raise RuntimeError(f"Trening nie powiódł się: {str(e)}")

        # Tworzenie folderu dla profilu do zapisu modelu i konfiguracji
        profile_folder = os.path.join(app_config['ASR_MODELS_FOLDER'], f"profile_{profile.id}")
        os.makedirs(profile_folder, exist_ok=True)

        # Ścieżki do plików modelu, tokenizerów i konfiguracji
        pytorch_model_path = os.path.join(profile_folder, "pytorch_model.bin")
        config_path = os.path.join(profile_folder, "config.json")
        feature_extractor_config_path = os.path.join(profile_folder, "feature_extractor_config.json")
        preprocessor_config_path = os.path.join(profile_folder, "preprocessor_config.json")
        vocab_path = os.path.join(profile_folder, "vocab.json")
        special_tokens_map_path = os.path.join(profile_folder, "special_tokens_map.json")

        try:
            # Zapisanie modelu w formacie pytorch_model.bin
            model_state_dict = asr_brain_instance.modules['model'].state_dict()
            torch.save(model_state_dict, pytorch_model_path)
            logger.info(f"Model ASR zapisany jako {pytorch_model_path}")

            # Zapisanie tokenizer
            tokenizer = asr_brain_instance.hparams.processor.tokenizer
            tokenizer.save_pretrained(profile_folder)
            logger.info(f"Tokenizer zapisany w folderze profilu {profile_folder}")

            # Zapisanie konfiguracji modelu
            config = asr_brain_instance.modules['model'].config.to_dict()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Konfiguracja modelu zapisana jako {config_path}")

            # Zapisanie plików konfiguracyjnych związanych z ekstraktorem cech i preprocessorem
            feature_extractor_config = asr_brain_instance.hparams.processor.feature_extractor.to_dict()
            with open(feature_extractor_config_path, 'w') as f:
                json.dump(feature_extractor_config, f, indent=4)

            preprocessor_config = asr_brain_instance.hparams.processor.to_dict()
            with open(preprocessor_config_path, 'w') as f:
                json.dump(preprocessor_config, f, indent=4)

            # Zapisanie słownika (vocab.json)
            vocab = tokenizer.get_vocab()
            with open(vocab_path, 'w') as f:
                json.dump(vocab, f, indent=4)

            # Zapisanie mapy specjalnych tokenów
            special_tokens_map = tokenizer.special_tokens_map
            with open(special_tokens_map_path, 'w') as f:
                json.dump(special_tokens_map, f, indent=4)

            logger.info(f"Wszystkie pliki zostały poprawnie zapisane w {profile_folder}")

            # Dodanie modelu do bazy danych
            asr_model_entry = ASRModel(
                user_id=profile.user_id,
                voice_profile_id=profile.id,
                name=f"ASR_Model_Profile_{profile.id}",
                model_file="pytorch_model.bin",
                language=profile.language
            )
            db.session.add(asr_model_entry)
            db.session.commit()
            logger.info(f"Wpis ASRModel utworzony w bazie danych dla profilu {profile.id}")

        except Exception as e:
            db.session.rollback()
            logger.error(f"Błąd podczas zapisywania modelu ASR, tokenizera lub konfiguracji: {e}")
            raise RuntimeError(f"Zapisywanie modelu nie powiodło się: {str(e)}")

        # Ewaluacja modelu po zakończeniu treningu
        if valid_dataset:
            evaluate_metrics(asr_brain_instance, valid_dataset, profile_id, app_config)

        return "Trening zakończony pomyślnie."

    except Exception as e:
        logger.error(f"Błąd podczas treningu ASR dla profilu {profile.name if profile else 'unknown'}: {e}", exc_info=True)
        raise RuntimeError(f"Trening nie powiódł się: {str(e)}")

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
    """
    Transkrybuje dany plik audio za pomocą modelu Whisper.

    Args:
        audio_path (str): Ścieżka do pliku audio.

    Returns:
        str: Transkrypcja pliku audio lub pusty string w przypadku błędu.
    """
    if not isinstance(audio_path, str) or not audio_path.endswith(('.wav', '.mp3')):
        logger.error("Nieprawidłowy format pliku lub brak pliku audio.")
        return ""

    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="pl", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="pl", task="transcribe")

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        
        input_features = inputs.input_features.to(device)
        
        # Sprawdzanie, czy input_features są odpowiedniego typu i wymiarów
        if not isinstance(input_features, torch.Tensor):
            raise TypeError("input_features nie są typu torch.Tensor.")
        if len(input_features.shape) != 3:
            raise ValueError("input_features powinny mieć 3 wymiary.")

        # Alternatywa z encoder_outputs
        encoder_outputs = model.get_encoder()(input_features)
        # Ręczne ustawienie attention_mask na wszystkie jedynki, ponieważ pad token jest taki sam jak eos token
        attention_mask = torch.ones((input_features.shape[0], input_features.shape[2]), dtype=torch.long).to(whisper_device)
        logger.debug(f"attention_mask shape: {attention_mask.shape}")
        logger.debug(f"attention_mask: {attention_mask}")

        predicted_ids = model.generate(
            encoder_outputs=encoder_outputs,  # Przekazanie encoder_outputs jako alternatywa
            attention_mask=attention_mask,
            forced_decoder_ids=model.config.forced_decoder_ids
        )

        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return transcription.strip()

    except FileNotFoundError:
        logger.error(f"Plik audio {audio_path} nie został znaleziony.")
        return ""
    except Exception as e:
        logger.error(f"Błąd podczas transkrypcji za pomocą Whisper: {e}", exc_info=True)
        return ""

from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.lobes.models.flair.embeddings import FlairEmbeddings

def evaluate_metrics(model, valid_dataset, profile_id, app_config):
    try:
        total_loss = 0.0
        total_samples = 0

        for batch in valid_dataset:
            if batch is None:
                logger.warning("Otrzymano pusty batch podczas walidacji. Pomijanie.")
                continue

            # Zmiana sposobu odbierania wyników
            outputs = model.compute_forward(batch, stage='valid')
            log_probs = outputs['log_probs']
            input_lengths = outputs['input_lengths']
            attention_mask = outputs.get('attention_mask', None)  # Opcjonalnie, jeśli potrzebny

            # Obliczanie straty
            loss = model.compute_objectives({'log_probs': log_probs, 'input_lengths': input_lengths}, batch, stage='valid')
            total_loss += loss.item()
            total_samples += 1

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        logger.info(f"Średnia strata walidacyjna: {avg_loss:.4f}")
        return avg_loss

    except Exception as e:
        logger.error(f"Błąd podczas metrics evaluation: {e}", exc_info=True)
        raise

def prepare_input(text: str, emotion: str = 'neutral', intonation: float = 1.0) -> torch.Tensor:
    """
    Przygotowuje dane wejściowe dla modelu ASR na podstawie tekstu i opcjonalnych parametrów emocji oraz intonacji.

    Args:
        text (str): Tekst, który ma zostać przekonwertowany na mowę.
        emotion (str): Opcjonalny parametr określający emocję w generowanej mowie.
        intonation (float): Opcjonalny parametr określający intonację mowy.

    Returns:
        torch.Tensor: Tensor reprezentujący dane wejściowe do modelu ASR.
    """
    try:
        # Tokenizacja tekstu - zamiana tekstu na sekwencję tokenów (np. za pomocą istniejącego tokenizer'a)
        tokenized_text = processor.tokenizer.encode(text, return_tensors="pt")

        # Dodanie parametrów emocji i intonacji (jeśli model ich wymaga)
        # Można dodać dodatkowe informacje o emocji lub intonacji do tokenów, jeśli model to obsługuje
        if hasattr(processor.tokenizer, 'add_emotion'):
            tokenized_text = processor.tokenizer.add_emotion(tokenized_text, emotion)

        if hasattr(processor.tokenizer, 'add_intonation'):
            tokenized_text = processor.tokenizer.add_intonation(tokenized_text, intonation)

        return tokenized_text

    except Exception as e:
        logger.error(f"Błąd podczas przygotowania danych wejściowych: {e}")
        raise

def process_audio_to_mel(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    if not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist at path: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        # Load audio and process to Mel-spectrogram
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_mels=80)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_tensor = torch.tensor(mel_spectrogram_db).unsqueeze(0)
        return mel_tensor
    except Exception as e:
        logger.error(f"Error during Mel-spectrogram processing: {e}")
        raise

from safetensors.torch import load_file
def load_asrtts_model(user_id, profile_id, asr_models_folder):
    """
    Wczytuje wytrenowany model ASR/TTS z folderu użytkownika.

    Args:
        user_id (int): ID użytkownika.
        profile_id (int): ID profilu głosowego.
        asr_models_folder (str): Główny folder, w którym znajdują się modele użytkowników.

    Returns:
        tuple: (model, processor, device) - model, processor (tokenizer), oraz urządzenie.
    """
    try:
        # Folder użytkownika i profilu
        profile_folder = os.path.join(asr_models_folder, f"user_{user_id}", f"profile_{profile_id}")

        # Sprawdzanie, czy folder istnieje
        if not os.path.exists(profile_folder):
            raise FileNotFoundError(f"Nie znaleziono folderu dla użytkownika {user_id} i profilu {profile_id} w {profile_folder}.")

        # Szukanie plików modelu i konfiguracji
        model_file = None
        config_file = None
        tokenizer_files = {}

        # Przeszukiwanie folderu profilu
        for file in os.listdir(profile_folder):
            if file.endswith(".safetensors"):
                model_file = os.path.join(profile_folder, file)
            elif file.endswith("config.json"):
                config_file = os.path.join(profile_folder, file)
            elif "tokenizer" in file:
                tokenizer_files[file] = os.path.join(profile_folder, file)

        # Sprawdzanie, czy pliki modelu istnieją
        if not model_file or not config_file:
            raise FileNotFoundError(f"Brak plików modelu lub konfiguracji dla profilu {profile_id} użytkownika {user_id}.")

        # Wczytanie modelu z safetensors
        logger.info(f"Wczytywanie modelu z {model_file}")
        model_state_dict = load_file(model_file)
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=profile_folder,
            config=config_file,
            local_files_only=True
        )
        model.load_state_dict(model_state_dict)

        # Wczytanie tokenizerów
        logger.info(f"Wczytywanie tokenizerów z {config_file}")
        processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path=profile_folder,
            local_files_only=True
        )

        # Wybór urządzenia
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, processor, device

    except Exception as e:
        logger.error(f"Błąd podczas wczytywania modelu dla użytkownika {user_id} i profilu {profile_id}: {e}")
        raise RuntimeError(f"Nie udało się wczytać modelu: {e}")

async def generate_speech(text, profile, model, processor, emotion='neutral', intonation=1.0):
    """
    Generuje mowę na podstawie tekstu przy użyciu modelu ASR/TTS.

    Args:
        text (str): Tekst do syntezy.
        profile (VoiceProfile): Profil głosowy użytkownika.
        model (Wav2Vec2ForCTC): Wczytany model ASR.
        processor (Wav2Vec2Processor): Processor (tokenizer + feature extractor) modelu.
        emotion (str): Emocja do generowanej mowy (opcjonalnie).
        intonation (float): Intonacja mowy (opcjonalnie).

    Returns:
        str: Ścieżka do wygenerowanego pliku audio.
    """
    try:
        # Tokenizacja tekstu
        input_tokens = processor.tokenizer(text, return_tensors='pt').input_ids.to(model.device)

        # Generowanie mowy
        with torch.no_grad():
            output = await asyncio.to_thread(model.generate, input_tokens)

        # Konwersja predykcji na fale dźwiękowe
        audio_data = await convert_predictions_to_waveform(output)

        # Zapisanie fali dźwiękowej do pliku
        return await save_audio_to_file(audio_data)

    except Exception as e:
        logger.error(f"Błąd podczas generowania mowy: {e}", exc_info=True)
        raise


def convert_predictions_to_waveform(predictions: torch.Tensor) -> np.ndarray:
    """
    Konwertuje predykcje modelu TTS bezpośrednio na fale dźwiękowe.

    Args:
        predictions (torch.Tensor): Wyniki modelu TTS, które mogą reprezentować bezpośrednio dane audio.

    Returns:
        np.ndarray: Dane audio jako fala dźwiękowa gotowa do zapisu.
    """
    try:
        if predictions.dim() == 2:  # Zakładamy, że predykcje to dane audio (np. generowane przez WaveRNN lub HiFi-GAN)
            predictions = predictions.squeeze(0)

        # Jeśli model generuje bezpośrednio falę dźwiękową, możemy bezpośrednio zwrócić predykcje
        audio_waveform = predictions.cpu().numpy()

        return audio_waveform

    except Exception as e:
        logger.error(f"Błąd podczas konwersji predykcji na dane audio: {e}")
        raise


async def save_audio_to_file(audio_data: np.ndarray, sample_rate: int = 22050) -> str:
    """
    Zapisuje dane audio do pliku w formacie .wav.

    Args:
        audio_data (np.ndarray): Wygenerowane dane audio (fala dźwiękowa).
        sample_rate (int): Częstotliwość próbkowania (domyślnie 22050 Hz).

    Returns:
        str: Ścieżka do zapisanego pliku audio.
    """
    try:
        # Tworzenie unikalnej nazwy pliku .wav
        output_filename = f"generated_{uuid.uuid4().hex}.wav"
        save_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

        # Zapis pliku audio w formacie .wav
        await asyncio.to_thread(sf.write, save_path, audio_data, sample_rate)

        logger.info(f"Plik audio został zapisany: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania pliku audio: {e}")
        raise





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

def handle_ajax_response(success, message, profile_id=None):
    """Helper function to handle AJAX responses"""
    response = {"success": success, "message": message}
    if profile_id:
        response["profile_id"] = profile_id
    return jsonify(response)

@app.route('/upload_voice', methods=['GET', 'POST'])
@jwt_required()
def upload_voice():
    user_id = get_jwt_identity()
    if request.method == 'POST':
        # Sprawdzenie, czy plik jest w żądaniu
        if 'file' not in request.files:
            message = "Brak pliku w żądaniu."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": False, "message": message}), 400
            flash(message, 'danger')
            return redirect(request.url)

        file = request.files['file']
        name = request.form.get('name', '').strip() or file.filename
        language = request.form.get('language', 'pl').strip()

        if file.filename == '':
            message = "Nie wybrano pliku."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": False, "message": message}), 400
            flash(message, 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            message = "Nieobsługiwany format pliku audio."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
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
            else:
                file.save(upload_path)
            logger.info(f"Plik audio został przesłany: {upload_path}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania lub konwertowania pliku: {e}")
            message = "Nie udało się zapisać lub przekonwertować pliku."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": False, "message": message}), 500
            flash(message, 'danger')
            return redirect(request.url)

        try:
            # Pobranie wybranych opcji augmentacji
            augment_options = request.form.getlist('augment_options')

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

            message = "Profil głosowy został utworzony, przetworzony i transkrybowany."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return handle_ajax_response(True, message, voice_profile.id)

            flash(message, 'success')
            return redirect(url_for('analyze_audio', profile_id=voice_profile.id))

        except Exception as e:
            db.session.rollback()
            logger.error(f"Błąd podczas przetwarzania audio: {e}")
            message = "Wystąpił błąd podczas przetwarzania audio."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": False, "message": message}), 500
            flash(message, 'danger')
            return redirect(request.url)

    # Żądanie GET
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
        flash("Profil głosowy nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    augmented_audio_paths = []
    augmented_transcriptions = []

    try:
        for i in range(3):  # Generowanie trzech różnych augmentacji
            unique_output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"augmented_{uuid.uuid4().hex}.wav")
            augmented_path = augment_audio(audio_path, unique_output_path, augmentation_type="noise")
            if augmented_path:
                augmented_audio_paths.append(augmented_path)
                augmented_transcriptions.append(profile.transcription)
    except Exception as e:
        logger.error(f"Nie udało się przeprowadzić augmentacji audio: {e}")
        flash("Błąd podczas augmentacji plików audio.", 'danger')
        return redirect(url_for('profile'))

    all_audio_files = [audio_path] + augmented_audio_paths
    all_transcriptions = [profile.transcription] + augmented_transcriptions

    with asr_model_lock:
        if profile_id in training_progress:
            flash("Trening dla tego profilu jest już w toku.", 'warning')
            return jsonify({"error": "Trening już w toku."}), 400

        training_progress[profile_id] = {
            "status": "Rozpoczęcie treningu...",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 10,
            "time_elapsed": 0,
            "is_paused": False,
            "metrics": {},
            "loss": 0.0  # Domyślna wartość dla loss
        }
        logger.debug(f"Zainicjowano training_progress dla profilu {profile_id}: {training_progress[profile_id]}")

    def train():
        with app.app_context():
            start_time = time.time()
            training_semaphore.acquire()
            try:
                update_training_progress(profile_id, status="Ładowanie modelu...", progress=5)

                # Wywołanie funkcji treningowej, przekazanie list plików audio i transkrypcji
                train_asr_on_voice_profile(profile_id, app.config, all_audio_files, all_transcriptions, num_epochs=10, batch_size=8, num_workers=0)

                update_training_progress(profile_id, status="Trening zakończony pomyślnie.", progress=100, current_epoch=10)
                logger.info(f"Trening ASR dla profilu ID {profile_id} zakończony pomyślnie.")
            except Exception as e:
                logger.error(f"Błąd podczas treningu ASR dla profilu ID {profile_id}: {e}")
                update_training_progress(profile_id, status=f"Błąd: {str(e)}", progress=0)
            finally:
                training_semaphore.release()
                with asr_model_lock:
                    if profile_id in training_progress:
                        del training_progress[profile_id]

    try:
        executor.submit(train)
        logger.info(f"Rozpoczęto trening dla profilu ID {profile_id}.")
        flash("Trening został rozpoczęty.", 'success')
    except Exception as e:
        logger.error(f"Nie udało się rozpocząć treningu: {e}")
        flash("Nie udało się rozpocząć treningu.", 'danger')
        return jsonify({"error": "Nie udało się rozpocząć treningu."}), 500

    return jsonify({"message": "Trening został rozpoczęty."}), 200


@app.route('/training_status/<int:profile_id>', methods=['GET'])
@jwt_required()
def training_status(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        return jsonify({"error": "Profil głosowy nie został znaleziony."}), 404

    if request.headers.get('Accept') == 'text/event-stream':
        def generate():
            while profile_id in training_progress:
                progress_data = training_progress.get(profile_id, {
                    "status": "Nie rozpoczęto",
                    "progress": 0,
                    "current_epoch": 0,
                    "total_epochs": 10,
                    "time_elapsed": 0,
                    "is_paused": False,
                    "loss": 0.0  # Domyślna wartość dla loss
                })
                yield f"data: {json.dumps(progress_data)}\n\n"
                time.sleep(1)
            yield f"data: {json.dumps({'status': 'Zakończono', 'progress': 100, 'current_epoch': 10, 'total_epochs': 10, 'time_elapsed': 0, 'is_paused': False, 'loss': 0.0})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        progress = training_progress.get(profile_id, {
            "status": "Nie rozpoczęto",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 10,
            "time_elapsed": 0,
            "is_paused": False,
            "loss": 0.0  # Domyślna wartość dla loss
        })
        return jsonify(progress)

# ------------------- Helper Functions for Training Progress -------------------

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
                "loss": loss or 0.0  # Zawsze zapewniaj, że loss jest liczbą
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
                training_progress[profile_id]["loss"] = loss  # Aktualizacja straty

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
        try:
            text = request.form.get('text', '').strip()
            voice_id = request.form.get('voice_id')
            emotion = request.form.get('emotion', 'neutral')

            try:
                intonation = float(request.form.get('intonation', 1.0))
            except ValueError:
                intonation = 1.0  # Default value if conversion fails

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

            # Wczytanie wytrenowanego modelu ASR/TTS
            asr_models_folder = app.config['ASR_MODELS_FOLDER']
            asr_model, processor, device = load_asrtts_model(user_id, profile.id, asr_models_folder)

            # Generowanie mowy w tle
            output_filename = asyncio.run(generate_speech(text, profile, processor, emotion, intonation))

            flash("Mowa została wygenerowana i jest dostępna do pobrania.", 'success')
            return send_from_directory(app.config['GENERATED_FOLDER'], output_filename, as_attachment=True)

        except Exception as e:
            logger.error(f"Błąd podczas generowania mowy: {e}", exc_info=True)
            flash(f"Błąd podczas generowania mowy: {e}", 'danger')
            return redirect(request.url)

    # GET request - wyświetlenie formularza
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
        flash("Profil g³osowy nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie zosta³ znaleziony.", 'danger')
        return redirect(url_for('profile'))

    try:
        # Pobierz instancjê ASRBrain, jeli potrzebna w dalszej czêci
        asr_brain = get_asr_brain()

        # Poprawione wywo³anie funkcji bez przekazywania asr_brain jako argumentu n_mels
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
            flash("Wyst¹pi³ b³¹d podczas analizy audio.", 'danger')
            return redirect(url_for('profile'))

        suitability = evaluate_audio_suitability(processing_info)

        audio_signal = audio_signal_function(audio_path)
        if audio_signal:
            processing_info['audio_signal'] = audio_signal
        else:
            processing_info['audio_signal'] = []
            logger.warning(f"Sygna³ audio nie zosta³ poprawnie za³adowany dla pliku: {audio_path}")

        return render_template('audio_analysis.html',
                               profile=profile,
                               processing_info=processing_info,
                               mel_spectrogram=mel_tensor.tolist(),
                               suitability=suitability,
                               save_status=save_status,
                               transcription=profile.transcription)
    except Exception as e:
        logger.error(f"Error during audio analysis: {e}", exc_info=True)
        flash(f"Wyst¹pi³ b³¹d podczas analizy audio: {str(e)}", 'danger')
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
            # Inicjalizacja bazy danych
            db.create_all()
            logger.info("Starting to load ASR model...")
            
            # Ładowanie ASR Brain (Singleton)
            asr_brain_instance = get_asr_brain()
            
            # Inicjalizacja modelu Whisper przy starcie aplikacji
            initialize_whisper_model()

            if asr_brain_instance:
                logger.info("ASR model loaded successfully.")
            else:
                logger.critical("Failed to load ASR model. Exiting application.")
                sys.exit(1)

        logger.info("Flask application has been started.")

        # Uruchamianie wątku monitorowania pamięci
        monitor_thread = threading.Thread(target=background_system_memory_monitor, args=(240,), daemon=True)
        monitor_thread.start()
        logger.info("System memory monitor thread started.")

        # Uruchamianie serwera Flask
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)

    except KeyboardInterrupt:
        logger.info("Application has been stopped by the user.")
    except Exception as e:
        logger.critical(f"Failed to start the application: {e}", exc_info=True)
    finally:
        logger.info("Application cleanup process completed.")
