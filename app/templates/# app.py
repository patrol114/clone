# app.py
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
from speechbrain.utils.data_pipeline import takes, provides, DataPipeline
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
# Inicjalizacja semafora z liczbą dostępnych rdzeni CPU
training_semaphore = threading.Semaphore(1)

# Ustawienia PyTorch dla optymalizacji CPU
torch.set_num_threads(os.cpu_count()) # U?yj wszystkich rdzeni CPU
torch.backends.cudnn.benchmark = True  # Optymalizuj dla sprz?tu
torch.set_num_interop_threads(os.cpu_count())
import magic
import json
import asyncio

# Importy Wav2Vec2 z Hugging Face
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Używamy klas z transformers zamiast z SpeechBrain
from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.dataio.batch import PaddedBatch

from speechbrain.utils.autocast import fwd_default_precision
import speechbrain as sb
from speechbrain.core import Brain
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from concurrent.futures import ThreadPoolExecutor, as_completed

from speechbrain.nnet.losses import ctc_loss as compute_cost
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
                    "max_epochs": 10
                },
                run_opts={
                    "device": device.type,
                    "precision": "bf16" if device.type == "cpu" else "fp16"
                }
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

# Usunięcie domyślnych handlerów
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
        # Formatowanie wiadomości z użyciem koloru
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

def background_system_memory_monitor(interval=120):
    """
    Funkcja monitorująca zużycie pamięci RAM, CPU i GPU w tle. Zamyka proces tylko wtedy, gdy zużycie RAM przekroczy 100%.

    Args:
        interval (int): Czas oczekiwania między kolejnymi pomiarami w sekundach.
    """
    while True:
        try:
            # Monitorowanie pamięci GPU
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)  # Zużycie GPU w MB
            gpu_total = info_gpu.total // (1024 ** 2)  # Całkowita pamięć GPU w MB
            gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Nie udało się uzyskać informacji o pamięci GPU: {e}")
            gpu_used, gpu_total, gpu_percent = 0, 0, 0

        try:
            # Monitorowanie pamięci RAM i CPU
            ram_info = psutil.virtual_memory()
            ram_used = ram_info.used // (1024 ** 2)  # Zużycie RAM w MB
            ram_total = ram_info.total // (1024 ** 2)  # Całkowita pamięć RAM w MB
            ram_percent = ram_info.percent  # Zużycie RAM w procentach

            cpu_percent = psutil.cpu_percent(interval=1)  # Zużycie CPU w procentach
        except Exception as e:
            logger.warning(f"Nie udało się uzyskać informacji o pamięci RAM/CPU: {e}")
            ram_used, ram_total, ram_percent, cpu_percent = 0, 0, 0, 0

        # Logowanie informacji o systemie
        logger.info(f"Zużycie Systemowe - GPU: {gpu_used} MB / {gpu_total} MB ({gpu_percent:.2f}%) | RAM: {ram_used} MB / {ram_total} MB ({ram_percent}%) | CPU: {cpu_percent}%")

        # Sprawdzenie, czy zużycie RAM osiągnęło 100%
        if ram_percent >= 100:
            logger.critical(f"Zużycie RAM przekroczyło 100% ({ram_percent}%). Zamykanie aplikacji.")
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
    transcription = db.Column(db.Text, nullable=True)  # Nowa kolumna na transkrypcję
    sample_rate = db.Column(db.Integer, nullable=True)  # Dodanie częstotliwości próbkowania
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

def reduce_noise_sample(args):
    """
    Funkcja pomocnicza do redukcji szumów dla pojedynczej próbki audio.

    Args:
        args (tuple): Krotka zawierająca próbkę audio, jej indeks oraz częstotliwość próbkowania.

    Returns:
        tuple: Indeks próbki oraz przetworzona próbka audio.
    """
    sample, sample_idx, sample_rate = args
    try:
        # Redukcja szumów tylko jeśli próbka jest wystarczająco głośna
        if np.max(np.abs(sample)) > 1e-5:
            max_length = 16000  # 1 sekunda przy próbkowaniu 16kHz
            segments = []

            # Podziel próbkę na segmenty, jeśli jej długość przekracza max_length
            for j in range(0, len(sample), max_length):
                segment = sample[j:j + max_length]
                reduced_segment = nr.reduce_noise(y=segment, sr=sample_rate)
                segments.append(reduced_segment)

            # Sklej zredukowane segmenty w jedną całość
            reduced_sample = np.concatenate(segments, axis=0)
            return (sample_idx, reduced_sample)
        else:
            logger.warning(f"Sygnał zbyt cichy, pomijanie redukcji szumów dla próbki {sample_idx}")
            return (sample_idx, sample)
    except Exception as e:
        logger.error(f"Błąd podczas redukcji szumów dla próbki {sample_idx}: {e}", exc_info=True)
        return (sample_idx, sample)

# ------------------- Definicja Klasy ASRBrain -------------------
class ASRBrain(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts=run_opts, checkpointer=checkpointer)
        self.device = torch.device(getattr(self.hparams, 'device', 'cpu'))
        logger.debug(f"ASRBrain urządzenie: {self.device} (Typ: {type(self.device)})")

        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)

        self.hparams.batch_size = getattr(hparams, 'batch_size', 8)

        self.enable_gradient_checkpointing = getattr(self.hparams, "enable_gradient_checkpointing", False)
        if self.enable_gradient_checkpointing:
            if hasattr(self.modules['model'], 'config'):
                self.modules['model'].config.gradient_checkpointing = True
                logger.info("Gradient checkpointing włączony.")
            else:
                logger.warning("Model nie posiada atrybutu 'config'. Gradient checkpointing nie może być włączony.")

        if not torch.cuda.is_available() and self.enable_gradient_checkpointing:
            logger.warning("Gradient checkpointing wymaga GPU. Wyłączenie gradient_checkpointing.")
            self.enable_gradient_checkpointing = False

        self.compile_using_fullgraph = getattr(run_opts, 'compile_using_fullgraph', False)
        self.compile_using_dynamic_shape_tracing = getattr(run_opts, 'compile_using_dynamic_shape_tracing', False)

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

    def compute_objectives(self, predictions, batch, stage):
        """
        Definiuje funkcję straty dla modelu ASR przy użyciu CTC Loss.

        Args:
            predictions (torch.Tensor): Logity wyjściowe modelu o kształcie [batch_size, max_time, num_classes].
            batch (Batch): Aktualna partia danych.
            stage (sb.Stage): Aktualny etap (TRAIN, VALID, TEST).

        Returns:
            torch.Tensor: Obliczona strata.
        """
        # Pobierz transkrypcje docelowe
        tokens_eos, tokens_eos_lens = batch["tokens_eos"], batch["tokens_eos_lens"]
        
        # Przekształć przewidywania na log-probabilities
        log_probs = torch.log_softmax(predictions, dim=-1)

        # Oblicz straty CTC
        loss = self.ctc_loss(
            log_probs=log_probs,
            targets=tokens_eos,
            input_lens=torch.tensor([log_probs.size(1)] * log_probs.size(0)),
            target_lens=tokens_eos_lens,
            blank_index=self.blank_index,
            reduction='mean'
        )

        return loss

    def configure_optimizers(self):
        """
        Konfiguracja optymalizatorów dla modułów.
        """
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules['model'].parameters())
            logger.info("Optymalizator skonfigurowany dla modelu.")
        else:
            logger.error("Brak klasy optymalizatora. Nie można skonfigurować optymalizatora.")

    def fit(self, epoch_counter, train_set, valid_set=None, progressbar=True, train_loader_kwargs={}, valid_loader_kwargs={}):
        """
        Nadpisana metoda fit() zapewniająca kontrolę nad procesem trenowania w kontekście wieloprocesowym.

        Args:
            epoch_counter (iterable): Zakres epok (np. range(1, 11)).
            train_set (Dataset): Zbiór danych treningowych.
            valid_set (Dataset, optional): Zbiór walidacyjny. Domyślnie None.
            progressbar (bool, optional): Czy pokazywać pasek postępu. Domyślnie True.
            train_loader_kwargs (dict, optional): Argumenty dla ładowania danych treningowych.
            valid_loader_kwargs (dict, optional): Argumenty dla ładowania danych walidacyjnych.
        """
        logger.info("Rozpoczęcie trenowania modelu.")
        try:
            # Użycie tqdm do pokazywania paska postępu
            if progressbar:
                # Tworzymy DataLoader dla zestawu treningowego
                train_dataloader = sb.dataio.dataloader.make_dataloader(train_set, **train_loader_kwargs)
                total_steps = len(train_dataloader)

                # Iterujemy po epokach
                with tqdm(total=len(epoch_counter), desc="Epoki treningowe", unit="epoch") as epoch_pbar:
                    for epoch in epoch_counter:
                        logger.info(f"Rozpoczęcie epoki {epoch}")

                        # Tworzymy DataLoader dla batchów w każdej epoce
                        with tqdm(total=total_steps, desc=f"Batch epoka {epoch}", unit="batch") as batch_pbar:
                            # Uruchomienie oryginalnej metody fit dla każdej epoki z aktualizacją paska postępu batchów
                            super().fit([epoch], train_set, valid_set, progressbar=batch_pbar, train_loader_kwargs=train_loader_kwargs, valid_loader_kwargs=valid_loader_kwargs)
                            
                            # Aktualizacja paska postępu dla każdej epoki
                            epoch_pbar.update(1)

                logger.info("Trenowanie modelu zakończone.")
            else:
                # Jeśli pasek postępu jest wyłączony, po prostu uruchamiamy super().fit
                super().fit(epoch_counter, train_set, valid_set, progressbar=progressbar, train_loader_kwargs=train_loader_kwargs, valid_loader_kwargs=valid_loader_kwargs)
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("Błąd CUDA: Brak pamięci. Zmniejsz batch_size i spróbuj ponownie.")
            else:
                logger.error(f"RuntimeError podczas trenowania: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}", exc_info=True)

    def monitor_memory_usage(self):
        """
        Monitoruje zużycie pamięci GPU, RAM i CPU.
        Zwraca informacje o zużyciu pamięci w MB.
        """
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)  # Zużycie GPU w MB
            gpu_total = info_gpu.total // (1024 ** 2)  # Całkowita pamięć GPU w MB
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Nie udało się uzyskać informacji o pamięci GPU: {e}")
            gpu_used, gpu_total = 0, 0

        ram_info = psutil.virtual_memory()
        ram_used = ram_info.used // (1024 ** 2)  # Zużycie RAM w MB
        ram_total = ram_info.total // (1024 ** 2)  # Całkowita pamięć RAM w MB

        cpu_percent = psutil.cpu_percent(interval=1)  # Zużycie CPU w procentach

        logger.info(f"Zużycie GPU: {gpu_used} MB / {gpu_total} MB")
        logger.info(f"Zużycie RAM: {ram_used} MB / {ram_total} MB")
        logger.info(f"Zużycie CPU: {cpu_percent}%")

        return {
            "gpu_used": gpu_used,
            "gpu_total": gpu_total,
            "ram_used": ram_used,
            "ram_total": ram_total,
            "cpu_percent": cpu_percent
        }

    def adjust_model_parameters(self, memory_info, max_ram=16000, max_gpu=12000):
        """
        Dostosowuje parametry modelu na podstawie dostępnej pamięci RAM i GPU.
        """
        if memory_info['ram_used'] > max_ram or memory_info['gpu_used'] > max_gpu:
            logger.warning("Zużycie pamięci przekracza limit, zmniejszamy rozmiar batcha.")
            current_batch_size = max(1, getattr(self.hparams, "batch_size", 16) // 2)

            self.hparams["batch_size"] = current_batch_size
            logger.info(f"Nowy rozmiar batcha: {current_batch_size}")

            if getattr(self.hparams, "use_augmentation", False):
                self.hparams["use_augmentation"] = False
                logger.info("Wyłączono augmentację danych audio.")
        else:
            logger.info("Zużycie pamięci w normie. Parametry modelu pozostają bez zmian.")

    def reduce_noise(self, wavs):
        """
        Redukuje szumy w tensorach audio za pomocą wieloprocesowości.

        Args:
            wavs (torch.Tensor): Tensor audio o kształcie [batch_size, channels, samples].

        Returns:
            torch.Tensor: Tensor audio po redukcji szumów.
        """
        try:
            wavs_np = wavs.squeeze(1).cpu().numpy()
            sample_rate = self.hparams.sample_rate

            # Przygotowanie argumentów dla każdego procesu
            args = [(wavs_np[i], i, sample_rate) for i in range(wavs_np.shape[0])]

            # Ograniczenie liczby procesów do mniejszej wartości, np. 4
            max_workers = min(4, cpu_count())
            with Pool(processes=max_workers) as pool:
                results = pool.map(reduce_noise_sample, args)

            # Przypisanie przetworzonych próbek do wavs_np
            for idx, reduced_sample in results:
                wavs_np[idx, :] = reduced_sample

            # Konwersja przetworzonego numpy array z powrotem do tensorów PyTorch
            wavs_reduced = torch.from_numpy(wavs_np).unsqueeze(1).to(wavs.device)
            return wavs_reduced

        except Exception as e:
            logger.error(f"Błąd podczas redukcji szumów: {e}", exc_info=True)
            return wavs
    
    def adjust_batch_size_based_on_lengths(self, input_lengths, max_memory_usage=14000):
        """
        Dostosowuje rozmiar batcha na podstawie średniej długości sekwencji i dostępnej pamięci.
        
        Args:
            input_lengths (torch.Tensor): Długości sekwencji w bieżącej partii danych.
            max_memory_usage (int): Maksymalne zużycie pamięci RAM w MB.
        """
        try:
            avg_length = input_lengths.float().mean().item()
            estimated_memory = avg_length * self.hparams.batch_size * 4 / (1024 ** 2)  # Przybliżona pamięć w MB
            
            if estimated_memory > max_memory_usage:
                new_batch_size = max(1, int(self.hparams.batch_size * (max_memory_usage / estimated_memory)))
                logger.warning(f"Przekroczono limit pamięci. Zmniejszanie batch_size z {self.hparams.batch_size} na {new_batch_size}.")
                self.hparams.batch_size = new_batch_size
            else:
                logger.info("Batch size jest odpowiedni.")
        except Exception as e:
            logger.error(f"Błąd podczas dostosowywania batch_size: {e}", exc_info=True)
                
    def compute_forward(self, batch, stage):
        try:
            memory_info = self.monitor_memory_usage()
            self.adjust_model_parameters(memory_info)

            logger.debug("Wejście do compute_forward...")
            wavs, wav_lens = batch['inputs'], batch['input_lengths']
            logger.debug(f"Kształt wavs: {wavs.shape}, wav_lens: {wav_lens}")

            # Pasek postępu dla każdej partii w compute_forward
            with tqdm(total=wavs.shape[0], desc="Przetwarzanie batcha", unit="sample") as pbar:
                # Sprawdzenie liczby kanałów w audio i konwersja na mono
                if wavs.dim() == 3 and wavs.shape[1] > 1:
                    logger.info(f"Wykryto {wavs.shape[1]} kanałów, konwertowanie na mono...")
                    wavs = wavs.mean(dim=1, keepdim=True)
                    logger.debug(f"Konwertowano na mono, nowy kształt: {wavs.shape}")

                elif wavs.dim() == 2:
                    wavs = wavs.unsqueeze(1)
                    logger.debug("Dane audio są już mono, dodano wymiar kanału.")

                # Ensure minimum input size (1 sekunda przy 16kHz)
                MIN_INPUT_SIZE = 16000  
                current_length = wavs.shape[2]
                if current_length < MIN_INPUT_SIZE:
                    padding_size = MIN_INPUT_SIZE - current_length
                    wavs = torch.nn.functional.pad(wavs, (0, padding_size), "constant", 0)
                    logger.info(f"Sygnał był zbyt krótki. Dodano padding, nowy kształt: {wavs.shape}")

                # Redukcja szumów (jeśli włączona)
                if getattr(self.hparams, "noise_reduction", False):
                    logger.debug("Wykonywanie redukcji szumów...")
                    wavs = self.reduce_noise(wavs)

                # Ensure the correct sample rate via resampling if needed
                target_sampling_rate = getattr(self.hparams, "target_sampling_rate", self.hparams.sample_rate)
                if target_sampling_rate != self.hparams.sample_rate:
                    logger.debug(f"Resampling z {self.hparams.sample_rate} Hz na {target_sampling_rate} Hz...")
                    wavs = self.resample_audio(wavs, orig_sr=self.hparams.sample_rate, target_sr=target_sampling_rate)

                # Normalizacja audio (jeśli włączona)
                if getattr(self.hparams, "normalize_audio", False):
                    logger.debug("Normalizacja sygnału audio...")
                    wavs = self.normalize_audio(wavs, target_rms_db=-40.0)

                # Convert audio tensors to lists for further processing
                wavs_list = [wavs[i, 0, :int(wav_lens[i] * wavs.shape[2])].cpu().numpy() for i in range(wavs.shape[0])]
                
                # Process the audio data using the processor
                inputs = self.hparams.processor(
                    wavs_list,
                    sampling_rate=target_sampling_rate,
                    return_tensors="pt",
                    padding=True
                )

                input_values = inputs.input_values.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                logger.debug(f"Kształt input_values: {input_values.shape}")

                # Compute logits using the model (with gradient enabled for training)
                with torch.set_grad_enabled(stage == sb.Stage.TRAIN):
                    logits = self.modules['model'](input_values, attention_mask=attention_mask).logits
                logger.debug(f"Kształt logits: {logits.shape}")

                # Compute input lengths based on attention mask
                input_lengths = attention_mask.sum(-1).long()
                logger.debug(f"Długości wejściowe: {input_lengths}")

                pbar.update(1)  # Aktualizowanie paska dla każdego przetworzonego sample

                return logits, input_lengths

        except Exception as e:
            logger.error(f"Błąd w compute_forward: {e}", exc_info=True)
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

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            avg_loss = stage_loss["loss"]
            logger.info(f"Epoka: {epoch} | Strata: {avg_loss:.4f}")
        else:
            avg_loss = stage_loss["loss"]
            wer = self.wer_metric.summarize("WER")
            cer = self.cer_metric.summarize("CER")
            logger.info(f"Zakończenie etapu: {stage}, Strata: {avg_loss:.4f} | WER: {wer:.2f}% | CER: {cer:.2f}%")

    @staticmethod
    def normalize_audio(wavs, target_rms_db=None):
        """
        Normalizuje sygnał audio do określonego RMS w decybelach lub do maksymalnej wartości.
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
            logger.error(f"Błąd podczas normalizacji audio: {e}")
            return wavs

# ------------------- Model Configuration -------------------

BASE_ASR_MODEL_ID = "badrex/xlsr-polish"  # Stała zmienna dla modelu bazowego

# ThreadPoolExecutor for handling asynchronous training tasks
executor = ThreadPoolExecutor(max_workers=1)

# Thread lock for thread-safe operations on model progress and cache
asr_model_lock = threading.Lock()

# Cache to hold loaded models
asr_model_cache = {}

# Global progress tracking for model loading
model_loading_progress = {}

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
    
    with asr_model_lock:
        # Check if the model is already loaded and cached
        if model_id in asr_model_cache:
            logger.info(f"Model '{model_id}' found in cache, reusing...")
            return asr_model_cache[model_id]

    try:
        logger.info(f"Loading ASR model '{model_id}'...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=cache_dir)

        if not model or not processor:
            raise RuntimeError("Failed to load ASR model or processor.")

        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
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
    
def initialize_asr_brain(model_id=BASE_ASR_MODEL_ID, cache_dir=".cache"):
    try:
        model, processor, device = load_asr_model(model_id, cache_dir)
        if model is None or processor is None or device is None:
            raise RuntimeError("ASR model or processor was not loaded.")

        modules = {
            "model": model
        }
        modules["model"].to(device)

        opt_class = torch.optim.Adam

        # Define hparams as a dictionary
        hparams_dict = {
            "batch_size": 8,
            "sample_rate": 16000,
            "processor": processor,
            "convert_to_mono": "average",
            "noise_reduction": True,
            "target_sampling_rate": 16000,
            "normalize_audio": True,
            "use_augmentation": True,
            "char_list": list(processor.tokenizer.get_vocab().keys()) if hasattr(processor.tokenizer, 'get_vocab') else []
        }

        # Convert hparams to SimpleNamespace
        hparams = types.SimpleNamespace(**hparams_dict)

        # Keep run_opts as a dictionary
        run_opts = {
            "compile_using_fullgraph": False,
            "compile_using_dynamic_shape_tracing": False,
            "precision": "bf16" if device.type == "cpu" else "fp16",
            "device": device.type
        }

        if run_opts.get("compile_using_fullgraph") or run_opts.get("compile_using_dynamic_shape_tracing"):
            try:
                torch.compile(model, fullgraph=run_opts["compile_using_fullgraph"], dynamic=run_opts["compile_using_dynamic_shape_tracing"])
                logger.info("Model compiled using dynamic options.")
            except Exception as e:
                logger.error(f"Error during model compilation: {e}", exc_info=True)

        asr_brain = ASRBrain(modules=modules, opt_class=opt_class, hparams=hparams, run_opts=run_opts)

        logger.info("ASRBrain initialized successfully.")
        return asr_brain

    except Exception as e:
        logger.error(f"Error during ASRBrain initialization: {e}", exc_info=True)
        return None

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

#---
@takes("audio_path")
@provides("sig")
def audio_pipeline(audio_path):
    """
    Extracts the audio signal from the audio path and returns the waveform (sig).
    """
    try:
        # Load the audio using librosa or torchaudio
        signal, sample_rate = librosa.load(audio_path, sr=16000, mono=True)  # Ensure audio is mono and 16kHz
        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dim
        logger.debug(f"Audio waveform shape: {waveform.shape}")
        return waveform
    except Exception as e:
        logger.error(f"Error in audio_pipeline: {e}", exc_info=True)
        raise


@takes("transcription")
@provides("tokens_encoded", "tokens_lens")
def token_pipeline(transcription, asr_brain):
    try:
        tokens_encoded = asr_brain.hparams.processor.tokenizer(
            transcription,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        ).input_ids  # [1, seq_len] or [1, 1, seq_len]

        # Ensure tokens_encoded has dimensions [batch_size, seq_len]
        if tokens_encoded.dim() == 3:
            tokens_encoded = tokens_encoded.squeeze(1)  # [1, seq_len]
            logger.debug(f"tokens_encoded after squeeze: {tokens_encoded.shape}")

        # Ensure tokens_encoded now has 2 dimensions
        if tokens_encoded.dim() != 2:
            logger.error(f"tokens_encoded has incorrect dimension after correction: {tokens_encoded.dim()}")
            raise ValueError("tokens_encoded must have exactly 2 dimensions [batch_size, seq_len].")

        tokens_lens = torch.tensor([tokens_encoded.size(1)], dtype=torch.long)  # [batch_size]
        logger.debug(f"tokens_encoded: {tokens_encoded}, tokens_lens: {tokens_lens}")
        return tokens_encoded, tokens_lens
    except Exception as e:
        logger.error(f"Error in token_pipeline: {e}", exc_info=True)
        raise
    
def prepare_training_data(audio_paths, transcriptions, split_ratio=0.8):
    """
    Prepares training and validation datasets for ASR model training.

    Args:
        audio_paths (list): List of paths to audio files.
        transcriptions (list): List of corresponding transcriptions.
        split_ratio (float): Ratio of data to be used for training (rest will be used for validation).

    Returns:
        tuple: (train_set, valid_set) DynamicItemDataset instances or (None, None) in case of error.
    """
    try:
        # Validate input lengths
        if len(audio_paths) != len(transcriptions):
            raise ValueError("Mismatch between the number of audio files and transcriptions.")

        data = {}
        for idx, (audio_path, transcription) in enumerate(zip(audio_paths, transcriptions)):
            sample_id = f"sample_{idx}"
            # Load the audio signal
            signal = read_audio(audio_path)  # Zakładam, że read_audio zwraca torch.Tensor

            if not isinstance(signal, torch.Tensor):
                signal = torch.tensor(signal, dtype=torch.float32).clone().detach()
            else:
                signal = signal.clone().detach()

            # Dodaj transkrypcję jako osobny klucz, który będzie przetwarzany przez token_pipeline
            data[sample_id] = {
                'sig': signal,
                'transcription': transcription
            }

        # Split data into training and validation sets
        split = int(split_ratio * len(data))
        data_items = list(data.items())
        train_data_dict = dict(data_items[:split])
        valid_data_dict = dict(data_items[split:])

        train_data = DynamicItemDataset(train_data_dict)
        valid_data = DynamicItemDataset(valid_data_dict)

        # Logging the sizes of datasets
        logger.info(f"Training data size: {len(train_data)}, Validation data size: {len(valid_data)}")

        return train_data, valid_data

    except Exception as e:
        logger.error(f"Error preparing training data: {e}", exc_info=True)
        return None, None

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

        # Zmiana wysokości tonu za pomocą librosa
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
        return audio  # Zwraca oryginalne nagranie w przypadku błędu

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
                 augment: bool = False) -> str:
    """
    Processes an audio file by stripping silence and optionally augmenting it.

    Args:
        upload_path (str): Path to the input audio file.
        processed_path (str): Path to save the processed audio file.
        trim_silence (bool, optional): Whether to strip silence. Defaults to True.
        augment (bool, optional): Whether to augment the audio. Defaults to True.

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
            augmented_path = augment_audio(upload_path, processed_path, augmentation_type="noise")
            return augmented_path
        else:
            # Export the processed (silence-stripped) audio
            audio.export(processed_path, format="wav")
            logger.info(f"Audio file processed and saved: {processed_path}")

            return processed_path
    except Exception as e:
        logger.error(f"Error processing audio file: {e}", exc_info=True)
        raise


def generate_speech(text: str, profile: VoiceProfile, emotion: str = 'neutral', intonation: float = 1.0) -> str:
    if not Tacotron2 or not HIFIGAN:
        logger.error("TTS models are not loaded.")
        raise ValueError("Modele TTS są niedostępne.")

    try:
        mel_output, mel_length, alignment = Tacotron2.encode_text(text, emotion=emotion, intonation=intonation)
        waveforms = HIFIGAN.decode_batch(mel_output)
        audio_data = waveforms.squeeze().cpu().numpy()

        output_filename = f"generated_{uuid.uuid4().hex}.wav"
        generated_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)
        sf.write(generated_path, audio_data, 22050)

        logger.info(f"Speech generated: {generated_path}")
        return output_filename
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise

def save_transcription_result(transcription: str, save_path: str):
    """Zapisuje wynik transkrypcji do pliku."""
    try:
        with open(save_path, 'w') as f:
            f.write(transcription)
        logger.info(f"Transcription result saved: {save_path}")
    except Exception as e:
        logger.error(f"Error saving transcription result: {e}")

from torch.utils.data import DataLoader

def collate_fn(batch):
    """
    Custom collate function for DataLoader to pad sequences to equal length.
    Ensures input_lengths and batch_size consistency.
    """
    inputs = [item['sig'] for item in batch]
    tokens = [item['tokens_encoded'] for item in batch]
    tokens_lens = [item['tokens_lens'] for item in batch]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    input_lengths = torch.tensor([len(i) for i in inputs], dtype=torch.long)

    return {
        'inputs': inputs_padded,
        'tokens_encoded': tokens_padded,
        'tokens_lens': tokens_lens,
        'input_lengths': input_lengths
    }


def fine_tune_asr(model, train_data, valid_data, user_id, language, app_config, profile_id, num_epochs=10, batch_size=8, learning_rate=0.001):
    """
    Fine-tunes the ASR model based on the user's voice input.
    """
    try:
        logger.debug("Starting fine_tune_asr function...")
        logger.debug(f"User ID: {user_id}, Language: {language}, Profile ID: {profile_id}")
        logger.debug(f"Type of train_data: {type(train_data)}, Number of samples: {len(train_data)}")
        logger.debug(f"Type of valid_data: {type(valid_data)}, Number of samples: {len(valid_data)}")

        if not train_data or not valid_data:
            raise ValueError("No training or validation data.")

        config = {
            "lr": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "auto_mix_prec": True  # Enable mixed precision
        }
        logger.debug(f"Training configuration: {config}")

        # Update hyperparameters in model
        model.hparams.lr = config['lr']
        model.hparams.device = config['device']
        model.device = config['device']

        # Move all model modules to specified device
        for mod in model.modules.values():
            mod.to(config['device'])
        logger.debug(f"Model modules moved to device: {config['device']}")

        logger.debug("Starting ASR model training...")
        model.fit(
            epoch_counter=range(num_epochs),
            train_set=train_data,
            valid_set=valid_data,
            train_loader_kwargs={"batch_size": batch_size, "collate_fn": collate_fn},
            valid_loader_kwargs={"batch_size": batch_size, "collate_fn": collate_fn}
        )
        logger.debug("ASR model training completed successfully.")
        return "Trening zakończony pomyślnie."
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error during ASR training for profile {profile_id}: {e}")
        if profile_id in training_progress:
            training_progress[profile_id].update({
                "status": f"Błąd: {str(e)}",
                "progress": 0
            })
        raise e
    
def train_asr_on_voice_profile(profile, app_config, audio_files, transcriptions, batch_size=8, num_epochs=10, num_workers=4):
    """
    Trains the ASR model for the selected voice profile.

    Args:
        profile (VoiceProfile): The user voice profile object.
        app_config (dict): Application configuration settings.
        audio_files (list): List of paths to audio files.
        transcriptions (list): Corresponding transcriptions.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        num_epochs (int, optional): Number of epochs for training. Defaults to 10.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.

    Returns:
        str: Status message indicating success or failure of the training process.
    """
    try:
        # Validate input data
        if len(audio_files) != len(transcriptions):
            raise ValueError("Liczba plików audio i transkrypcji musi być taka sama.")
        
        logger.info(f"Rozpoczynanie treningu ASR dla profilu: {profile.name}")

        # Initialize the ASR brain model
        asr_brain = get_asr_brain()
        if not asr_brain:
            raise RuntimeError("Nie udało się załadować modelu ASR.")
        
        # Preprocess audio files (optional silence trimming, noise reduction)
        preprocessed_audio_files = []
        for audio_file in audio_files:
            processed_file = process_audio(audio_file, processed_path=f"processed/{uuid.uuid4().hex}.wav", trim_silence=True, augment=True)
            preprocessed_audio_files.append(processed_file)
        
        # Prepare training and validation data
        train_data, valid_data = prepare_training_data(preprocessed_audio_files, transcriptions)
        if not train_data or not valid_data:
            raise ValueError("Brak danych treningowych lub walidacyjnych.")

        # Create DataLoaders for training and validation
        train_loader, valid_loader = create_dataloaders(train_data, valid_data, batch_size=batch_size, num_workers=num_workers)

        if not train_loader or not valid_loader:
            raise RuntimeError("Nie udało się utworzyć DataLoaderów.")

        logger.info(f"Rozpoczęcie treningu modelu ASR na {num_epochs} epokach.")
        
        # Configure model settings for training
        asr_brain.hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mod in asr_brain.modules.values():
            mod.to(asr_brain.hparams.device)

        # Fine-tune the ASR model
        asr_brain.fit(
            epoch_counter=range(1, num_epochs + 1),
            train_set=train_data,
            valid_set=valid_data,
            train_loader_kwargs={"batch_size": batch_size, "collate_fn": collate_fn, "num_workers": num_workers},
            valid_loader_kwargs={"batch_size": batch_size, "collate_fn": collate_fn, "num_workers": num_workers}
        )
        
        logger.info(f"Trening ASR zakończony pomyślnie dla profilu: {profile.name}")
        return "Trening zakończony pomyślnie."

    except Exception as e:
        logger.error(f"Błąd podczas treningu ASR dla profilu {profile.name}: {e}", exc_info=True)
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
            reasons.append(f"Zbyt mała liczba Mel Bands ({mel_tensor_shape[1]}). Powinno być przynajmniej 40.")
        if snr_db < min_snr_db:
            is_suitable = False
            reasons.append(f"Stosunek sygnału do szumu (SNR) jest za niski: {snr_db} dB. Minimalny wymagany SNR to {min_snr_db} dB.")
        elif snr_db > max_snr_db:
            is_suitable = False
            reasons.append(f"Stosunek sygnału do szumu (SNR) jest za wysoki: {snr_db} dB. Maksymalny dozwolony SNR to {max_snr_db} dB.")
        if clipping_detected:
            is_suitable = False
            reasons.append("Nagranie zawiera clipping, co może wpływać negatywnie na jakość treningu.")
        if zcr > max_zcr:
            is_suitable = False
            reasons.append(f"Zero-Crossing Rate (ZCR) jest za wysoki: {zcr}. Maksymalny dozwolony ZCR to {max_zcr}.")
        if rms_db < min_rms_db:
            is_suitable = False
            reasons.append(f"Średnia głośność (RMS Energy) jest za niska: {rms_db} dB. Minimalny wymagany poziom to {min_rms_db} dB.")
        if is_suitable:
            return {
                "is_suitable": True,
                "reason": "Nagranie spełnia wszystkie wymagane kryteria."
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
            "reason": f"Nie udało się ocenić przydatności nagrania: {str(e)}"
        }

def create_dataloaders(train_set: DynamicItemDataset, 
                       valid_set: DynamicItemDataset, 
                       batch_size: int = 8, 
                       num_workers: int = 4):
    """
    Tworzy instancje DataLoader dla zestawów treningowego i walidacyjnego.

    Args:
        train_set (DynamicItemDataset): Zbiór danych treningowych.
        valid_set (DynamicItemDataset): Zbiór danych walidacyjnych.
        batch_size (int, optional): Liczba próbek w jednej partii. Domyślnie 8.
        num_workers (int, optional): Liczba procesów do ładowania danych. Domyślnie 4.

    Returns:
        tuple: Krotka zawierająca (train_loader, valid_loader) lub (None, None) w przypadku błędu.
    """
    try:
        logger.info("Tworzenie DataLoaderów dla zestawów treningowego i walidacyjnego.")

        # Tworzenie DataLoadera dla zestawu treningowego
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True  # Optymalizacja dla GPU
        )
        logger.info(f"DataLoader treningowy utworzony: batch_size={batch_size}, num_workers={num_workers}")

        # Tworzenie DataLoadera dla zestawu walidacyjnego
        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True  # Optymalizacja dla GPU
        )
        logger.info(f"DataLoader walidacyjny utworzony: batch_size={batch_size}, num_workers={num_workers}")

        return train_loader, valid_loader

    except Exception as e:
        logger.error(f"Błąd podczas tworzenia DataLoaderów: {e}", exc_info=True)
        return None, None
 
def correct_clipping(audio, threshold=0.99, max_corrections=5):
    """
    Korekuje clipping w nagraniu audio poprzez skalowanie sygnału.
    """
    try:
        for i in range(max_corrections):
            if np.any(np.abs(audio) > threshold):
                max_val = np.max(np.abs(audio))
                scale = threshold / max_val
                audio = audio * scale
                logger.warning(f"Clipping został wykryty i skorygowany. Iteracja {i+1}/{max_corrections}.")
            else:
                break
        clipping_detected = detect_clipping(audio, threshold)
        if clipping_detected:
            logger.warning("Clipping nadal występuje po korekcji.")
        return audio
    except Exception as e:
        logger.error(f"Error correcting clipping: {e}")
        return audio

def apply_low_pass_filter(audio, cutoff=3000, sample_rate=16000):
    """
    Stosuje filtr dolnoprzepustowy w celu zmniejszenia ilości przejść przez zero (ZCR).
    """
    try:
        from scipy.signal import butter, lfilter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        filtered_audio = lfilter(b, a, audio)
        logger.info("Filtr dolnoprzepustowy został zastosowany w celu redukcji ZCR.")
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
        # Automatyczna detekcja szumu na początku nagrania
        noisy_part = audio[:int(0.5 * sample_rate)]  # Pierwsze 0.5 sekundy
        reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noisy_part, prop_decrease=1.0)
        logger.debug("Redukcja szumów zakończona.")
        return reduced_noise
    except Exception as e:
        logger.error(f"Error reducing noise: {e}", exc_info=True)
        return audio  # Zwraca oryginalne nagranie w przypadku błędu


def ensure_min_duration(audio_segment, min_duration_sec=5.0):
    """
    Upewnia się, że nagranie ma minimalną długość poprzez dodanie ciszy.
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
    Wygładza sygnał audio za pomocą filtru Savitzky-Golay w celu redukcji przejść przez zero (ZCR).
    
    Args:
        audio (np.array): Sygnał audio do wygładzenia.
        window_length (int): Długość okna filtru. Musi być nieparzysta.
        polyorder (int): Rząd wielomianu używany do aproksymacji w filtrze.
    
    Returns:
        np.array: Wygładzony sygnał audio.
    """
    try:
        # Upewnij się, że window_length jest mniejsze niż długość sygnału i nieparzyste
        if window_length >= len(audio):
            window_length = len(audio) // 2 * 2 + 1  # Ustaw na najbliższą mniejszą wartość nieparzystą
        if window_length % 2 == 0:
            window_length += 1
        
        # Zastosowanie filtru Savitzky-Golay do wygładzenia sygnału
        smoothed_audio = savgol_filter(audio, window_length=window_length, polyorder=polyorder)
        
        return smoothed_audio
    except Exception as e:
        logger.error(f"Błąd podczas wygładzania audio: {e}", exc_info=True)
        return audio  # W przypadku błędu zwróć oryginalny sygnał

def process_audio_to_dataset(audio_path: str, n_mels=80, n_fft=1024, hop_length=256, max_duration_sec=400.0, save_dir="processed_data", max_iterations=3):
    """
    Przetwarza plik audio, oblicza spektrogram Mel i przygotowuje dane do treningu modelu ASR.

    Args:
        audio_path (str): Ścieżka do pliku audio.
        n_mels (int): Liczba pasm Mel w spektrogramie.
        n_fft (int): Wielkość FFT.
        hop_length (int): Długość kroku między oknami FFT.
        max_duration_sec (float): Maksymalny czas trwania audio (w sekundach).
        save_dir (str): Katalog, w którym zapisane zostaną przetworzone dane.
        max_iterations (int): Maksymalna liczba iteracji przetwarzania w celu korekcji jakości dźwięku.

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
            return None, {"error": "Załadowane audio jest puste."}, {"success": False, "error": "Załadowane audio jest puste."}

        duration_sec = len(audio) / sample_rate
        if duration_sec < 5.0:
            logger.warning(f"Audio duration {duration_sec} sekund jest poniżej minimalnej wartości (5.0 sekund). Dodaję ciszę.")
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
            logger.warning("Clipping nadal występuje po korekcji.")

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
            logger.warning(f"Liczba Mel Bands ({mel_tensor.shape[1]}) jest za niska. Próbuję przeliczyć mel spektrogram.")
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
    """Oblicza stosunek sygnału do szumu (SNR) nagrania audio."""
    try:
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) == 0:
            logger.warning("Nie wykryto mowy w nagraniu.")
            return 0.0
        signal_power = np.sum([
            np.sum(audio[start:end] ** 2) for start, end in non_silent_intervals
        ])

        if signal_power == 0:
            logger.warning("Moc sygnału wynosi zero po usunięciu ciszy.")
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
            logger.warning("Nie wykryto szumu w nagraniu. Ustawianie SNR na nieskończoność.")
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    except Exception as e:
        logger.error(f"Błąd podczas obliczania SNR: {e}", exc_info=True)
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
        logger.error(f"Błąd podczas transkrypcji za pomocą Whisper: {e}", exc_info=True)
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
            flash("Użytkownik z tym nazwiskiem lub adresem email już istnieje.", 'danger')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        try:
            db.session.commit()
            logger.info(f"New user registered: {username}")
            flash("Rejestracja zakończona sukcesem. Proszę się zalogować.", 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during user registration: {e}")
            flash("Wystąpił błąd podczas rejestracji. Spróbuj ponownie.", 'danger')
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
            flash("Logowanie zakończone sukcesem.", 'success')
            return response
        else:
            flash("Nieprawidłowe dane logowania.", 'danger')
            logger.warning(f"Failed login attempt for: {username_or_email}")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    unset_jwt_cookies(response)
    flash("Zostałeś wylogowany.", 'success')
    return response

@app.route('/dashboard')
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        flash("Użytkownik nie został znaleziony.", 'danger')
        return redirect(url_for('login'))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('dashboard.html', username=user.username, current_time=current_time)

@app.route('/upload_voice', methods=['GET', 'POST'])
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
            if filename.rsplit('.', 1)[1].lower() != 'wav':
                audio = AudioSegment.from_file(file)
                audio.export(upload_path, format='wav')
            else:
                file.save(upload_path)
            logger.info(f"Plik audio został przesłany: {upload_path}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania lub konwertowania pliku: {e}")
            flash("Nie udało się zapisać lub przekonwertować pliku.", 'danger')
            return redirect(request.url)

        try:
            processed_filename = unique_filename
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            process_audio(upload_path, processed_path)
            logger.info(f"Plik audio przetworzony: {processed_path}")

            # Transcription using Whisper
            transcription = transcribe_with_whisper(processed_path)
            print(f"Transkrypcja zakończona: {transcription}")

            voice_profile = VoiceProfile(
                user_id=user_id,
                name=name,
                audio_file=processed_filename,
                transcription=transcription,
                language=language
            )
            db.session.add(voice_profile)
            db.session.commit()

            flash("Profil głosowy został utworzony, przetworzony i transkrybowany.", 'success')
            return redirect(url_for('analyze_audio', profile_id=voice_profile.id))

        except Exception as e:
            db.session.rollback()
            logger.error(f"Błąd podczas przetwarzania audio: {e}")
            flash("Wystąpił błąd podczas przetwarzania audio.", 'danger')
            return redirect(request.url)

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

    # Przygotowanie danych augmentowanych
    augmented_audio_paths = []
    for i in range(3):  # Generowanie trzech różnych augmentacji
        unique_output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"augmented_{uuid.uuid4().hex}.wav")
        augmented_path = augment_audio(audio_path, unique_output_path, augmentation_type="noise")
        if augmented_path:
            augmented_audio_paths.append(augmented_path)

    # Upewnij się, że liczba transkrypcji odpowiada liczbie plików audio
    transcriptions = [profile.transcription] * len(augmented_audio_paths)

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
            "metrics": {}
        }
        logger.debug(f"Zainicjowano training_progress dla profilu {profile_id}: {training_progress[profile_id]}")

    def train():
        start_time = time.time()
        # Zabezpieczenie przy użyciu semafora
        training_semaphore.acquire()
        try:
            # Aktualizacja statusu na "Ładowanie modelu"
            update_training_progress(profile_id, status="Ładowanie modelu...", progress=5)

            # Wywołanie funkcji treningowej dla oryginalnego i augmentowanych plików
            audio_files = [audio_path] + augmented_audio_paths
            all_transcriptions = [profile.transcription] + transcriptions
            train_asr_on_voice_profile(profile, app.config, audio_files, all_transcriptions)

            # Aktualizacja statusu na zakończony trening
            update_training_progress(profile_id, status="Trening zakończony pomyślnie.", progress=100, current_epoch=10)

            logger.info(f"Trening ASR dla profilu ID {profile_id} zakończony pomyślnie.")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Błąd podczas treningu ASR dla profilu ID {profile_id}: {e}")
            update_training_progress(profile_id, status=f"Błąd: {str(e)}", progress=0)
        finally:
            # Zwolnienie semafora
            training_semaphore.release()
            # Usunięcie statusu treningu po zakończeniu lub błędzie
            with asr_model_lock:
                if profile_id in training_progress:
                    del training_progress[profile_id]

    # Uruchomienie treningu w osobnym wątku
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
        # Handle SSE for real-time updates
        def generate():
            while profile_id in training_progress:
                progress_data = training_progress.get(profile_id, {
                    "status": "Nie rozpoczęto",
                    "progress": 0,
                    "current_epoch": 0,
                    "total_epochs": 10,
                    "time_elapsed": 0,
                    "is_paused": False
                })
                yield f"data: {json.dumps(progress_data)}\n\n"
                time.sleep(1)
            yield f"data: {json.dumps({'status': 'Zakończono', 'progress': 100, 'current_epoch': 10, 'total_epochs': 10, 'time_elapsed': 0, 'is_paused': False})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        # Return JSON status
        progress = training_progress.get(profile_id, {
            "status": "Nie rozpoczęto",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 10,
            "time_elapsed": 0,
            "is_paused": False
        })
        return jsonify(progress)

# ------------------- Helper Functions for Training Progress -------------------

def update_training_progress(profile_id, status=None, progress=None, current_epoch=None, total_epochs=None, time_elapsed=None, metrics=None):
    with asr_model_lock:
        if profile_id not in training_progress:
            training_progress[profile_id] = {
                "status": status or "Rozpoczynanie...",
                "progress": progress or 0,
                "current_epoch": current_epoch or 0,
                "total_epochs": total_epochs or 10,
                "time_elapsed": time_elapsed or 0,
                "is_paused": training_progress.get(profile_id, {}).get("is_paused", False),
                "metrics": metrics or {}
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

@app.route('/pause_training/<int:profile_id>', methods=['POST'])
@jwt_required()
def pause_training(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()

    if not profile:
        return jsonify({"error": "Profil głosowy nie został znaleziony."}), 404

    if profile_id in training_progress:
        training_progress[profile_id]["status"] = "Trening wstrzymany"
        training_progress[profile_id]["is_paused"] = True
        pause_flags[profile_id] = True
        logger.info(f"Training paused for profile {profile_id}.")
        return jsonify({"message": "Trening został wstrzymany."}), 200
    return jsonify({"error": "Trening nie został znaleziony lub już zakończony."}), 400

@app.route('/resume_training/<int:profile_id>', methods=['POST'])
@jwt_required()
def resume_training(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()

    if not profile:
        return jsonify({"error": "Profil głosowy nie został znaleziony."}), 404

    if profile_id in training_progress and training_progress[profile_id].get("is_paused", False):
        training_progress[profile_id]["status"] = "Wznowiono trening"
        training_progress[profile_id]["is_paused"] = False
        pause_flags[profile_id] = False
        logger.info(f"Training resumed for profile {profile_id}.")
        return jsonify({"message": "Trening został wznowiony."}), 200
    return jsonify({"error": "Trening nie został wstrzymany lub już zakończony."}), 400

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

        try:
            output_filename = generate_speech(text, profile, emotion, intonation)
            flash("Mowa została wygenerowana i jest dostępna do pobrania.", 'success')
            return send_from_directory(app.config['GENERATED_FOLDER'], output_filename, as_attachment=True)
        except Exception as e:
            logger.error(f"Błąd podczas generowania mowy: {e}")
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
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    # Check if the file exists
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        logger.error(f"Plik {filename} nie istnieje w katalogu {app.config['UPLOAD_FOLDER']}")
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

@app.route('/edit_profile/<int:profile_id>', methods=['GET', 'POST'])
@jwt_required()
def edit_profile(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil głosowy nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    if request.method == 'POST':
        new_name = request.form.get('name', '').strip()
        new_language = request.form.get('language', '').strip()
        if new_name and new_language:
            profile.name = new_name
            profile.language = new_language
            try:
                db.session.commit()
                flash("Profil głosowy został zaktualizowany.", 'success')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error updating voice profile: {e}")
                flash("Wystąpił błąd podczas aktualizacji profilu.", 'danger')
                return redirect(request.url)
        else:
            flash("Nazwa profilu i język nie mogą być puste.", 'danger')
            return redirect(request.url)

    return render_template('edit_profile.html', profile=profile)

@app.route('/delete_profile/<int:profile_id>', methods=['POST'])
@jwt_required()
def delete_profile(profile_id):
    user_id = get_jwt_identity()
    profile = VoiceProfile.query.filter_by(id=profile_id, user_id=user_id).first()
    if not profile:
        flash("Profil głosowy nie został znaleziony.", 'danger')
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
        flash("Profil głosowy został usunięty.", 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting voice profile: {e}")
        flash("Wystąpił błąd podczas usuwania profilu.", 'danger')

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
        return redirect(url_for('profile'))

    audio_path = os.path.join(app.config['PROCESSED_FOLDER'], profile.audio_file)
    if not os.path.exists(audio_path):
        flash("Plik audio nie został znaleziony.", 'danger')
        return redirect(url_for('profile'))

    try:
        # Pobierz instancję ASRBrain, jeśli potrzebna w dalszej części
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
            return redirect(url_for('profile'))
        
        suitability = evaluate_audio_suitability(processing_info)

        audio_signal = audio_signal_function(audio_path)
        if audio_signal:
            processing_info['audio_signal'] = audio_signal
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
        # Inicjalizacja kontekstu aplikacji
        with app.app_context():
            db.create_all()  # Tworzenie wszystkich tabel w bazie danych
            
            # Synchroniczne ładowanie modelu ASR przed startem aplikacji
            logger.info("Rozpoczynanie ładowania modelu ASR...")
            asr_brain_instance = get_asr_brain()
            if asr_brain_instance:
                logger.info("Model ASR został pomyślnie załadowany.")
            else:
                logger.critical("Nie udało się załadować modelu ASR. Aplikacja zostanie zamknięta.")
                exit(1)
        
        logger.info("Flask application has been started.")
        
        # Rozpoczęcie wątku monitorującego pamięć systemową
        monitor_thread = threading.Thread(target=background_system_memory_monitor, args=(120,), daemon=True)
        monitor_thread.start()
        logger.info("Wątek monitorujący pamięć systemową został uruchomiony.")
        
        # Uruchomienie aplikacji Flask
        app.run(host="0.0.0.0",port=5000, debug=True, use_reloader=True)
        
    except KeyboardInterrupt:
        logger.info("Application has been stopped by the user.")
    except Exception as e:
        # Użycie loggera o wysokim poziomie krytycznym do logowania błędów
        logger.critical(f"Failed to start the application: {e}")
    finally:
        # Opcjonalnie można dodać kod do czyszczenia lub zamknięcia zasobów
        logger.info("Application cleanup process completed.")

