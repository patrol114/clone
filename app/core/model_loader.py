import os
import torch
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor
from safetensors.torch import load_file

from config import Config
from app.models import ASRModel

logger = logging.getLogger(__name__)

asr_model_cache = {}

def _load_profile_model(profile_id, base_model_id, cache_dir):
    asr_model_entry = ASRModel.query.filter_by(voice_profile_id=profile_id).first()
    if not asr_model_entry or not asr_model_entry.model_file.endswith('.pt'):
        return None

    model_path = os.path.join(Config.ASR_MODELS_FOLDER, asr_model_entry.model_file)
    if model_path in asr_model_cache:
        logger.info(f"Profile-specific model for profile ID {profile_id} loaded from cache.")
        return asr_model_cache[model_path]

    if os.path.exists(model_path):
        logger.info(f"Loading profile-specific ASR model from: {model_path}")
        try:
            processor = Wav2Vec2Processor.from_pretrained(base_model_id, cache_dir=cache_dir)
            model = Wav2Vec2ForCTC.from_pretrained(base_model_id, cache_dir=cache_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            asr_model_cache[model_path] = (model, processor, device)
            return model, processor, device
        except Exception as e:
            logger.error(f"Error loading profile model {model_path}: {e}")
            raise
    return None

def _load_base_model(base_model_id, cache_dir, use_gradient_checkpointing):
    if base_model_id in asr_model_cache:
        logger.info(f"Base ASR model '{base_model_id}' loaded from cache.")
        return asr_model_cache[base_model_id]

    logger.info(f"Loading base ASR model: {base_model_id}")
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

def load_asr_model(profile_id=None, base_model_id=Config.BASE_ASR_MODEL_ID, cache_dir=".cache", use_gradient_checkpointing=True):
    if profile_id:
        model_tuple = _load_profile_model(profile_id, base_model_id, cache_dir)
        if model_tuple:
            return model_tuple

    return _load_base_model(base_model_id, cache_dir, use_gradient_checkpointing)


def initialize_whisper_model():
    try:
        logger.info("Ładowanie WhisperProcessor.")
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="pl", task="transcribe")
        logger.info("Ładowanie WhisperForConditionalGeneration.")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        whisper_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        whisper_model.to(whisper_device)
        logger.info(f"Model Whisper uruchomiony na urządzeniu: {whisper_device}")
        whisper_model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="pl", task="transcribe")
        logger.info("Whisper model and processor initialized successfully.")
        return whisper_model, whisper_processor, whisper_device
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji modelu Whisper: {e}", exc_info=True)
        raise

def transcribe_with_whisper(audio_path: str, model, processor, device) -> str:
    if not isinstance(audio_path, str) or not audio_path.endswith(('.wav', '.mp3')):
        logger.error("Nieprawidłowy format pliku lub brak pliku audio.")
        return ""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        if not isinstance(input_features, torch.Tensor):
            raise TypeError("input_features nie są typu torch.Tensor.")
        if len(input_features.shape) != 3:
            raise ValueError("input_features powinny mieć 3 wymiary.")
        encoder_outputs = model.get_encoder()(input_features)
        attention_mask = torch.ones((input_features.shape[0], input_features.shape[2]), dtype=torch.long).to(device)
        predicted_ids = model.generate(
            encoder_outputs=encoder_outputs,
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

def load_asrtts_model(user_id, profile_id, asr_models_folder):
    try:
        profile_folder = os.path.join(asr_models_folder, f"user_{user_id}", f"profile_{profile_id}")
        if not os.path.exists(profile_folder):
            raise FileNotFoundError(f"Nie znaleziono folderu dla użytkownika {user_id} i profilu {profile_id} w {profile_folder}.")

        model_file = None
        config_file = None
        for file in os.listdir(profile_folder):
            if file.endswith(".safetensors"):
                model_file = os.path.join(profile_folder, file)
            elif file.endswith("config.json"):
                config_file = os.path.join(profile_folder, file)

        if not model_file or not config_file:
            raise FileNotFoundError(f"Brak plików modelu lub konfiguracji dla profilu {profile_id} użytkownika {user_id}.")

        logger.info(f"Wczytywanie modelu z {model_file}")
        model_state_dict = load_file(model_file)
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=profile_folder,
            config=config_file,
            local_files_only=True
        )
        model.load_state_dict(model_state_dict)

        logger.info(f"Wczytywanie tokenizerów z {config_file}")
        processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path=profile_folder,
            local_files_only=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, processor, device
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania modelu dla użytkownika {user_id} i profilu {profile_id}: {e}")
        raise RuntimeError(f"Nie udało się wczytać modelu: {e}")