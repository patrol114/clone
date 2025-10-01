import os
import random
import uuid
import time
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import butter, lfilter, savgol_filter
import noisereduce as nr

from config import Config

def add_reverb(audio_segment, delay_ms=50, decay_db=6, num_echoes=3):
    try:
        reverberated = audio_segment
        for i in range(1, num_echoes + 1):
            delayed = AudioSegment.silent(duration=delay_ms * i) + (audio_segment - (decay_db * i))
            reverberated = reverberated.overlay(delayed)
        return reverberated
    except Exception as e:
        raise

def augment_audio(audio_path, output_path, augmentation_type=None):
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.unsqueeze(0)

        if augmentation_type == "noise":
            noise = generate_noise(audio, noise_factor=0.05)
            augmented_audio = audio + noise
            augmented_audio = torch.clamp(augmented_audio, -1.0, 1.0)
        elif augmentation_type == "speed_pitch":
            speed_factor = random.uniform(0.9, 1.1)
            pitch_factor = random.uniform(-2, 2)
            augmented_audio = change_speed(audio, speed_factor)
            augmented_audio = change_pitch(augmented_audio, pitch_factor)
        elif augmentation_type == "reverb":
            augmented_audio = T.Reverberate(sample_rate)(audio)
        else:
            augmented_audio = audio

        augmented_audio = augmented_audio.squeeze(0)
        augmented_audio = augmented_audio / augmented_audio.abs().max()
        torchaudio.save(output_path, augmented_audio, sample_rate)
        return output_path
    except Exception as e:
        raise

def generate_noise(audio, noise_factor=0.05):
    noise = torch.randn_like(audio) * noise_factor
    return noise

def change_speed(audio_segment, speed_factor):
    try:
        new_frame_rate = int(audio_segment.frame_rate * speed_factor)
        return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(audio_segment.frame_rate)
    except Exception as e:
        raise

def change_pitch(audio, semitones=2):
    try:
        y = np.array(audio.get_array_of_samples()).astype(np.float32)
        y /= np.iinfo(audio.array_type).max
        y_shifted = librosa.effects.pitch_shift(y, sr=audio.frame_rate, n_steps=semitones)
        y_shifted = np.clip(y_shifted, -1.0, 1.0)
        y_shifted_int16 = (y_shifted * 32767).astype(np.int16)
        shifted_audio = AudioSegment(
            y_shifted_int16.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        )
        return shifted_audio
    except Exception as e:
        return audio

def strip_silence(audio_segment, silence_thresh=-40.0, min_silence_len=500, keep_silence=100):
    try:
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        combined_audio = AudioSegment.empty()
        for chunk in chunks:
            combined_audio += chunk
        return combined_audio
    except Exception as e:
        raise

def process_audio(upload_path: str, processed_path: str,
                  trim_silence: bool = True,
                  augment: bool = False,
                  augment_options: dict = None) -> str:
    try:
        audio = AudioSegment.from_file(upload_path)
        if trim_silence:
            audio = strip_silence(audio)
        audio = audio.normalize()
        if augment:
            if augment_options is None:
                augment_options = {"augmentation_type": "noise"}
            augmented_path = augment_audio(upload_path, processed_path, **augment_options)
            return augmented_path
        else:
            audio.export(processed_path, format="wav")
            return processed_path
    except Exception as e:
        raise

def save_transcription_result(transcription: str, save_path: str):
    try:
        with open(save_path, 'w') as f:
            f.write(transcription)
    except Exception as e:
        raise

def correct_clipping(audio, threshold=0.99, max_corrections=5):
    try:
        for i in range(max_corrections):
            if np.any(np.abs(audio) > threshold):
                max_val = np.max(np.abs(audio))
                scale = threshold / max_val
                audio = audio * scale
            else:
                break
        return audio
    except Exception as e:
        return audio

def apply_low_pass_filter(audio, cutoff=3000, sample_rate=16000):
    try:
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        filtered_audio = lfilter(b, a, audio)
        return filtered_audio
    except Exception as e:
        return audio

def reduce_noise_audio(audio, sample_rate):
    try:
        noisy_part = audio[:int(0.5 * sample_rate)]
        reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noisy_part, prop_decrease=1.0)
        return reduced_noise
    except Exception as e:
        return audio

def ensure_min_duration(audio_segment, min_duration_sec=5.0):
    try:
        current_duration_sec = len(audio_segment) / 1000.0
        if current_duration_sec >= min_duration_sec:
            return audio_segment
        else:
            required_duration_ms = int((min_duration_sec - current_duration_sec) * 1000)
            silence = AudioSegment.silent(duration=required_duration_ms)
            return audio_segment + silence
    except Exception as e:
        raise

def smooth_audio(audio, window_length=101, polyorder=2):
    try:
        if window_length >= len(audio):
            window_length = len(audio) // 2 * 2 + 1
        if window_length % 2 == 0:
            window_length += 1
        smoothed_audio = savgol_filter(audio, window_length=window_length, polyorder=polyorder)
        return smoothed_audio
    except Exception as e:
        return audio

def process_audio_to_dataset(audio_path: str, n_mels=80, n_fft=1024, hop_length=256, max_duration_sec=400.0, save_dir="processed_data", max_iterations=3):
    try:
        if not os.path.exists(audio_path):
            return None, {"error": "Plik audio nie istnieje."}, {"success": False, "error": "Plik audio nie istnieje."}

        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)

        if len(audio) == 0:
            return None, {"error": "Załadowane audio jest puste."}, {"success": False, "error": "Załadowane audio jest puste."}

        duration_sec = len(audio) / sample_rate
        if duration_sec < 5.0:
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.dtype.itemsize,
                channels=1
            )
            audio_segment = ensure_min_duration(audio_segment, min_duration_sec=5.0)
            audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2**15)
            duration_sec = len(audio) / sample_rate

        for iteration in range(1, max_iterations + 1):
            audio = reduce_noise_audio(audio, sample_rate)
            audio = correct_clipping(audio, threshold=0.99)
            audio = smooth_audio(audio)

            additional_metrics = compute_additional_metrics(audio, sample_rate)
            rms_db = additional_metrics["rms_db"]
            zcr = additional_metrics["zcr"]

            clipping_detected = detect_clipping(audio)
            if not clipping_detected and zcr <= 0.1:
                break
            else:
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

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        mel_tensor = torch.FloatTensor(mel_spectrogram_normalized).unsqueeze(0)
        signal_tensor = torch.tensor(audio, dtype=torch.float32)
        processing_info["mel_tensor_shape"] = mel_tensor.shape

        if mel_tensor.shape[1] < 40:
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_mels=40, n_fft=n_fft, hop_length=hop_length, power=2.0
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
            mel_tensor = torch.FloatTensor(mel_spectrogram_normalized).unsqueeze(0)
            processing_info["n_mels"] = 40
            processing_info["mel_tensor_shape"] = mel_tensor.shape

        save_status = save_processed_data(mel_tensor, signal_tensor, processing_info, save_dir)
        return mel_tensor, processing_info, save_status
    except Exception as e:
        return None, {"error": str(e)}, {"success": False, "error": str(e)}

def save_processed_data(mel_tensor, signal_tensor, processing_info, save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"processed_audio_{int(time.time())}.pt"
        file_path = os.path.join(save_dir, file_name)
        torch.save({
            'mel_spectrogram': mel_tensor,
            'audio_signal': signal_tensor,
            'processing_info': processing_info
        }, file_path)
        return {"success": True, "path": file_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def compute_snr(audio, sample_rate, top_db=30):
    try:
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) == 0:
            return 0.0
        signal_power = np.sum([
            np.sum(audio[start:end] ** 2) for start, end in non_silent_intervals
        ])
        if signal_power == 0:
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
        if noise_power < 1e-10:
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    except Exception as e:
        return 0.0

def detect_clipping(audio, threshold=0.99):
    try:
        return np.any(np.abs(audio) > threshold)
    except Exception as e:
        return False

def compute_additional_metrics(audio, sample_rate):
    try:
        if len(audio) == 0:
            return {"rms_db": 0.0, "zcr": 0.0}
        rms_energy = np.sqrt(np.mean(audio ** 2))
        rms_db = librosa.amplitude_to_db(np.array([rms_energy]))[0]
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        return {"rms_db": round(rms_db, 2), "zcr": round(zcr, 4)}
    except Exception as e:
        return {"rms_db": 0.0, "zcr": 0.0}

def audio_signal_function(audio_path: str) -> list:
    try:
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        return audio.tolist()
    except Exception as e:
        return []

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
        return {
            "is_suitable": False,
            "reason": f"Nie uda³o siê oceniæ przydatnoci nagrania: {str(e)}"
        }