import os
import random
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.nnet.losses import ctc_loss
from speechbrain.utils.metric_stats import ErrorRateStats
import logging
import pynvml
import psutil

from .model_loader import load_asr_model

logger = logging.getLogger(__name__)

def audio_pipeline(audio_path):
    try:
        import librosa
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Plik audio nie istnieje: {audio_path}")
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        if audio.size == 0:
            raise ValueError("Sygnał audio jest pusty.")
        waveform = torch.tensor(audio, dtype=torch.float32)
        audio_lens = torch.tensor([len(audio)], dtype=torch.long)
        return waveform, audio_lens
    except Exception as e:
        logger.error(f"Błąd w audio_pipeline: {e}", exc_info=True)
        raise

def create_text_pipeline(tokenizer):
    def text_pipeline(transcription):
        try:
            tokens = tokenizer.encode(transcription)
            tokens_encoded = torch.tensor(tokens, dtype=torch.long)
            tokens_lens = torch.tensor([len(tokens)], dtype=torch.long)
            return tokens_encoded, tokens_lens
        except Exception as e:
            logger.error(f"Błąd w text_pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Tokenizacja transkrypcji nie powiodła się: {e}")
    return text_pipeline

def prepare_training_data(preprocessed_audio_files: list, transcriptions: list, split_ratio: float = 0.8):
    try:
        if len(preprocessed_audio_files) != len(transcriptions):
            error_msg = f"Liczba plików audio ({len(preprocessed_audio_files)}) nie zgadza się z liczbą transkrypcji ({len(transcriptions)})."
            logger.error(error_msg)
            raise ValueError(error_msg)
        data = {
            f'sample_{idx}': {
                'audio_path': audio_path,
                'transcription': transcription
            } for idx, (audio_path, transcription) in enumerate(zip(preprocessed_audio_files, transcriptions))
        }
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

def setup_dataio(
    asr_brain: "ASRBrain",
    preprocessed_audio_files: list,
    transcriptions: list,
    split_ratio: float = 0.8
) -> tuple:
    try:
        train_data, valid_data = prepare_training_data(preprocessed_audio_files, transcriptions, split_ratio)
        train_dataset = sb.dataio.dataset.DynamicItemDataset(train_data)
        valid_dataset = sb.dataio.dataset.DynamicItemDataset(valid_data)
        sb.dataio.dataset.add_dynamic_item(
            [train_dataset, valid_dataset],
            audio_pipeline,
            takes=["audio_path"],
            provides=["sig", "audio_lens"]
        )

        tokenizer = asr_brain.hparams.processor.tokenizer
        text_pipeline = create_text_pipeline(tokenizer)

        sb.dataio.dataset.add_dynamic_item(
            [train_dataset, valid_dataset],
            text_pipeline,
            takes=["transcription"],
            provides=["tokens_encoded", "tokens_lens"]
        )
        sb.dataio.dataset.set_output_keys(
            [train_dataset, valid_dataset],
            ['sig', 'audio_lens', 'tokens_encoded', 'tokens_lens']
        )
        logger.info("DataIO setup completed successfully.")
        return train_dataset, valid_dataset, create_collate_fn(tokenizer)
    except Exception as e:
        logger.error(f"Error in setup_dataio: {e}", exc_info=True)
        raise

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        try:
            batch = [sample for sample in batch if sample is not None]
            if len(batch) == 0:
                return None
            batch_tokens_encoded = [
                item['tokens_encoded'].clone().detach() if isinstance(item['tokens_encoded'], torch.Tensor)
                else torch.tensor(item['tokens_encoded'], dtype=torch.long)
                for item in batch
            ]
            tokens_encoded_padded = torch.nn.utils.rnn.pad_sequence(
                batch_tokens_encoded,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            for item in batch:
                if isinstance(item['sig'], torch.Tensor):
                    item['sig'] = item['sig'].float()
                else:
                    item['sig'] = torch.tensor(item['sig'], dtype=torch.float32)
            batched_data = PaddedBatch(
                examples=batch,
                padded_keys=['sig', 'tokens_encoded'],
                device_prep_keys=['sig', 'tokens_encoded'],
                apply_default_convert=False,
                nonpadded_stack=True
            )
            batched_data.audio_lens = torch.tensor(
                [sample['audio_lens'] for sample in batch],
                dtype=torch.long
            ).clone().detach()
            batched_data.tokens_lens = torch.tensor(
                [sample['tokens_lens'] for sample in batch],
                dtype=torch.long
            ).clone().detach()
            return batched_data
        except Exception as e:
            logger.error(f"Błąd podczas paddingu batcha: {str(e)}", exc_info=True)
            raise
    return collate_fn

class ASRBrain(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts=None, checkpointer=None, use_amp=True):
        super().__init__(modules, opt_class, hparams, run_opts=run_opts, checkpointer=checkpointer)
        self.run_opts = run_opts or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modules['model'].to(self.device)
        self.checkpointer = checkpointer
        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)
        self.configure_optimizers()
        self.hparams.batch_size = getattr(hparams, 'batch_size', 8)
        self.hparams.num_workers = getattr(hparams, 'num_workers', 0)
        self.hparams.lr = getattr(hparams, 'lr', 0.001)
        self.enable_gradient_checkpointing = getattr(self.hparams, "enable_gradient_checkpointing", False)
        if self.enable_gradient_checkpointing and hasattr(self.modules['model'], 'config'):
            self.modules['model'].config.gradient_checkpointing = True
        precision = self.run_opts.get("precision", "fp16")
        if self.device.type == "cuda" and precision == "fp16":
            self.scaler = GradScaler() if use_amp else None
        else:
            self.scaler = None
        self.train_stats_history = {}
        self.valid_stats_history = {}

    def configure_optimizers(self):
        try:
            if not hasattr(self, 'opt_class') or self.opt_class is None:
                raise ValueError("Klasa optymalizatora nie jest zdefiniowana")
            if not hasattr(self.modules, 'model'):
                raise ValueError("Model nie jest zdefiniowany w self.modules")
            parameters = list(self.modules['model'].parameters())
            if not parameters:
                raise ValueError("Model nie ma parametrów do optymalizacji")
            self.optimizer = self.opt_class(parameters)
            return self.optimizer
        except Exception as e:
            logger.error(f"Błąd podczas konfigurowania optymalizatora: {e}", exc_info=True)
            raise

    def compute_forward(self, batch: dict, stage: sb.Stage) -> dict:
        try:
            wavs, wav_lens = batch['sig'].data, batch['audio_lens'].data
            if wavs is None:
                raise ValueError("Dane audio (wavs) nie zostały poprawnie załadowane.")
            if wavs.dim() == 1:
                wavs = wavs.unsqueeze(0).unsqueeze(1)
            elif wavs.dim() == 2:
                wavs = wavs.unsqueeze(1)

            MIN_INPUT_SIZE = 16000
            current_length = wavs.shape[2]
            if current_length < MIN_INPUT_SIZE:
                padding_size = MIN_INPUT_SIZE - current_length
                wavs = torch.nn.functional.pad(wavs, (0, padding_size), "constant", 0)

            target_sampling_rate = getattr(self.hparams, "target_sampling_rate", self.hparams.sample_rate)
            wavs_list = [wavs[i, 0, :wav_lens[i].item()].cpu().numpy() for i in range(wavs.shape[0])]
            inputs = self.hparams.processor(
                wavs_list, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
            input_lengths = (wav_lens // getattr(self.hparams, "downsample_factor", 320)).long().to(self.device)
            precision = self.run_opts.get("precision", "fp16")
            autocast_enabled = (precision == "fp16")
            device_type = self.device.type
            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                logits = self.modules['model'](input_values, attention_mask=attention_mask).logits
                logits = logits.transpose(0, 1)
            log_probs = F.log_softmax(logits, dim=-1)
            return {"log_probs": log_probs, "input_lengths": input_lengths, "attention_mask": attention_mask}
        except Exception as e:
            logger.error(f"Błąd w compute_forward: {e}", exc_info=True)
            raise

    def compute_objectives(self, predictions, batch, stage):
        try:
            if isinstance(predictions, dict):
                log_probs = predictions["log_probs"]
                input_lengths = predictions["input_lengths"]
            else:
                raise ValueError("Invalid predictions format")
            targets = batch['tokens_encoded'].data.to(self.device)
            target_lengths = batch['tokens_lens'].to(self.device)
            batch_size = log_probs.size(1)
            assert target_lengths.size(0) == batch_size, "target_lengths should match batch size"
            loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean')
            return loss
        except Exception as e:
            logger.error(f"Błąd podczas obliczania funkcji straty: {e}", exc_info=True)
            raise

    def train_step(self, batch):
        optimizer = self.optimizer
        optimizer.zero_grad()
        try:
            outputs = self.compute_forward(batch, stage=sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, stage=sb.Stage.TRAIN)
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
        from tqdm import tqdm
        try:
            batch_size = getattr(self.hparams, 'batch_size', 8)
            num_workers = getattr(self.hparams, 'num_workers', 0)
            train_loader_kwargs['collate_fn'] = collate_fn
            valid_loader_kwargs['collate_fn'] = collate_fn
            train_dataloader = sb.dataio.dataloader.make_dataloader(
                train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn
            )
            total_steps = len(train_dataloader)
            epoch_pbar = tqdm(total=len(epoch_counter), desc="Epoki", unit="epoch", colour="blue") if progressbar else None
            for epoch in epoch_counter:
                epoch_loss = 0.0
                batch_pbar = tqdm(total=total_steps, desc=f"Batch {epoch}", unit="batch", colour="green") if progressbar else None
                self.on_stage_start(sb.Stage.TRAIN, epoch)
                for batch in train_dataloader:
                    if batch is None:
                        continue
                    loss = self.train_step(batch)
                    epoch_loss += loss.item()
                    if batch_pbar:
                        batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
                        batch_pbar.update(1)
                if batch_pbar:
                    batch_pbar.close()
                avg_loss = epoch_loss / total_steps
                self.on_stage_end(sb.Stage.TRAIN, stage_loss=avg_loss, epoch=epoch)
                if valid_set is not None:
                    valid_dataloader = sb.dataio.dataloader.make_dataloader(
                        valid_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
                    )
                    valid_loss = 0.0
                    valid_steps = len(valid_dataloader)
                    valid_pbar = tqdm(total=valid_steps, desc=f"Walidacja epoka {epoch}", unit="batch", colour="yellow") if progressbar else None
                    self.on_stage_start(sb.Stage.VALID, epoch)
                    for batch in valid_dataloader:
                        if batch is None:
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
                    self.on_stage_end(sb.Stage.VALID, stage_loss=avg_valid_loss, epoch=epoch)
                if self.checkpointer is not None:
                    if epoch % getattr(self.hparams, "save_interval", 1) == 0:
                        self.checkpointer.save_checkpoint()
                if epoch_pbar:
                    epoch_pbar.update(1)
            if epoch_pbar:
                epoch_pbar.close()
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}", exc_info=True)
            raise

    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            self.modules['model'].train()
        else:
            self.modules['model'].eval()
        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_stats = {"epoch": epoch, "loss": stage_loss}
        elif stage == sb.Stage.VALID:
            if hasattr(self.hparams, "lr") and hasattr(self.hparams, "lr_annealing_factor"):
                old_lr = self.hparams.lr
                self.hparams.lr *= self.hparams.lr_annealing_factor
                self.checkpointer.save_checkpoint(name=f"epoch_{epoch}_loss_{stage_loss:.4f}")
                self.hparams.lr = old_lr
            self.valid_stats_history[epoch] = {"loss": stage_loss, "epoch": epoch}

def get_asr_brain(config) -> 'ASRBrain':
    try:
        model, processor, device = load_asr_model(base_model_id=config['BASE_ASR_MODEL_ID'])
        if not model or not processor:
            raise RuntimeError("Nie udało się załadować modelu ASR.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        checkpointer = Checkpointer(
            checkpoints_dir=config['ASR_MODELS_FOLDER'],
            recoverables={"model": model, "optimizer": optimizer}
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
        return asr_brain_instance
    except Exception as e:
        logger.error(f"Błąd inicjalizacji ASRBrain: {e}", exc_info=True)
        raise