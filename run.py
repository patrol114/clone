import threading
import sys
import logging
from app import create_app, db
from app.core.model_loader import initialize_whisper_model
from app.core.asr_brain import get_asr_brain

app = create_app()

def background_system_memory_monitor(interval=240):
    import time
    import pynvml
    import psutil

    logger = logging.getLogger(__name__)
    while True:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info_gpu.used // (1024 ** 2)
            gpu_total = info_gpu.total // (1024 ** 2)
            gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            pynvml.nvmlShutdown()
        except Exception:
            gpu_used, gpu_total, gpu_percent = 0, 0, 0

        try:
            ram_info = psutil.virtual_memory()
            ram_used = ram_info.used // (1024 ** 2)
            ram_total = ram_info.total // (1024 ** 2)
            ram_percent = ram_info.percent
            cpu_percent = psutil.cpu_percent(interval=1)
        except Exception:
            ram_used, ram_total, ram_percent, cpu_percent = 0, 0, 0, 0

        logger.info(f"Zużycie Systemowe - GPU: {gpu_used} MB / {gpu_total} MB ({gpu_percent:.2f}%) | RAM: {ram_used} MB / {ram_total} MB ({ram_percent}%) | CPU: {cpu_percent}%")
        time.sleep(interval)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # Initialize models
        try:
            app.logger.info("Ładowanie modelu Whisper...")
            initialize_whisper_model()
            app.logger.info("Model Whisper załadowany.")

            app.logger.info("Ładowanie modelu ASR...")
            get_asr_brain(app.config)
            app.logger.info("Model ASR załadowany.")
        except Exception as e:
            app.logger.critical(f"Nie udało się załadować modeli AI: {e}", exc_info=True)
            sys.exit(1)

    monitor_thread = threading.Thread(target=background_system_memory_monitor, daemon=True)
    monitor_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True)