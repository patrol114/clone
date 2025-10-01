from . import db
from werkzeug.security import generate_password_hash, check_password_hash

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