"""
HuMERT-300M: Universal Audio Understanding Model
Flash Linear Transformer + Multi-Teacher Learning
"""

__version__ = "1.0.0"

from .model import HuMERTModel
from .encoder import FlashLinearTransformerEncoder
from .frontend import ConvFrontend24k
from .teachers import DACTeacher, SpeechTeacher, MusicalTeacher
from .heads import PredictionHeads

__all__ = [
    "HuMERTModel",
    "FlashLinearTransformerEncoder", 
    "ConvFrontend24k",
    "DACTeacher",
    "SpeechTeacher", 
    "MusicalTeacher",
    "PredictionHeads"
]