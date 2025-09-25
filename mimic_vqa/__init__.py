"""
MIMIC-VQA: Teacher-Student Knowledge Distillation Framework
for Visual Question Answering on Documents
"""

__version__ = "1.0.0"
__author__ = "MIMIC-VQA Team"

from .config import Config
from .teacher.agent import TeacherAgent
from .student.model import StudentModel
from .student.trainer import StudentTrainer
from .inference.pipeline import InferencePipeline

__all__ = [
    "Config",
    "TeacherAgent", 
    "StudentModel",
    "StudentTrainer",
    "InferencePipeline"
]
