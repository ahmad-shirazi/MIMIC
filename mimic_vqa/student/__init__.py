"""
Student model modules for Phase B - Student Training
"""

from .model import StudentModel
from .trainer import StudentTrainer
from .pruning import IterativeMagnitudePruner

__all__ = [
    "StudentModel",
    "StudentTrainer",
    "IterativeMagnitudePruner"
]
