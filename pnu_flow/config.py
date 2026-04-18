"""
Citations (open-source):
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- SimPy: https://simpy.readthedocs.io/
- PyTorch: https://pytorch.org/
- NetworkX: https://networkx.org/
- scikit-learn: https://scikit-learn.org/
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PathsConfig:
    root_dir: Path = Path(__file__).resolve().parent
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")
    models_dir:    Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "models")
    scalers_dir:   Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "scalers")
    data_dir:      Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "data")


@dataclass
class SimulationConfig:
    seed: int = 42
    days: List[str] = field(default_factory=lambda: ["Sun","Mon","Tue","Wed","Thu"])
    
    # تحديث أسماء المناطق لتطابق الـ Graph الجديد
    building_zones: List[str] = field(default_factory=lambda: [
        "Main_Lobby", "North_Wing_G", "South_Wing_G", "Elevators_G",
        "Student_Lounge", "Central_Stairs_G1", "Elevators_F1", "North_Wing_F1",
        "South_Wing_F1", "Quiet_Study_Area", "Central_Stairs_12", "Elevators_F2",
        "North_Wing_F2", "South_Wing_F2", "Grand_Auditorium",
    ])
    
    # تحديث السعات الاستيعابية بالأسماء الجديدة
    zone_capacity: Dict[str,int] = field(default_factory=lambda: {
        "Main_Lobby":120, "North_Wing_G":180, "South_Wing_G":150,
        "Elevators_G":100, "Student_Lounge":220, "Central_Stairs_G1":80,
        "Elevators_F1":90, "North_Wing_F1":160, "South_Wing_F1":140,
        "Quiet_Study_Area":120, "Central_Stairs_12":70, "Elevators_F2":80,
        "North_Wing_F2":150, "South_Wing_F2":120, "Grand_Auditorium":200,
    })
    
    # تحديث الإحداثيات بالأسماء الجديدة (ضروري جداً لعمل الـ A*)
    zone_coords: Dict[str,Tuple[float,float]] = field(default_factory=lambda: {
        "Main_Lobby":        (20.0,  0.0),
        "North_Wing_G":      (10.0, 18.0),
        "South_Wing_G":      (35.0, 18.0),
        "Elevators_G":       (22.0, 18.0),
        "Student_Lounge":    (38.0,  8.0),
        "Central_Stairs_G1": (24.0, 20.0),
        "Elevators_F1":      (22.0, 36.0),
        "North_Wing_F1":     (10.0, 36.0),
        "South_Wing_F1":     (35.0, 36.0),
        "Quiet_Study_Area":  ( 5.0, 46.0),
        "Central_Stairs_12": (24.0, 40.0),
        "Elevators_F2":      (22.0, 58.0),
        "North_Wing_F2":     (10.0, 58.0),
        "South_Wing_F2":     (35.0, 58.0),
        "Grand_Auditorium":  (20.0, 72.0),
    })
    
    simulation_interval_minutes: int = 5
    operating_hours: Tuple[int,int] = (7, 20)


@dataclass
class ModelConfig:
    lookback_steps:       int   = 4
    hidden_size:          int   = 64
    num_layers:           int   = 2
    dropout:              float = 0.2
    learning_rate:        float = 1e-3
    batch_size:           int   = 64
    epochs:               int   = 50
    early_stop_patience:  int   = 7
    confidence_threshold: float = 0.60
    penalty_factor:       float = 1.2
    
    # تحديث أماكن الدراسة المتاحة بالأسماء الجديدة
    study_zones: List[str] = field(default_factory=lambda: [
        "Quiet_Study_Area", "Student_Lounge", "North_Wing_F1", "South_Wing_F1",
    ])


@dataclass
class EvaluationConfig:
    mae_acceptance_threshold: float = 0.10


PATHS     = PathsConfig()
SIM_CFG   = SimulationConfig()
MODEL_CFG = ModelConfig()
EVAL_CFG  = EvaluationConfig()


def ensure_directories() -> None:
    for d in [PATHS.artifacts_dir,PATHS.models_dir,PATHS.scalers_dir,PATHS.data_dir]:
        d.mkdir(parents=True, exist_ok=True)