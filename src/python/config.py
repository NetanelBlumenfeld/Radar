from typing import Callable, Optional

from pydantic import BaseModel
from utils.utils_images import Normalization


class TrainCfg(BaseModel):
    epochs: int = 300


class PathsCfg(BaseModel):
    data_dir: str = "/home/aviran/netanel/Radar/11G/"
    output_dir: str = "/home/aviran/netanel/Radar/outputs/"
    checkpoint: Optional[str] = None


class ExperimentCfg(BaseModel):
    verbose: int = 0
    en_cm: bool = False


class DataTransformCfg(BaseModel):
    down_sample_factor: int = 4
    back_to_original_dim: bool = False
    pix_norm: Normalization = Normalization.Range_0_1

    def __str__(self) -> str:
        return f"ds_{self.down_sample_factor}_original_dim_{self.back_to_original_dim}_pix_norm_{self.pix_norm}"


class DataCfg(BaseModel):
    task: str = "sr"
    gestures: list[str] = ["Circle"]
    people: int = 2
    data_type: str = "npy"
    transform_pipeline: Optional[Callable] = None
