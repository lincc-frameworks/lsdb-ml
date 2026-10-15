"""Running models over LSDB catalogs on Dask workers."""

from .outputs import OutputSpec
from .plugin import ModelPlugin, ModelSpec
from .predict import ModelDeployment, predict

__all__ = ["ModelDeployment", "ModelPlugin", "ModelSpec", "OutputSpec", "predict"]
