from pathlib import Path

import torch
from dask.distributed import get_client
from distributed import Client, WorkerPlugin
from hyrax import Hyrax
from hyrax.models.model_registry import fetch_model_class
from hyrax.models.model_utils import load_model_weights
from hyrax.plugin_utils import load_prepare_inputs
from hyrax.pytorch_ignite import (
    _auto_model,
    create_process_func,
    setup_dataset,
)

from lsdb_ml.hyrax.config_utils import ModelConfig
from lsdb_ml.hyrax.datasets.hyrax_dataframe_dataset import create_meta_dict

DEFAULT_PLUGIN_NAME = "__hyrax_infer"


class TorchModelPlugin(WorkerPlugin):
    """Worker plugin that loads a Hyrax model and prepares it for inference."""

    def __init__(self, name: str = DEFAULT_PLUGIN_NAME, model_config: ModelConfig = None):
        self.name = name  # plugins are keyed by name on the worker
        self.model_config = model_config
        self.model = None
        self.dataset = None
        self.dataset_prepped = False

        config_file = None
        if model_config:
            config_file = model_config.hyrax_config_file
        h = Hyrax(config_file=config_file)

        # call h.set_config("key.sub_key", value) for each field mapping in ModelConfig.FIELD_MAPPING
        for k, v in ModelConfig.FIELD_MAPPING.items():
            h.set_config(v, getattr(model_config, k))

        # Never use caching for inference
        h.set_config("data_set.use_cache", False)
        self.config = h.config

    def setup(self, worker):
        """Set up the worker by loading the Hyrax model and preparing it for inference."""
        model_cls = fetch_model_class(self.config)
        model = model_cls(self.config, None)

        # Load the model weights
        load_model_weights(self.config, model, "infer")

        # Attached the saved `prepare_inputs` function to the model
        model_weights_file = Path(self.config["infer"]["model_weights_file"])
        input_directory = model_weights_file.parent
        model.prepare_inputs = load_prepare_inputs(input_directory)

        model.eval()
        self.model = _auto_model(model)

        # Build the function that is called that processes data
        device = torch.accelerator.current_accelerator()
        if device is None:
            device = torch.device("cpu")
        self.process_fn = create_process_func("infer_batch", device.type, self.model, self.config)

    def teardown(self, worker):
        """Tear down the worker by releasing the model and dataset."""
        self.model = None

    def prep_ml_dataset(self, df):
        """Prepare the ML dataset for inference if it hasn't been prepared yet."""

        # Build the data request for inference
        data = {
            "dataset_class": "DataframeDataset",
            "data_location": create_meta_dict(df),
            "primary_id_field": self.model_config.primary_id_field,
        }

        # Include fields if provided, excluding uses _all_ available fields by default
        if self.model_config.input_fields is not None:
            data["fields"] = self.model_config.input_fields

        data_request = {"infer": {"data": data}}
        self.config["data_request"] = data_request

        if not self.dataset_prepped:
            datasets = setup_dataset(self.config, shuffle=False)
            self.dataset = datasets["infer"]
            self.dataset_prepped = True


def register_hyrax_plugin(
    client: Client,
    model_config: ModelConfig = None,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
):
    """Registers the worker plugin with the Dask client.

    Parameters
    ----------
    client : Client
        The Dask client to register the plugin with.
    model_config : ModelConfig, optional
        Configuration object for the Hyrax model, by default None.
    plugin_name : str, optional
        Name of the plugin to register, by default DEFAULT_PLUGIN_NAME.

    Raises
    ------
    ValueError
        If the id_field is not specified or not present in the catalog.
    """

    if client is None:
        client = get_client()

    client.register_plugin(TorchModelPlugin(plugin_name, model_config))
