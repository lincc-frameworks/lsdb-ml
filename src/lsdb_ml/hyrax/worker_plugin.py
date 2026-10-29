from pathlib import Path

import hyrax
import numpy as np
import torch
from dask.distributed import get_client
from distributed import Client, WorkerPlugin, get_worker
from hyrax.models.model_registry import fetch_model_class
from hyrax.models.model_utils import load_model_weights
from hyrax.plugin_utils import load_prepare_inputs
from hyrax.pytorch_ignite import (
    _auto_model,
    create_process_func,
    dist_data_loader,
    results_to_numpy,
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
        h = hyrax.Hyrax(config_file=config_file)

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


def _predict_with_worker(df, process_fn, dataset, config, output_fields):
    """Inner function that performs inference on a single DataFrame partition."""
    with torch.no_grad():
        # Here we assign the current dataframe to the dataset's data attribute
        # for inference.
        # NOTE: `dataset` is a Hyrax DataProvider passed to `dist_data_loader`,
        # not the bare dataset class.
        dataset.prepped_datasets["data"].data = df

        # ^ We could create our own dataloader here instead of using Hyrax's dist_data_loader
        # ^ This might provide a small efficiency gain by avoid repeatedly creating
        # ^ a dataloader here.
        data_loader = dist_data_loader(dataset, config)

        # Preallocate the output columns so each batch can be written in place.
        out = df.copy(deep=False)
        for col in output_fields.columns:
            # TODO: Probably need to be more clever about the dtypes
            # i.e. what if a model returns an image?
            out[col] = np.full(len(out), np.nan, output_fields[col].dtype)

        current_idx = 0
        for batch in data_loader:
            # Hyrax ensures a "object_id" key is in the batch dictionary
            batch_size = len(batch["object_id"])
            if batch_size:
                # The first `None` parameter is a placeholder for the pytorch ignite engine
                model_out = results_to_numpy(process_fn(None, batch))

                rows = slice(current_idx, current_idx + batch_size)

                # `model_out` is either a dictionary of arrays (keyed by column
                # name) or a bare array, which is written to the single output column.
                if not isinstance(model_out, dict):
                    model_out = {output_fields.columns[0]: model_out}

                # TODO: Put some guards here to explode and notify the user if
                # the keys in `model_out` do not match the expected output columns.
                for key, value in model_out.items():
                    # Positional write, independent of the DataFrame's index labels
                    out.iloc[rows, out.columns.get_loc(key)] = value
                current_idx += batch_size

    return out


def _predict(
    df,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    output_fields=None,
):
    """This is UDF that is provided to `catalog.map_partitions(...).
    It will perform inference using the Hyrax model registered with the worker
    plugin.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data for inference.
    plugin_name : str, optional
        Name of the registered Hyrax plugin to use, by default DEFAULT_PLUGIN_NAME.
    output_fields : pd.DataFrame
        DataFrame specifying the output fields and their dtypes, by default None.
    """

    # Prep dataset for inference on the first call, only instantiated once.
    if not get_worker().plugins[plugin_name].dataset_prepped:
        get_worker().plugins[plugin_name].prep_ml_dataset(df)

    return _predict_with_worker(
        df,
        get_worker().plugins[plugin_name].process_fn,
        get_worker().plugins[plugin_name].dataset,
        get_worker().plugins[plugin_name].config,
        output_fields,
    )


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


def predict(
    df,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
):
    """Main prediction function, should call `df.map_partitions(_predict)`
    internally.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to be used for prediction.
    plugin_name : str, optional
        Name of the plugin to use for prediction, by default DEFAULT_PLUGIN_NAME

    Returns
    -------
    pd.DataFrame
        DataFrame containing the prediction results.
    """

    # ^ Having trouble getting the worker here.
    output_fields = get_worker().plugins[plugin_name].model_config.output_fields

    meta = df._operation.meta.copy(deep=True).join(output_fields, how="left")

    return df.map_partitions(
        _predict,
        plugin_name=plugin_name,
        output_fields=output_fields,
        meta=meta,
    )
