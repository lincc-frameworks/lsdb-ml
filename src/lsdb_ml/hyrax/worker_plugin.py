from pathlib import Path

import hyrax
import numpy as np
import torch
from distributed import Client, WorkerPlugin, get_worker
from hyrax.models.model_registry import fetch_model_class
from hyrax.models.model_utils import load_model_weights
from hyrax.plugin_utils import load_prepare_inputs
from hyrax.pytorch_ignite import _auto_model, create_process_func, dist_data_loader, setup_dataset

from lsdb_ml.hyrax.datasets.hyrax_dataframe_dataset import create_meta_dict

DEFAULT_PLUGIN_NAME = "__hyrax_infer"


class TorchModelPlugin(WorkerPlugin):
    """Worker plugin that loads a Hyrax model and prepares it for inference."""

    def __init__(self, name=DEFAULT_PLUGIN_NAME, config=None):
        self.name = name  # plugins are keyed by name on the worker
        self.config = config or {}
        self.model = None
        self.dataset = None
        self.dataset_prepped = False

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
        self.process_fn = create_process_func("infer_batch", "mps", self.model, self.config)

    def teardown(self, worker):
        """Tear down the worker by releasing the model and dataset."""
        self.model = None

    def prep_ml_dataset(self, df, primary_id_field="_healpix_29", fields=None):
        """Prepare the ML dataset for inference if it hasn't been prepared yet."""

        # Build the data request for inference
        data = {
            "dataset_class": "DataframeDataset",
            "data_location": create_meta_dict(df),
            "primary_id_field": primary_id_field,
        }

        # Include fields if provided, excluding uses _all_ available fields by default
        if fields is not None:
            data["fields"] = fields

        data_request = {"infer": {"data": data}}
        self.config["data_request"] = data_request

        if not self.dataset_prepped:
            datasets = setup_dataset(self.config, shuffle=False)
            self.dataset = datasets["infer"]
            self.dataset_prepped = True


def _predict_with_worker(df, process_fn, dataset, config):
    """Inner function that performs inference on a single DataFrame partition.

    TODO: Need to handle cases where the model output is not a bare tensor but a
    dictionary of tensors.
    TODO: Figure out how to handle mismatch between the model output and the
    DataFrame meta. The user will update the meta for the dataframe and the column
    names _should_, but might not, match the keys in the model output.
    """
    with torch.no_grad():
        # Note that `dataset` is a Hyrax DataProvider that is passed to `dist_data_loader`
        # It is not the bare dataset class.
        dataset.prepped_datasets["data"].data = df

        #! Be aware, this is Pytorch-ignite.
        data_loader = dist_data_loader(dataset, config)

        # NOTE: This is a placeholder, but we need to be more flexible because values
        # returned from a model can be 1) a bare tensor or 2) a dictionary of tensors.
        z = np.full(len(df), np.nan, np.float32)
        current_idx = 0

        for batch in data_loader:
            # The first `None` parameter is a placeholder for the pytorch ignite engine
            model_out = process_fn(None, batch)
            if len(batch):
                batch_size = len(batch["object_id"])

                # Recall that the output of the model can be a bare tensor or a dictionary of tensors
                # So we should really use `results_to_numpy` from PR 1076
                z[current_idx : current_idx + batch_size] = model_out.to("cpu").numpy()
                current_idx += batch_size

    df["pred"] = z
    return df


def predict(
    df,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    primary_id_field: str = "_healpix_29",
    fields: list[str] | None = None,
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
    primary_id_field : str, optional
        The primary ID field in the DataFrame, by default "_healpix_29".
    fields : list of str, optional
        List of fields to include in the inference dataset, by default None.
    """

    # Prep dataset for inference on the first call, only instantiated once.
    get_worker().plugins[plugin_name].prep_ml_dataset(df=df, primary_id_field=primary_id_field, fields=fields)

    return _predict_with_worker(
        df,
        get_worker().plugins[plugin_name].process_fn,
        get_worker().plugins[plugin_name].dataset,
        get_worker().plugins[plugin_name].config,
    )


def register_hyrax_plugin(
    client: Client,
    config: dict = None,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
):
    """Prepares the Hyrax instance and registers the worker plugin with the Dask client.

    This function sets up the Hyrax configuration, builds the data request for inference,
    and registers the TorchModelPlugin with the specified client.

    Parameters
    ----------
    client : Client
        The Dask client to register the plugin with.
    config : dict, optional
        Configuration dictionary for Hyrax, by default None.
    plugin_name : str, optional
        Name of the plugin to register, by default DEFAULT_PLUGIN_NAME.

    Raises
    ------
    ValueError
        If the id_field is not specified or not present in the catalog.
    """

    # For this `config` to work, we need to update Hyrax so that it can accept a plain dictionary
    if config is None:
        config = {}
    h = hyrax.Hyrax()

    # call h.set_config("key.sub_key", value) for each key-value pair in the config dictionary
    if config is not None:
        for key, value in config.items():
            for sub_key, sub_value in value.items():
                h.set_config(f"{key}.{sub_key}", sub_value)

    # Never use caching for inference
    h.set_config("data_set.use_cache", False)

    # ^ Now that this is not a function that a user would call, but instead is called
    # ^ by the `ml_client` context manager, can we safely assume that a Dask client
    # ^ is available and remove the next 3 lines?
    from dask.distributed import get_client

    if client is None:
        client = get_client()

    client.register_plugin(TorchModelPlugin(plugin_name, h.config))
