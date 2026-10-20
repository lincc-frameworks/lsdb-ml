from pathlib import Path

import numpy as np
import torch
from distributed import Client, WorkerPlugin, get_worker
from hyrax.models.model_registry import fetch_model_class
from hyrax.models.model_utils import load_model_weights
from hyrax.plugin_utils import load_prepare_inputs
from hyrax.pytorch_ignite import _auto_model, create_process_func, dist_data_loader, setup_dataset

DEFAULT_PLUGIN_NAME = "__hyrax_infer"


class TorchModelPlugin(WorkerPlugin):
    def __init__(self, name=DEFAULT_PLUGIN_NAME, config={}):
        self.name = name  # plugins are keyed by name on the worker
        self.config = config
        self.model = None
        self.dataset = None

    def setup(self, worker):
        model_cls = fetch_model_class(self.config)
        model = model_cls(self.config, None)

        # Load the model weights
        load_model_weights(self.config, model, "infer")

        # Attached the saved `prepare_inputs` function to the model
        model_weights_file = Path(self.config["infer"]["model_weights_location"])
        input_directory = model_weights_file.parent
        model.prepare_inputs = load_prepare_inputs(input_directory)

        model.eval()
        self.model = _auto_model(model)

        # Build the function that is called that processes data
        self.process_fn = create_process_func("infer_batch", "mps", self.model, self.config)

        # setup the dataset class
        datasets = setup_dataset(self.config, shuffle=False)
        self.dataset = datasets["infer"]

    def teardown(self, worker):
        self.model = None


def _predict_with_worker(df, process_fn, dataset, config):
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


def predict(df, plugin_name=DEFAULT_PLUGIN_NAME):
    return _predict_with_worker(
        df,
        get_worker().plugins[plugin_name].process_fn,
        get_worker().plugins[plugin_name].dataset,
        get_worker().plugins[plugin_name].config,
    )


def register_hyrax_plugins(client: Client, config: dict, plugin_name: str = DEFAULT_PLUGIN_NAME):
    client.register_plugin(TorchModelPlugin(plugin_name, config))
