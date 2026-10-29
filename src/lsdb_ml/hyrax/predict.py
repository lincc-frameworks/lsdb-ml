import numpy as np
import pandas as pd
import torch
from dask.distributed import get_client, get_worker
from hyrax.pytorch_ignite import (
    dist_data_loader,
    results_to_numpy,
)

from lsdb_ml.hyrax.worker_plugin import DEFAULT_PLUGIN_NAME


def predict(
    df,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    meta: pd.DataFrame = None,
):
    """Main prediction function, calls `df.map_partitions(_predict)` internally.

    The output schema is read from the `ModelConfig` of the plugin registered
    on the workers of the default Dask client, so the plugin must be registered
    (via `ml_client` or `register_hyrax_plugin`) before calling this.

    The user can optionally pass the `meta` DataFrame to control the output schema,
    but use this with caution - if it is inconsistent with the output_fields defined
    in the plugin's `ModelConfig`, the behavior may be unpredictable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to be used for prediction.
    plugin_name : str, optional
        Name of the plugin to use for prediction, by default DEFAULT_PLUGIN_NAME
    meta : pd.DataFrame, optional
        Metadata DataFrame to use for the output, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the prediction results.

    Raises
    ------
    ValueError
        If the plugin is not registered on every worker of the default client.
    """

    # `predict` runs on the client, so `get_worker()` is unavailable here.
    # Instead, ask the workers for the output fields of the registered plugin.
    results = get_client().run(_get_output_fields, plugin_name=plugin_name)
    if not results or any(fields is None for fields in results.values()):
        raise ValueError(
            f"Plugin '{plugin_name}' is not registered on every worker. "
            "Use `ml_client(...)` or `register_hyrax_plugin(...)` before calling `predict`."
        )
    output_fields = next(iter(results.values()))

    if meta is None:
        meta = df.meta.copy(deep=True).join(output_fields, how="left")

    return df.map_partitions(
        _predict,
        plugin_name=plugin_name,
        output_fields=output_fields,
        meta=meta,
    )


def _get_output_fields(dask_worker, plugin_name):
    """Run on each worker via `Client.run` to fetch the plugin's output fields.

    Dask injects `dask_worker` because of the parameter name. Returns None if
    the plugin is not registered on this worker.
    """
    plugin = dask_worker.plugins.get(plugin_name)
    return None if plugin is None else plugin.model_config.output_fields


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
