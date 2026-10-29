from contextlib import contextmanager

from distributed import Client

from lsdb_ml.hyrax.config_utils import ModelConfig
from lsdb_ml.hyrax.worker_plugin import register_hyrax_plugin


@contextmanager
def ml_client(model_config: ModelConfig = None, **kwargs):
    """Context manager that instantiates a client and then preps it for prediction
    by starting a Hyrax model inside the client.

    Parameters
    ----------
    model_config : ModelConfig, optional
        Configuration object for the Hyrax model.
    **kwargs : dict
        Additional keyword arguments to pass to the Dask client.

    Yields
    ------
    client : distributed.Client
        The initialized Dask client with the Hyrax model registered.
    """

    client = None
    try:
        # Initialize the standard client here
        client = Client(**kwargs)
        print(client.dashboard_link)

        # Register the Hyrax plugin to make it a ml_client
        register_hyrax_plugin(client=client, model_config=model_config)

        yield client
    finally:
        if client is not None:
            client.close()
