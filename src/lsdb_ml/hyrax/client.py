from contextlib import contextmanager

from distributed import Client

from lsdb_ml.hyrax.worker_plugin import register_hyrax_plugin


@contextmanager
def ml_client(
    hyrax_config=None,
    **kwargs,
):
    """Context manager that instantiates a client and then preps it for prediction
    by starting a Hyrax model inside the client.

    Parameters
    ----------
    hyrax_config : dict, optional
        Configuration dictionary for Hyrax.
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
        register_hyrax_plugin(
            client=client,
            config=hyrax_config,
        )

        yield client
    finally:
        if client is not None:
            client.close()
