# create a context manager that will manage the lifecycle of an ML client,
# including setup and teardown

from contextlib import contextmanager

from distributed import Client

from lsdb_ml.hyrax.worker_plugin import register_hyrax_plugins


@contextmanager
def ml_client(
    catalog=None,
    hyrax_config=None,
    num_workers=1,
    mem_limit="2GB",
    id_field=None,
    fields=None,
):
    """Context manager that instantiates a client and then preps it for prediction
    by starting a Hyrax model inside the client.

    Parameters
    ----------
    catalog : lsdb.Catalog, optional
        The LSDB catalog to use for inference.
    hyrax_config : dict, optional
        Configuration dictionary for Hyrax.
    num_workers : int, default=1
        Number of workers for the Dask client.
    mem_limit : str, default="2GB"
        Memory limit for each worker.
    id_field : str, optional
        The primary ID field in the catalog.
    fields : list of str, optional
        List of fields to include in the inference dataset.

    Yields
    ------
    client : distributed.Client
        The initialized Dask client with the Hyrax model registered.
    """

    client = None
    try:
        # Initialize the ML client here
        client = Client(n_workers=num_workers, threads_per_worker=1, memory_limit=mem_limit)
        print(client.dashboard_link)

        register_hyrax_plugins(
            client=client,
            catalog=catalog,
            config=hyrax_config,
            id_field=id_field,
            fields=fields,
        )

        yield client
    finally:
        if client is not None:
            client.close()
