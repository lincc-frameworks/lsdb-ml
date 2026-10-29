@staticmethod
def prepare_inputs(data_dict) -> tuple:
    """This function converts structured data to the input tensor we need to run

    Parameters
    ----------
    data_dict : dict
        The dictionary returned from our data source
    """
    # Necessary for e2e tests to pass since this function ends up written out as a file and
    # must stand alone from imports when loaded for inference
    import numpy as np  # noqa: F811

    data = data_dict.get("data", {})

    image = data.get("ra_obj", np.ndarray([]))

    # make sure that image dtype is float32
    return image.astype(np.float32)
