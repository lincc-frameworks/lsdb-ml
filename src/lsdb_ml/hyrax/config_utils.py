from dataclasses import dataclass

# Consider the two use cases for configs:
# 1) A user has trained a model and is running inference with that model
# 2) A user was given a trained model and wants to run inference with it


@dataclass
class ModelConfig:
    """Configuration class for a Hyrax model used for inference.

    Parameters
    ----------
    model : str
        The name of the model.
    model_weights_file : str
        Path to the model weights file.
    output_fields : object
        The output fields for the model.
    batch_size : int, optional
        The batch size for inference. Defaults to 64.
    primary_id_field : str, optional
        The primary ID field. Defaults to "_healpix_29".
    input_fields : list, optional
        The input fields for the model. Defaults to None.
    hyrax_config_file : str, optional
        Path to the Hyrax configuration file. Defaults to None.
    """

    # Required fields
    model: str
    model_weights_file: str
    output_fields: object

    # Optional fields
    batch_size: int = 64
    primary_id_field: str = "_healpix_29"
    input_fields: list = None
    hyrax_config_file: str = None

    # These are special fields that have a direct mapping to the Hyrax configuration.
    # Feels brittle, but the goal is simplicity. For more complex configurations
    # the user should prefer to call Hyrax verbs directly.
    FIELD_MAPPING = {
        "model": "model.name",
        "model_weights_file": "infer.model_weights_file",
        "batch_size": "data_loader.batch_size",
    }
