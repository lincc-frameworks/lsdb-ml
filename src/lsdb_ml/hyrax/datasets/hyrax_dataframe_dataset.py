from types import MethodType

import nested_pandas as npd
import numpy as np
from hyrax.datasets.dataset_registry import HyraxDataset


def create_meta_dict(nested_frame):
    """Convert the metadata of a nested dataframe or LSDB catalog instance into
    dictionary that can be used by the DataframeDataset class"""
    meta_dict = {}

    # This allows a user to pass either the nestedframe or just the metadata
    if isinstance(nested_frame, npd.NestedFrame):
        cols = nested_frame.columns
        nested_cols = nested_frame.nested_columns
    else:  # assume it's an LSDB catalog instance
        cols = nested_frame._operation.meta.columns
        nested_cols = nested_frame._operation.meta.nested_columns

    non_nested_cols = [col for col in cols if col not in nested_cols]

    for col in non_nested_cols:
        meta_dict[col] = str(nested_frame[col].dtype)

    for col in nested_cols:
        meta_dict[col] = {}
        for ncols in nested_frame[col].columns:
            meta_dict[col][ncols] = str(nested_frame[f"{col}.{ncols}"].dtype)

    return meta_dict


class DataframeDataset(HyraxDataset):
    """Experimental dataset that should be used by LSDB to feed nested dataframes
    into Hyrax models"""

    def __init__(self, config, data_location=None):
        super().__init__(config, metadata_table=None)

        if not data_location:
            data_location = npd.NestedFrame([])

        self._data = data_location
        self._update_getters()

    @property
    def data(self):
        """The variable holding the nested dataframe."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def _update_getters(self):
        # Dynamic getter creation - assumes only one level of nesting.

        # get all the top level columns
        all_columns = set()
        for k, v in self._data.items():
            if not isinstance(v, dict):
                all_columns.add(k)

        # get all the top-level _nested_ columns
        nested_columns = set()
        for k, v in self._data.items():
            if isinstance(v, dict):
                nested_columns.add(k)

        self._register_getters(list(all_columns - nested_columns))

        # For each of the nested columns, get all the subcolumns
        for nested_column in nested_columns:
            nested_subcols = list(self._data[nested_column].keys())
            self._register_nested_getters(nested_column, nested_subcols)
            self._register_nested_collators(nested_column, nested_subcols)

    def _register_getters(self, columns) -> None:
        def _make_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                return self._data.iloc[idx][_field_name]

            return getter

        for field_name in columns:
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def _register_nested_getters(self, nested_column, nested_subcolumns) -> None:
        def _make_getter(nested_column: str, subnested_column: str):
            def getter(self, idx, _nested_name=nested_column, _subnested_name=subnested_column):
                return np.asarray(self._data.iloc[idx][_nested_name][_subnested_name])

            return getter

        for subnested_column in nested_subcolumns:
            method_name = f"get_{nested_column}_{subnested_column}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(nested_column, subnested_column), self))

    def _register_nested_collators(self, nested_column, nested_subcolumns) -> None:
        def _make_collator(nested_column: str, subnested_column: str):
            def collator(self, batch, _nested_name=nested_column, _subnested_name=subnested_column):
                # Get the length of each subnested array in the batch
                col_name = f"{_nested_name}_{_subnested_name}"
                lengths = [len(sample[col_name]) for sample in batch]
                max_length = max(lengths)

                # Create a padded array and mask with shape (# arrays, max length)
                padded_batch = np.zeros((len(batch), max_length))
                mask = np.zeros_like(padded_batch, dtype=bool)
                for i, sample in enumerate(batch):
                    padded_batch[i, : lengths[i]] = sample[col_name]
                    mask[i, : lengths[i]] = True
                return {
                    col_name: padded_batch,
                    f"{col_name}_mask": mask,
                }

            return collator

        for subnested_column in nested_subcolumns:
            method_name = f"collate_{nested_column}_{subnested_column}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_collator(nested_column, subnested_column), self))

    def __getitem__(self, idx):
        """Currently required by Hyrax machinery, but likely to be phased out."""
        return {}

    def __len__(self) -> int:
        """Return the number of records in the CSV."""
        return len(self._data)
