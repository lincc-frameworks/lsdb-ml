# lsdb-ml MVP: column spec and inference

Status: implemented on this branch. This document covers only what the MVP ships. The fuller
design, which also covers training, is on the `design/initial-design` branch; the parts of it
that this MVP leaves out are listed in §7.

## 1. Scope

This MVP runs a PyTorch model over every row of an LSDB catalog on Dask workers. It returns
a lazy catalog of predictions that has correct `meta` and is aligned row by row with the
input. Two pieces make that up:

- **Column spec:** maps flat and nested catalog columns to tensors, validated against
  catalog `meta` before any data is read.
- **Inference:** a Dask `WorkerPlugin` installs the model once per worker, and
  `map_partitions` runs it partition by partition.

Training, the Hyrax backend, and running without a `distributed` client are out of scope.
`hyrax` remains only an optional extra.

## 2. API

```python
import torch
from distributed import Client
from lsdb_ml import ColumnSpec, Flat, ModelDeployment, ModelSpec, Nested

client = Client(...)  # a dask.distributed client is required

spec = ColumnSpec({
    "mag": Flat("mag"),                                              # -> (B,)
    "flux": Nested("lc", "flux", max_length=32, truncate="tail"),    # -> (B, L), flux_mask, flux_lengths
})

with ModelDeployment(ModelSpec(MyModel, weights="model.pt")) as deployed:
    scored = deployed.predict(
        catalog, spec,
        outputs={"score": "float32"},
        inputs=["mag", "flux", "flux_mask"],   # batch keys passed positionally to the model
        batch_size=4096,
    )
    result = scored.compute()
```

`predict(catalog, model, spec, outputs, ...)` is the one-call form of the same thing. It
leaves the model installed, because the returned catalog is lazy. `ModelDeployment.close()`
removes the model from the workers.

## 3. Column spec

- **Fields.** `Flat(columns, dtype=float32)` stacks numeric columns into a `(B,)` or
  `(B, F)` tensor. `Nested(column, fields, dtype=float32, max_length=None, truncate="raise")`
  reads numeric sub-columns of a nested column.
- **Validation before compute.** `ColumnSpec.resolve(catalog)` uses only `catalog.meta`.
  Meta is an empty `NestedFrame` with full dtypes, and `NestedDtype.column_dtypes` gives the
  sub-column types. That replaces the hand-built column preview in hyrax#1062. Resolve checks:
  - columns and sub-columns exist;
  - flat vs nested is used correctly;
  - dtypes are numeric;
  - output keys don't collide.

  Every problem is reported in one `SpecError`.
- **Nested representation: zero padding with a mask.** Each batch is padded to its longest
  row and returns `<key>` `(B, L[, F])`, `<key>_mask` `(B, L)` and `<key>_lengths` `(B,)`.
  This works directly with attention masks, masked pooling and `pack_padded_sequence`.
  `torch.nested` is not mature enough to be the default. Other representations can be added
  later as new `Field` subclasses.
- **Collation is vectorized.** It is built from `series.nest.len()` (null rows count as 0)
  and `series.nest.to_flat()`, with no per-row Python.
- **Missing values raise.** Drop or fill them in the catalog first.
- **Every batch includes `_index`.** It holds the frame index (`_healpix_29`) as int64.

## 4. Inference lifecycle

- **Model spec.** `ModelSpec(factory, weights=None, device="cpu", torch_threads=None)`. Only
  the factory and the weights path go to the scheduler. Each worker builds the model, loads
  the `state_dict`, calls `eval()` and moves it to the device. The device is resolved *on the
  worker*: `"auto"` picks CUDA, then MPS, then CPU. A bare `nn.Module` is also accepted; it is
  pickled into the plugin once per worker.
- **Plugin.** `ModelPlugin(WorkerPlugin)` is named `lsdb-ml-model-<hash of spec>`. Concurrent
  deployments therefore don't collide, and redeploying the same spec to the same client
  doesn't rebuild the model. Workers that join later get the plugin automatically.
- **Tasks.** Each task looks the model up with `get_worker().plugins[name]`. If the model is
  missing, the task raises a clear error, e.g. when the catalog is computed on another client
  or after `close()`.
- **Concurrency.** Tasks on one worker share only the read-only model, under
  `torch.inference_mode()`. GPU workers should use `threads_per_worker=1`.

## 5. Alignment (the hyrax#1062 "shuffled results" issue)

The prototype kept one mutable dataset per worker, overwrote it in each task, and wrote
predictions back by position into a buffer. Concurrent tasks on a worker could race on that
shared state. Here, each task:

1. holds only task-local state;
2. walks its partition in order, in `iloc` slices of `batch_size` rows;
3. checks every batch's output for exactly one row per input row;
4. assigns the concatenated predictions **by position** to the same partition frame they came
   from.

Predictions are never joined on `_healpix_29`, which is not guaranteed unique. The output
`meta` comes from `catalog.meta` plus the `OutputSpec` dtypes. Columns are built with
`pd.Series(values, dtype=d).array`, so computed dtypes match the meta exactly.

## 6. Tests

All tests run on CPU, take about 16 seconds, and use a synthetic nested catalog and an
in-process 2-worker × 2-thread cluster.

- **Spec validation:** every error case, including a fixture catalog that raises if any
  partition is computed.
- **Collation:** checked against a per-row reference, covering truncation, empty and null
  nested rows, missing values, and row order.
- **Alignment:** a filter that empties partitions, then a reordered partition subset; and a
  catalog where every row shares one `_healpix_29` value. Both are checked per row and joined
  on object id. Injecting a reversed batch order makes these tests fail.
- **Load once per worker:** the model is built exactly once per worker across many
  partitions and repeated predictions.
- **Meta:** dtypes of the lazy result match the computed result.
- **Failure modes:** wrong output row count, closed deployment, no client, bad arguments.

## 7. Deferred (in the full design)

| Item | Why deferred |
|---|---|
| Training: `CatalogIterableDataset` (partition sharding across DataLoader workers and DDP) and `TrainSession` | Larger surface; depends on upstream `CatalogStream` changes. Planned as two follow-up PRs. |
| `transform=` hooks, `pad_to`/`pad_value`, missing-value filling | Convenience; easy to add back when needed. |
| Dict-valued and list-column model outputs | Not needed for the first use cases. |
| Running without a `distributed` client | A second code path; requiring a `Client` is a reasonable MVP constraint. |
| Hyrax backend | The seams already exist: the model factory in `ModelSpec` and `inputs=` (a key list or callable). |
| Automatic output meta from a dummy forward pass | Fragile for models that can't take an empty or synthetic batch. |

## 8. CI note

The workflows only trigger on `main`, so PRs into `dev` get no CI. Add `dev` to the
`push`/`pull_request` branch lists in `testing-and-coverage.yml`, `pre-commit-ci.yml` and
`build-documentation.yml`. With torch as a dependency, CI should also install the CPU build
of torch, because Linux defaults to the multi-GB CUDA wheels.
