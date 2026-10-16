# lsdb-ml MVP: catalog inference

Status: implemented on this branch. This document covers only what the MVP ships. The fuller
design, which also covers training, is on the `design/initial-design` branch; the parts of it
that this MVP leaves out are listed in §7.

## 1. Scope

This MVP runs a PyTorch model over every row of an LSDB catalog on Dask workers. It returns
a lazy catalog of predictions that has correct `meta` and is aligned row by row with the
input. Two pieces make that up:

- **Inputs by column name:** the model's inputs are plain catalog column names. Whether
  each is flat or nested, and its dtype, are read from catalog `meta`, and everything is
  checked before any data is read.
- **Inference:** a Dask `WorkerPlugin` installs the model once per worker, and
  `map_partitions` runs it partition by partition.

Training, the Hyrax backend, and running without a `distributed` client are out of scope.
`hyrax` remains only an optional extra.

Data preparation (filtering, derived columns, unit conversion, encoding strings) is left to
ordinary LSDB operations before calling `predict`. lsdb-ml has no column-spec layer of its
own; one can be added later if a clear need appears.

## 2. API

```python
import torch
from distributed import Client
from lsdb_ml import ModelDeployment, ModelSpec

client = Client(...)  # a dask.distributed client is required

class MyModel(torch.nn.Module):
    def forward(self, batch):          # batch: dict[str, Tensor] keyed by column name
        mag = batch["mag"]                                       # (B,)
        flux, mask = batch["lc.flux"], batch["lc._mask"]         # (B, L), (B, L)
        ...

with ModelDeployment(ModelSpec(MyModel, weights="model.pt")) as deployed:
    scored = deployed.predict(
        catalog,
        inputs=["mag", "lc.flux"],     # flat column, nested sub-column (LSDB dotted name)
        outputs={"score": "float32"},
        max_length=32, truncate="tail",
        batch_size=4096,
    )
    result = scored.compute()
```

- **One-call form.** `predict(catalog, model, inputs, outputs, ...)` does the same thing.
  It leaves the model installed, because the returned catalog is lazy.
  `ModelDeployment.close()` removes the model from the workers.
- **Models with positional arguments.** These use an adapter,
  `forward=lambda m, b: m(b["mag"], b["lc.flux"], b["lc._mask"])`.

## 3. Inputs and batches

- **Names.** A flat column is named as itself (`"mag"`). A nested sub-column uses LSDB's
  dotted name (`"lc.flux"`), the same form `catalog["lc.flux"]` accepts. Batch keys are
  those same names.
- **Validation before compute.** `plan_inputs(catalog, inputs)` reads only `catalog.meta`.
  Meta is an empty `NestedFrame` with full dtypes, and `NestedDtype.column_dtypes` gives the
  sub-column types. That replaces the hand-built column preview in hyrax#1062. It checks:
  - each name exists;
  - a whole nested column isn't named where a sub-column is needed;
  - dtypes are numeric or bool.

  Every problem is reported in one `InputError`. `predict` then dry-runs the conversion on
  `catalog.meta` itself, which yields zero-row tensors and still reads no data.
- **Dtypes are inferred.** Float columns become `float_dtype` (float32 by default), integers
  int64, and booleans bool.
- **Nested representation: zero padding with a shared mask.** Each nested sub-column
  becomes a `(B, L)` tensor, padded to the longest row in the batch. Sub-columns of one
  nested column share offsets, so they share one `"<column>._mask"` (bool, `(B, L)`) and
  one `"<column>._lengths"` (int64, `(B,)`). This works directly with attention masks,
  masked pooling and `pack_padded_sequence`. `torch.nested` is not mature enough to be the
  default.
- **Sub-columns stay separate tensors.** They aren't stacked into `(B, L, F)`, to keep
  things simple. Models can stack them if they want.
- **Conversion is vectorized.** It is built from `series.nest.len()` (null rows count as 0)
  and `series.nest.to_flat()`, with no per-row Python. Missing values raise; drop or fill
  them in the catalog first.
- **Batch-level padding is always correct, but has two costs.**
  - *Width follows the longest row.* One very long light curve makes its whole batch that
    wide, so memory and compute scale with `B × L_max` per batch. `max_length` with
    `truncate` caps this.
  - *Width varies from batch to batch.* Models must accept a variable `L`. That's natural
    for sequence models and pooling, but a model that needs a fixed `L` (e.g. an MLP over a
    flattened sequence) isn't supported yet. A fixed-width padding option (`pad_to`, or a
    catalog-level `pad_nested`) is the natural follow-up if that's needed.

  Because padding is masked, predictions don't depend on which other rows share a batch,
  as long as the model uses the mask.

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

- **Input validation:** every error case, including a fixture catalog that raises if any
  partition is computed, and dtype inference from meta.
- **Conversion:** checked against a per-row reference, covering the shared mask, truncation,
  empty and null nested rows, missing values, and row order.
- **Alignment:** a filter that empties partitions, then a reordered partition subset; and a
  catalog where every row shares one `_healpix_29` value. Both are checked per row and joined
  on object id. Injecting a reversed batch order makes these tests fail.
- **Load once per worker:** the model is built exactly once per worker across many
  partitions and repeated predictions.
- **Meta:** dtypes of the lazy result match the computed result.
- **`forward=` adapter:** a model taking positional tensors works through it.
- **Failure modes:** wrong output row count, closed deployment, no client, bad arguments.

## 7. Deferred (in the full design)

| Item | Why deferred |
|---|---|
| Training: `CatalogIterableDataset` (partition sharding across DataLoader workers and DDP) and `TrainSession` | Larger surface; depends on upstream `CatalogStream` changes. Planned as two follow-up PRs. |
| A column-spec layer (`ColumnSpec`, transforms, column grouping, missing-value filling) | Data preparation belongs to LSDB operations; add a middle layer only if a clear need appears. |
| Fixed-width padding (`pad_to` or catalog-level `pad_nested`), stacking sub-columns into `(B, L, F)` | Needed only for fixed-`L` models or as an optimization; batch-level padding covers correctness. |
| Dict-valued and list-column model outputs | Not needed for the first use cases. |
| Running without a `distributed` client | A second code path; requiring a `Client` is a reasonable MVP constraint. |
| Hyrax backend | The seams already exist: the model factory in `ModelSpec` and the `forward=` adapter. |
| Automatic output meta from a dummy forward pass | Fragile for models that can't take an empty or synthetic batch. |

## 8. CI note

The workflows only trigger on `main`, so PRs into `dev` get no CI. Add `dev` to the
`push`/`pull_request` branch lists in `testing-and-coverage.yml`, `pre-commit-ci.yml` and
`build-documentation.yml`. With torch as a dependency, CI should also install the CPU build
of torch, because Linux defaults to the multi-GB CUDA wheels.
