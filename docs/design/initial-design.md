# lsdb-ml: initial design for LSDB ↔ PyTorch training and inference

Status: **approved with changes** (see "Review decisions" below). Implementation in progress.

### Review decisions

1. **Device:** `"cpu"` is the default for training and inference; `"auto"` is opt-in.
2. **Batching follows Dask's chunks.** A batch never crosses a chunk boundary. Each chunk
   from the stream (one or more stacked partitions) is yielded whole (`batch_size=None`)
   or cut into consecutive slices of at most `batch_size` rows. There is no cross-chunk
   carry-over buffer, so ragged batches are normal. Uneven step counts across DDP ranks
   are handled with `model.join()`.
3. **Shuffling stays as `CatalogStream` provides it.** No cross-chunk shuffle buffer.
   Useful upstream `CatalogStream` changes are collected in §14 rather than worked around.
4. **Upstream issues are not opened from this work.** Their rough shape is recorded in
   §14 for the maintainer to file.

## 1. Goals and non-goals

lsdb-ml connects LSDB catalogs to PyTorch, for both training and inference. This first
round covers LSDB plus plain PyTorch only. Hyrax is a planned *optional* backend: the
design leaves a seam for it (§8), but nothing here imports `hyrax` or needs it installed.

In scope:

- **Shared:** a column spec that maps flat and nested catalog columns to tensors. It is
  validated against catalog `meta` before any data is computed (§4).
- **Training:** a `torch.utils.data.IterableDataset` backed by an LSDB catalog. It is
  correct under `DataLoader(num_workers>0)` and under DDP sharding, and it supports both
  finite epochs and open-ended streaming (§5). On top of it sits a context-manager
  training session modeled on Hyrax's `train_stream` (§6).
- **Inference:** `predict(catalog, model, ...)`. It installs the model once per Dask
  worker through a `WorkerPlugin` and returns a lazy `lsdb.Catalog` that is index-aligned
  with the input and has correct `meta` (§7).

Out of scope: Hyrax integration; MLflow and TensorBoard (a callback hook is provided
instead); gradient accumulation; validation streams; distributed training beyond basic
DDP-correct sharding; resuming from a mid-stream data position.

## 2. What the investigation found

These facts were checked against lsdb 0.11.0, nested-pandas 0.7.2, dask/distributed
2026.8.0 and torch 2.14.1, using small synthetic catalogs. The probe scripts are not
committed. Each fact below drives a design decision.

**`lsdb.streams.CatalogStream`** (in `src/lsdb/streams/catalog_streams.py`):

1. **Prefetch depth is exactly one chunk.** `CatalogIterator.__next__` submits the next
   chunk's future before it returns the current chunk. With a `distributed` client this
   overlaps fetching with compute. With `client=None`, it computes the *next* chunk inline
   before returning the current one, so nothing overlaps.
2. **The unshuffled order is reversed.** `get_next_partitions` pops from the tail of
   `partitions_left`, so `shuffle=False` streams partitions in reverse pixel order
   (verified).
3. **`shuffle=True` shuffles twice.** It permutes the partition order and also
   row-shuffles each chunk (`result.sample(frac=1)`). Rows only ever mix within one chunk
   of `partitions_per_chunk` partitions.
4. **Each `iter()` reseeds.** Every `iter(stream)` spawns a child RNG, so re-iterating
   gives a new shuffle that is still deterministic for a given seed.
5. **There is no sharding hook and no way to pass an explicit partition order.**
6. **`InfiniteStream` draws chunks independently.** Each chunk is a fresh random draw of
   partitions, without replacement inside the chunk but *with* replacement across chunks.
   It has no notion of epochs or shards.

**Catalog and partition APIs:**

7. **`catalog.partitions[[3, 0, 5]]` re-sorts to pixel order.** It goes through
   `PixelSearch`, so a subset does not keep the requested order (verified). Code must
   never assume that it does.
8. **Partition indexes are unique but not monotonic.** `_healpix_29` is unique per
   partition in our tests but not sorted within a partition. In general `_healpix_29` is
   not guaranteed to be unique.
9. **Catalogs pickle and cloudpickle cleanly.** A `from_dataframe` toy catalog is about
   73 KB. A spawned DataLoader worker can rebuild the catalog and compute its own
   partitions with Dask's synchronous scheduler. With 0, 2 and 3 workers, partition-level
   sharding produced every row exactly once (verified).

**Meta and nested dtypes, the blocker in #1062:**

10. **Full schema is available without computing data.** `catalog.meta` is an empty
    `NestedFrame` with complete dtypes. A nested column's dtype is a `NestedDtype`, and
    `NestedDtype.column_dtypes` returns `{subcolumn: pyarrow type}`, e.g.
    `{'t': double, 'flux': float, 'band': large_string}`. This is everything #1062's
    `create_meta_dict` was trying to get, so getters and collation can be built up front.
11. **Nested data can be read in vectorized form.** `series.nest.to_flat([...])` returns
    the flat values, and the nested extension array exposes `list_offsets`. Together they
    allow vectorized padding with no per-row Python.
12. **Empty nested lists can come back as null.** Rows whose nested list is empty can be
    null (`isna()` is True) rather than zero-length. Collation must treat a null nested
    row as length 0.
13. **Arrow-backed `to_numpy()` can return read-only arrays.** `torch.as_tensor` warns on
    these, so collation must produce writable arrays.

**`distributed.WorkerPlugin`:**

14. **`setup()` runs once per worker.** This holds for both `processes=False` and
    `processes=True` clusters. Inside a task, `get_worker().plugins[name]` reaches the
    worker's plugin instance. Registering again under the same name replaces the plugin,
    which tears down the old one and runs setup on the new one.

**Prior art in Hyrax:**

15. **Hyrax already has an `LSDBStreamDataset`.** It wraps `CatalogStream`, buffers to
    exact batch sizes, and pads and masks nested columns. It also **forces
    `num_workers=0`**, because a plain `IterableDataset` would otherwise duplicate rows in
    every worker. lsdb-ml removes that restriction (§5).

**On #1062's "results appeared to be shuffled":** the prototype keeps **one mutable
dataset object per worker plugin**. Each task overwrites it
(`dataset.prepped_datasets["data"].data = df`) and then writes predictions back
positionally into a preallocated buffer. Whenever a worker runs more than one task
thread, two partitions can race on that shared object. Any reordering inside the loader
would also misplace results. I did not reproduce the exact symptom, but both are
structural risks, and §7.4 removes them by construction.

## 3. Package layout and dependencies

```
src/lsdb_ml/
    __init__.py            # re-exports the public API below
    spec.py                # ColumnSpec, Flat, Nested, ResolvedSpec (validation against meta)
    collate.py             # frame -> dict[str, Tensor]; padding helpers
    device.py              # resolve_device("cpu" | "auto" | "cuda[:n]" | "mps" | torch.device)
    data/
        sharding.py        # ShardInfo, rank/worker discovery, per-epoch partition assignment
        source.py          # PartitionSource protocol + CatalogStreamSource (wraps lsdb CatalogStream)
        batching.py        # iter_slices: cuts each chunk into <= batch_size row slices
        dataset.py         # CatalogIterableDataset
    training/
        steps.py           # TrainStep protocol, SupervisedStep
        session.py         # TrainSession
        checkpoint.py      # atomic save/load of model + step state + counters
    inference/
        outputs.py         # OutputSpec -> output meta; per-partition output validation
        plugin.py          # ModelSpec, ModelPlugin(WorkerPlugin)
        predict.py         # predict(), ModelDeployment
    backends/
        base.py            # Backend protocol (the PyTorch-vs-Hyrax seam)
        torch_backend.py   # TorchBackend (the only implementation this round)
```

Public API at the top level: `ColumnSpec`, `Flat`, `Nested`, `CatalogIterableDataset`,
`TrainSession`, `SupervisedStep`, `predict`, `ModelSpec`, `OutputSpec`.

Dependencies (these changes go in the implementation PRs, not this one):

- **Core:** add `torch` and `lsdb>=0.11`. lsdb already brings in nested-pandas, pyarrow,
  and `dask[complete]`, which includes `distributed`. Proposed torch floor: `>=2.4`, to be
  confirmed against CI. No other runtime dependencies.
- **`hyrax`:** stays an optional extra; it is already declared that way in
  `pyproject.toml`.
- **Test-only:** no new packages are needed.

## 4. Column spec and collation (shared)

### 4.1 API

```python
from lsdb_ml import ColumnSpec, Flat, Nested

spec = ColumnSpec(
    {
        "x": Flat(["ra", "dec", "mag"]),          # -> float32 (B, 3)
        "y": Flat("redshift"),                     # -> float32 (B,)
        "lc": Nested(
            "lightcurve", ["mjd", "flux", "flux_err"],
            max_length=512, truncate="tail",       # -> lc (B, L, 3), lc_mask (B, L), lc_lengths (B,)
        ),
    },
    id_column="objectId",                          # optional passthrough -> "id" (B,)
)

resolved = spec.resolve(catalog.meta)   # validates; no compute; raises with every problem listed
resolved.columns                        # top-level columns needed: used to project the catalog
resolved.describe()                     # output keys -> (dtype, shape), useful for building the model
batch = resolved.collate(nested_frame)  # dict[str, torch.Tensor]
```

Every batch also carries `"_index"`, an int64 `(B,)` tensor of the frame index
(`_healpix_29`). Batches can always be traced back to rows, and tests use it to check for
duplicated or dropped rows.

### 4.2 Validation against meta (no compute)

`spec.resolve(meta)` checks the following. Every error is collected and raised at once in
a single `SpecError`.

- Each column exists. If a column is missing from `catalog.columns` but present in
  `catalog.all_columns`, the hint says it exists on disk and was not loaded.
- `Flat` columns are not nested. `Nested` columns are nested, and every sub-column exists
  in `NestedDtype.column_dtypes`.
- Every dtype is numeric or bool (pyarrow int, uint, float, bool), or else the field has a
  `transform=` hook. Strings and timestamps are rejected with a message that points at
  `transform=`.
- A `Flat` with several columns needs mutually castable dtypes. Output dtype defaults to
  `float32`; `dtype=` overrides it.
- Output keys are unique, including derived keys such as `lc_mask` and `lc_lengths`.

`resolve` is pure, so the dataset constructor and `predict` both call it eagerly, and a
bad spec fails before Dask does any work.

### 4.3 Variable-length nested columns: padding with a mask

**Choice: pad to the longest row in the batch, and return a boolean mask plus lengths.**

Why:

- **Every consumer understands it.** Transformer `src_key_padding_mask`, masked pooling
  and `pack_padded_sequence(lengths=...)` all take this form directly.
- **Rectangular tensors move cheaply.** They pin, cross the DataLoader IPC boundary, and
  move to GPU without custom handling.
- **Padding waste is bounded.** Padding to the batch maximum, not the catalog maximum,
  keeps waste local to the batch. `max_length=` caps the worst case, with
  `truncate="head" | "tail" | "raise"`. `pad_to=int` gives fixed shapes, e.g. for
  `torch.compile`.
- **torch's ragged layout is not ready.** Many ops are unsupported for
  `torch.nested`/jagged tensors, so they cannot be the default yet.

The mask is the only reliable record of real data, because padding uses `pad_value`
(default 0). This matches Hyrax's `LSDBStreamDataset` convention, so a later Hyrax
backend sees familiar shapes.

**Extensibility:** `Flat` and `Nested` implement a small `Field` protocol:
`resolve(meta) -> ResolvedField` and `collate(frame) -> dict[str, Tensor]`. Later options
can be added as new `Field` types without touching the dataset or inference code, for
example `NestedRagged` (values plus offsets, or `torch.nested` jagged), or user subclasses.

**Implementation:** fully vectorized per batch. Read `list_offsets` to get the lengths and
`to_flat(subcols)` to get the values as one writable numpy array. Build row and position
index arrays and scatter into `(B, L, F)` with a single fancy-index assignment. Null
nested rows count as length 0 (finding 12). Null flat values follow a per-field policy:
`null="raise"` (default) or a fill value. Reading `list_offsets` goes through
nested-pandas internals, so it is isolated in one helper function in case that API moves.

## 5. Training data path

### 5.1 Pipeline

```
lsdb.Catalog (lazy)
  └─ project to resolved.columns                          [no compute]
  └─ ShardPlan(epoch): partition indices for this (rank, DataLoader worker)
       └─ sub-catalog = catalog.partitions[shard_indices]
            └─ lsdb CatalogStream(sub, client, partitions_per_chunk, shuffle, seed=derived)
                 └─ NestedFrame chunks (ragged sizes)            ← producer (Dask)
  ─────────────────────────────────────────── seam: PartitionSource.iter_chunks(epoch) ───
  └─ iter_slices: whole chunk, or consecutive <= batch_size slices within it (never across chunks)
       └─ resolved.collate(frame_slice) -> dict[str, Tensor]      ← consumer (torch)
            └─ DataLoader(batch_size=None, num_workers=k)
                 └─ TrainSession
```

Batching happens on frame rows within one chunk, and each batch is collated in one
vectorized call. Keeping batches inside a chunk means no frame concatenation on the torch
side and keeps the unit of work aligned with Dask's. The
dataset yields batches that are already collated, so the DataLoader runs with
`batch_size=None` and there is no per-sample dict plus `collate_fn` step, which would be
slow for nested data.

### 5.2 The producer/consumer seam

```python
class PartitionSource(Protocol):
    def iter_chunks(self, shard: ShardInfo, epoch: int) -> Iterator[npd.NestedFrame]: ...
```

`CatalogStreamSource` is the default implementation. It reuses lsdb's `CatalogStream` on
a per-shard sub-catalog, which keeps its prefetching, its client handling, and its
partition and row shuffling. Because order is decided before `partitions[...]` re-sorts
(finding 7), the shard *assignment* comes from our global permutation, and the order
*within* a shard comes from `CatalogStream`'s seeded shuffle.

The protocol lets a later source with deeper prefetch, or one that is not a catalog (for
example the Kafka source from #1053), plug in without changing batching, collation or the
session.

### 5.3 Sharding across DDP ranks and DataLoader workers

- **Unit of sharding: the partition.** Rows are never split across shards, so no shard
  computes a partition only to throw most of it away.
- **Shard id.** `shard_id = rank * num_workers + worker_id`, and
  `num_shards = world_size * num_workers`.
- **Assignment for an epoch.**
  `perm = default_rng(SeedSequence([seed, epoch])).permutation(npartitions)`; this shard
  takes `perm[shard_id::num_shards]`. Every shard computes the same permutation, so the
  shards are disjoint and together cover every partition. Within a shard,
  `CatalogStream` gets `seed = SeedSequence([seed, epoch, shard_id])`.
- **`shuffle=False`.** Assignment uses `arange` instead of a permutation, and order is
  deterministic. Within a shard the order is `CatalogStream`'s reversed pixel order
  (finding 2), which the docs will state.
- **Where rank and world size come from.** They are captured in the dataset's
  `__init__`, from `torch.distributed` if it is initialized, or from explicit
  `rank=`/`world_size=` arguments. They cannot be read in `__iter__`, because
  `torch.distributed` is not initialized inside DataLoader worker processes. The worker
  id is read in `__iter__` through `get_worker_info()`.
- **Seed handling.** With `seed=None`, single-process runs draw a base seed once in
  `__init__`. DataLoader workers then inherit the same seed when the dataset is pickled,
  so they agree on the permutation. Under DDP with `shuffle=True` and `seed=None`, the
  constructor raises a clear `ValueError`, because each rank would draw a different seed
  and the shards would overlap.
- **More shards than partitions.** If `num_shards > npartitions`, some shards are empty,
  and the dataset logs a warning that suggests repartitioning or fewer workers.
- **Dask inside DataLoader workers.** A `Client` object cannot usefully cross a process
  boundary. With `num_workers=0`, the source uses the active `distributed` client, if
  there is one, for asynchronous prefetch. With `num_workers>0`, each worker process
  computes with the synchronous scheduler by default, and DataLoader's `prefetch_factor`
  supplies the overlap. A worker can also connect its own client when given
  `scheduler_address=`. The dataset stores the address, never a `Client`.
- **Rows split unevenly across DDP ranks.** Partitions vary in size, so ranks run
  different numbers of steps, and a plain DDP all-reduce would hang. For this round,
  `TrainSession` runs its loop inside DDP's `model.join()` context when the model is
  `DistributedDataParallel`; this is torch's supported mechanism for uneven inputs. A
  stricter "equalize" mode is listed in §10.

### 5.4 Epochs and streaming

```python
ds = CatalogIterableDataset(
    catalog, spec,
    batch_size=256,           # max rows per batch; None = one batch per chunk
    shuffle=True, seed=0,
    epochs=1,                 # int, or None = stream forever (reseeded every pass)
    partitions_per_chunk=4,
)
ds.set_epoch(3)               # optional; like DistributedSampler.set_epoch
loader = DataLoader(ds, batch_size=None, num_workers=2)
```

- **One iteration runs `epochs` passes.** Pass `e` uses the epoch-`e` permutation, so
  every pass is reshuffled reproducibly. `epochs=None` loops forever and is the
  open-ended streaming mode.
- **`InfiniteStream` is not used.** Its draws with replacement across chunks (finding 6)
  cannot be sharded without duplication. Cycling through reshuffled epochs gives the same
  "never ends" behavior with exact coverage guarantees.
- **`set_epoch` and persistent workers.** `set_epoch` reaches the workers at the next
  `iter()` because workers re-pickle the dataset. With `persistent_workers=True` it does
  not propagate, which is torch's own limitation for iterable datasets; the docstring will
  say so.

### 5.5 Buffering, backpressure and memory

- **Per producer, memory is about 2 chunks.** That is the chunk being consumed and one
  prefetched chunk (finding 1). There is no carry-over buffer.
- **The DataLoader adds a bounded amount.** It holds up to `prefetch_factor * num_workers`
  collated batches.
- **No queue is unbounded.** Backpressure comes naturally: a producer only submits its
  next chunk when the consumer asks for the current one.
- **`partitions_per_chunk` trades memory for throughput and mixing.** Bigger chunks mean
  more rows in memory, more Dask parallelism per fetch, and better row mixing.
- **Without a client, `num_workers=0` is slow.** It gets no compute/consume overlap
  (finding 1), so the docs will recommend `Client(processes=False)` or `num_workers>0`.

### 5.6 Shuffle quality caveat

HEALPix partitions are spatially coherent, so rows that share a chunk are sky neighbors,
and batches drawn from one chunk are not i.i.d. The available mitigations are
`partitions_per_chunk>1`, together with `CatalogStream`'s within-chunk row shuffle, and
partition-order shuffling. Per the review, no cross-chunk shuffle buffer is added here;
better mixing belongs upstream in `CatalogStream` (§14). The docs will state the caveat.

## 6. Training session

### 6.1 The step seam: a user callable, with a default for the common case

```python
class TrainStep(Protocol):
    def __call__(self, model: nn.Module, batch: dict[str, Tensor]) -> Mapping[str, float | Tensor]: ...
    # optional: state_dict() / load_state_dict() so checkpoints include optimizer state

step = SupervisedStep(
    loss_fn=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    inputs="x",               # batch key, list of keys (positional args), or callable(batch) -> args
    target="y",
    scheduler=None,           # optional LR scheduler, stepped per batch
)
```

**Proposal: the session calls a `train_step(model, batch) -> metrics` callable.**
`SupervisedStep` covers the module + loss + optimizer case.

- **It matches the shape Hyrax already uses.** In #1053 the model's `train_batch` owns
  zero_grad → forward → loss → backward → step. Keeping that responsibility in one
  callable means a Hyrax backend is just another `TrainStep` that calls
  `model.train_batch(...)` (§8).
- **Hard cases stay possible.** Multi-loss models, AMP, gradient clipping, GANs and custom
  optimizer schedules need no session changes.
- **The easy case stays easy.** `SupervisedStep` handles single-input, single-target
  supervised learning in one line.

The alternative was a session that takes `(model, loss_fn, optimizer)` and owns the step.
That bakes in one training pattern and would need a parallel code path for Hyrax.

### 6.2 Session

```python
from lsdb_ml import TrainSession

with TrainSession(
    model, step, data=loader,             # data is optional: manual driving via session.train_batch(b)
    device="cpu",                         # or "auto"
    checkpoint_dir="ckpt/", checkpoint_every=500,
    min_batch_size=None,                  # e.g. 2 for BatchNorm models
    callbacks=[log_fn],                   # log_fn(session, batch_idx, metrics): the MLflow/TB hook point
) as session:
    for batch, metrics in session:        # metrics is None for a skipped batch
        if metrics and metrics["loss"] < 1e-3:
            session.stop()
trained = session.close()                 # idempotent; final checkpoint already written
```

Behavior carried over from #1053's spec:

- **Iteration.** `__iter__` iterates `data` and yields `(batch, process(batch))`.
  Iterating without a data source raises `RuntimeError`.
- **`process` / `train_batch`.**
  - Calls after `close()` raise.
  - Zero-row batches are skipped and return `None`, as are batches smaller than
    `min_batch_size` when it is set.
  - The batch moves to the device and the model goes into `train()` mode.
  - Then the session calls the step, increments `batch_count`, and runs the callbacks.
  - A checkpoint is written every `checkpoint_every` batches.
- **`stop()`.** It sets a flag that `__iter__` checks between batches, then closes the
  data iterator, which shuts down DataLoader workers and drops pending Dask futures.
- **`close()`.** It calls `stop()`, writes the final checkpoint, sets `_closed`, and
  returns the model (unwrapped from DDP if needed). It is idempotent, and `__exit__`
  calls it.
- **Per-batch weighting, documented.** Every optimizer step has equal weight no matter
  the batch size, so a 3-row tail slice of a chunk steps as hard as a full one. Use
  `min_batch_size` to skip small slices.

**Checkpoints** are written atomically (temporary file, then rename) as a `torch.save`
dict containing `model` (state_dict), `step` (from `step.state_dict()` if present),
`batch_count`, `epoch` and `lsdb_ml_version`. With `keep_last=N` (default: keep all) old
files are pruned. Only rank 0 writes under DDP. `TrainSession(..., resume_from=path)`
restores model, step and counters but **not** the data position (§10).

Not carried over: Ignite, MLflow and TensorBoard wiring, Hyrax config tables. Callbacks
take their place.

## 7. Inference path

### 7.1 API

```python
from lsdb_ml import ModelSpec, OutputSpec, predict

pred = predict(
    catalog,
    model=ModelSpec(factory=make_model, weights="model.pt", device="cpu"),  # or a bare nn.Module
    spec=spec,                                  # ColumnSpec; resolved against catalog.meta immediately
    inputs="x",                                 # how to call the model, as in SupervisedStep
    outputs=OutputSpec({"z_phot": "float32", "z_err": "float32"}),
    batch_size=4096,
    append=True,                                # keep input columns (False: prediction columns only)
)
pred                     # lazy lsdb.Catalog; pred.meta has the new columns with correct dtypes
pred.compute()
```

For explicit lifecycle control, deploy the model once and predict on several catalogs:

```python
with ModelDeployment(model_spec, client=client) as deployed:
    a = deployed.predict(cat_a, spec=..., outputs=...).compute()
    b = deployed.predict(cat_b, spec=..., outputs=...).compute()
# plugin unregistered from all workers here
```

### 7.2 Model serialization and the WorkerPlugin lifecycle

- **`ModelSpec(factory, weights=None, device="cpu", torch_threads=None)`.** The preferred
  form. Only the factory (a picklable callable) and a weights path or URL travel to the
  scheduler. Each worker builds the model and loads the weights itself
  (`map_location=device`), so large weights never pass through the scheduler.
- **A bare `nn.Module`.** It is converted to a `ModelSpec` whose factory holds a CPU deep
  copy. That is convenient, but the whole module is cloudpickled into the plugin once, and
  the docs will recommend the factory form for large models.
- **`ModelPlugin(WorkerPlugin)`.** Its name is `lsdb-ml-model-<token>`, where the token is
  a hash of the spec, so concurrent deployments never collide and redeploying the same
  spec is idempotent.
  - `setup(worker)`: build the model, load weights, `.eval()`, move to the resolved
    device, optionally `torch.set_num_threads`, and record a load counter.
  - `teardown`: drop references and free the CUDA cache.
- **Registration.** `client.register_plugin(plugin)` installs the plugin on all current
  workers and on any worker that joins later.
- **`predict()` leaves the plugin registered**, because the returned catalog is lazy and
  may be computed later. `ModelDeployment.close()` unregisters it.
- **No `distributed` client.** Under the synchronous or threaded scheduler, the model is
  built once in the local process and looked up from a process-local registry under the
  same name. The multiprocessing scheduler is rejected with a clear error.
- **A worker without the plugin.** If a task runs on a worker that lacks it, for example
  because the catalog was computed on a different client, the task raises a `RuntimeError`
  that names the plugin and explains how to fix it.

### 7.3 Device placement

`device="cpu"` is the default. `"auto"` picks `cuda` if available, then `mps`, then
`cpu`. `"cuda"`, `"cuda:N"` and `torch.device` are accepted. The device is resolved *on
the worker* in `setup()`, not on the client. For multi-GPU nodes the recommended
pattern is one worker per GPU with `CUDA_VISIBLE_DEVICES` set (e.g. `dask-cuda`), so each
worker's `"cuda"` is its own GPU. GPU workers should run with `threads_per_worker=1`.
On CPU, several threads per worker can share the read-only eval model safely under
`torch.inference_mode()`. `torch_threads=` avoids oversubscribing cores (Dask threads ×
torch intra-op threads).

### 7.4 Per-partition task and alignment guarantees

`predict` calls `catalog.map_partitions(_predict_partition, plugin_name=..., ..., meta=out_meta)`.
Each task:

1. Gets its model with `get_worker().plugins[plugin_name].model`. All other state is
   local to the task; there is no shared mutable dataset. This removes #1062's race.
2. Returns an empty frame built from `out_meta` if the partition is empty.
3. Walks the partition in order, `iloc[i : i + batch_size]`, with no shuffle and no
   reordering. For each slice it calls `resolved.collate`, moves the result to the device,
   runs the model under `torch.inference_mode()`, moves the output to CPU numpy, and
   appends it to a local list.
4. Concatenates the outputs and **asserts** that the total length equals
   `len(partition)`, so a model that drops or adds rows fails loudly. It then checks
   shapes and dtypes against `OutputSpec` and casts.
5. Returns `partition.assign(**columns)` when `append=True`, or a frame on
   `partition.index` with only the prediction columns when `append=False`.

Alignment holds by construction: outputs attach positionally to the same frame object
the inputs came from, inside one task. They are not joined on `_healpix_29`, which is not
guaranteed unique (finding 8). Rows are never dropped. Null policies either raise or fill,
so row count is preserved.

### 7.5 Output meta

`OutputSpec` maps model outputs to columns:

- A tensor of shape `(B,)` maps to one column.
- A tensor of shape `(B, k)` needs `k` column names, or `as_list=True` for a single
  fixed-size list column.
- A `dict[str, Tensor]` maps by key.

The output meta is `catalog.meta.assign(<empty typed Series per output>)`, or just the new
columns on the same index when `append=False`. It is built on the client without
computing anything. Each task validates its output against the same spec, so `meta`
cannot drift from what is actually computed. Inferring `OutputSpec` automatically by
running the model on a dummy batch is left for later (§10).

## 8. Backend seam: PyTorch now, Hyrax later

Training and inference depend only on three small protocols:

| Protocol | Torch implementation (this round) | Future Hyrax implementation |
|---|---|---|
| `TrainStep(model, batch) -> metrics` | `SupervisedStep`, or any user callable | wraps `model.train_batch(prepare_inputs(adapt(batch)))` |
| `PredictFn(model, batch) -> Tensor \| dict` | built from `inputs=` (keys or callable) | wraps `model.infer_batch(...)` |
| `ModelSpec.factory() -> nn.Module` (+ weights) | user factory + `torch.load` | Hyrax model registry + `load_model_weights` |

`backends/base.py` defines a `Backend` protocol that bundles the three (`make_model`,
`make_train_step`, `make_predict_fn`, `save`, `load`). `TorchBackend` is the default.
`TrainSession` and `predict` accept `backend=`.

A Hyrax backend would live in `lsdb_ml/backends/hyrax_backend.py`, import `hyrax` lazily,
and be enabled through the existing `hyrax` extra. Its main job is *batch adaptation*:
turning our `dict[str, Tensor]` into Hyrax's `{"data": {...}, "object_id": ...}` structure.
Core modules never import it.

## 9. Testing plan

All tests run on CPU and stay fast. They use tiny models (one `nn.Linear`), synthetic
catalogs built with `lsdb.from_dataframe` with a nested column and small
`partition_rows`, and `Client(processes=False)` / `LocalCluster(processes=False)`. The
fixtures live in `tests/conftest.py`: a session-scoped nested catalog (including
zero-length and null nested rows) and a distributed client fixture.

| Requirement | Test |
|---|---|
| Spec validation against meta | Missing column; flat vs nested mismatch; missing sub-column; string without `transform`; duplicate keys; all errors reported together; **no compute** (assert via a catalog whose partitions raise if computed). |
| Collation, flat and nested | Shapes, dtypes and values against a per-row reference implementation; mask and lengths; `max_length` truncation (head and tail); `pad_to`; null nested rows give length 0; writable tensors. |
| Ragged and empty batches | Slices never cross chunks and cover each chunk exactly; `batch_size=None` gives one batch per chunk; empty partitions and chunks; session skips zero-row and sub-`min_batch_size` batches. |
| No duplicated or dropped rows | For `num_workers` in {0, 2} × simulated `world_size` in {1, 2} (explicit `rank=`), the union of `_index` equals the catalog's rows exactly once; same across multiple epochs; per-epoch reshuffle differs but stays complete; same seed reproduces the same order. |
| Session runs and checkpoints | Loss decreases on a linear problem; periodic and final checkpoints exist and reload; `stop()` ends iteration; `close()` is idempotent; `process()` after close raises; `resume_from` restores counters. |
| Inference index alignment | Non-trivial layout: many small partitions, a `.query` that empties some partitions, and `partitions[...]` reordering. Run on a 2-worker × 2-thread cluster with a deterministic linear model (`pred = 2*mag + 1`), then compare row by row and by `id` against an eager torch computation. Includes a test where the model returns the wrong row count and must fail. |
| Plugin loads once per worker | Factory increments a process-global counter (`processes=False`); after predicting over more partitions than workers, counter == number of workers; redeploying the same spec does not reload. |
| Meta correctness | `pred.meta.dtypes` equals the computed dtypes; `append=False` layout. |

An optional `@pytest.mark.slow` test runs real two-process DDP with the `gloo` backend.

## 10. Open questions and risks

1. **Default device.** I proposed `"cpu"` for both training and inference, with `"auto"`
   opt-in. Should training default to `"auto"` instead?
2. **Projecting nested sub-columns.** Only top-level columns are projected before
   streaming. Whether lsdb can project nested *sub-columns* on an already-opened or
   in-memory catalog (beyond `open_catalog(columns=["lc.flux"])`) still needs checking.
   It matters a lot for I/O with wide light-curve tables.
3. **Uneven DDP inputs.** `model.join()` is the first-round answer. An "equalize" mode
   could use per-partition row counts from HATS metadata, which exist for unmodified
   catalogs but not after filtering. Is that worth doing?
4. **Shuffle quality.** Partition-local mixing (§5.6) may hurt training on spatially
   structured data. Should a cross-chunk shuffle buffer be in the first implementation?
5. **Upstream lsdb changes would simplify things.** Three would help: let
   `CatalogStream` accept an explicit partition order and a configurable prefetch depth,
   and make unshuffled order forward. With them we could drop the sub-catalog workaround.
   `partitions[[...]]` silently re-sorting is also worth reporting. Should I open lsdb
   issues for these?
6. **Mid-stream resume.** Checkpoints restore model and optimizer state, not the data
   position. Resuming in the middle of an epoch replays that epoch from the start.
7. **Automatic output meta.** Inferring `OutputSpec` from a dummy forward pass is
   convenient, but it fails for models that cannot accept B=0 or that need real values.
   I proposed explicit outputs for v1.
8. **Plugin lifetime.** `predict()` leaves its plugin registered, keyed by spec hash.
   That is acceptable for notebooks but holds model memory until the client closes or
   `ModelDeployment.close()` runs. Alternative: always require `ModelDeployment`.
9. **Where things live.** Per the comment on #1062, a general `NestedPandasDataset` with
   automatic getters belongs in Hyrax. lsdb-ml keeps its own lightweight spec/collate
   layer and does not depend on it. The two could converge later through the Hyrax
   backend's batch adapter.
10. **CI cost.** On Linux, `pip install torch` pulls CUDA wheels of several GB. CI should
    install the CPU build, e.g. `uv pip install --torch-backend=cpu`. Python 3.14 support
    in the CI matrix should also be checked against the torch version we pin.

## 11. Reused from #1053 / #1062, and what changed

| Idea | Source | Kept / changed |
|---|---|---|
| Context-manager session: iterate `(batch, metrics)`, `stop()`, `close()` returns the model | #1053 | **Kept**, with the same skip semantics |
| Driving the per-batch loop by hand instead of `trainer.run(max_epochs)` | #1053 | **Kept** |
| Skipping empty batches; optional `min_batch_size`; per-batch weighting documented | #1053 | **Kept** |
| Periodic plus final checkpointing | #1053 | **Kept**; added atomic writes, optimizer state, rank-0 only |
| Model-owned `train_batch` | #1053 | **Changed** to a `TrainStep` callable plus `SupervisedStep`, which keeps the same seam without Hyrax |
| MLflow/TensorBoard inside the session | #1053 | **Changed** to callbacks (out of scope) |
| Kafka source | #1053 | **Changed** to `PartitionSource` over lsdb `CatalogStream`; Kafka could become another source later |
| Model installed once per worker via `WorkerPlugin`; inference as `map_partitions` | #1062 | **Kept** |
| Discovering columns via a hand-built `create_meta_dict` / empty placeholder frame | #1062 | **Changed** to `catalog.meta` plus `NestedDtype.column_dtypes`, resolved before any compute |
| Shared mutable dataset on the plugin, positional write-back into a buffer | #1062 | **Removed**: per-task state, in-order slicing, length assertion, `assign` onto the same frame |
| Hard-coded `"mps"` device; `config` dicts | #1062 | **Changed** to device resolved on the worker; plain Python arguments |
| Pad + mask for nested columns | Hyrax `LSDBStreamDataset` / #1062 | **Kept** the representation; vectorized it and moved it behind a `Field` protocol |
| `num_workers=0` forced for streaming | Hyrax `LSDBStreamDataset` | **Changed**: partition sharding makes `num_workers>0` and DDP correct |

## 12. CI note

Every workflow except the smoke test triggers only on `push`/`pull_request` to `main`:
`testing-and-coverage.yml`, `pre-commit-ci.yml` and `build-documentation.yml`. PRs to
`dev` therefore get **no CI**. Suggested change, not made in this PR: in each of those
three workflows set

```yaml
on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]
```

and, once torch is a dependency, install the CPU build in CI (§10, item 10).

## 13. Implementation plan (after approval)

Each step is a feature branch off `dev` with a PR into `dev`:

1. `feature/spec-collate`: dependencies, `spec.py`, `collate.py`, `device.py`, fixtures
   and tests.
2. `feature/training`: `data/*`, `training/*` and `backends/*` (torch backend), tests,
   and an example script.
3. `feature/inference`: `inference/*`, tests, and an example notebook.

## 14. Upstream `CatalogStream` changes worth filing

Rough shape of lsdb issues that would simplify lsdb-ml. Not filed; recorded for the
maintainer.

1. **Accept an explicit partition list/order.** `CatalogStream(catalog, partitions=[...])`
   would replace the `catalog.partitions[idx]` sub-catalog workaround used for sharding.
2. **`catalog.partitions[[3, 0, 5]]` silently re-sorts to pixel order.** Either preserve
   the requested order or document that it does not.
3. **Unshuffled order is reversed.** `get_next_partitions` pops from the tail, so
   `shuffle=False` streams partitions last-to-first. Forward order would be less surprising.
4. **Configurable prefetch depth.** Prefetch is fixed at one chunk; a `prefetch=` argument
   would help hide fetch latency for slow (remote) partitions.
5. **No overlap without a distributed client.** With `client=None` the next chunk is
   computed inline *before* the current chunk is returned, which doubles latency for the
   first chunk and gives no overlap afterwards. Returning first and computing lazily, or
   using a background thread, would help single-process use.
6. **Built-in sharding.** `shard_id`/`num_shards` arguments (disjoint, covering partition
   assignment from a shared seed) would let any torch consumer shard correctly.
7. **Epoch-aware infinite streaming.** `InfiniteStream` draws chunks with replacement
   across chunks. An epoch-cycling mode (reshuffle per pass, no replacement within a pass)
   is shardable and has exact coverage per pass.
8. **Better mixing.** An optional cross-chunk row shuffle buffer inside the stream, since
   HEALPix chunks are spatially coherent.
9. **`CatalogIterator.__len__`** excludes the in-flight chunk, so it under-counts by one
   after the first `next()`.
10. **`catalog.partitions[[]]` raises** (`ValueError` from `PixelSearch`) instead of
    returning an empty catalog, so callers must special-case empty selections.
11. **Scheduler choice when `client=None`.** The stream then computes with whatever
    global Dask scheduler is configured. In a forked DataLoader worker that can be a
    parent's distributed client, which does not work in the child. lsdb-ml forces the
    synchronous scheduler inside workers; a `scheduler=` argument on `CatalogStream`
    would make that explicit.

## 15. Implementation notes (differences from the sections above)

Recorded while implementing; the PRs carry the code.

- **No separate `backends/` package.** The seam is three small pieces instead: the
  `TrainStep` protocol (training), `ModelSpec.factory` plus weights (model construction),
  and the `inputs=` argument (how a batch becomes model arguments). A Hyrax backend would
  supply a `TrainStep`, a factory and an `inputs` callable. A bundling `Backend` class can
  be added if Hyrax needs one; with only the torch path it would have no behavior.
- **`batching.py` folded into `data/dataset.py`** as `iter_slices`, since batching is now
  a few lines (review decision 2). There is no `drop_last`; use `min_batch_size`.
- **No `id_column` option.** `Flat("object_id", dtype=torch.int64)` does the same thing.
- **No `as_list` output columns yet.** `(B, k)` outputs map to `k` named columns.
- **`PartitionSource.iter_chunks(partitions, shuffle, seed)`.** The source receives the
  partition indices; shard assignment lives in the dataset so every source shares it.
- **Output columns are built with `pd.Series(values, dtype=d).array`** and assigned
  positionally. `pd.array(values, "float32")` produces a nullable `Float32` column,
  which would silently differ from a numpy `float32` meta.

