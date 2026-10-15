# Long-context memory estimator

## Previous repository behavior

Before this change, `circuit-tracer` could run attribution on long prompts, but it did not provide
a cheap preflight estimate for the dense attribution-graph tensors created during graph construction
and pruning. Users had to infer memory risk from model size, token count, and trial-and-error runs.
That made long-context experiments especially expensive because the graph representation includes:

- selected transcoder feature nodes,
- one reconstruction-error node for every `(layer, token)` pair,
- one token node per prompt token, and
- one or more output-logit nodes.

The dense adjacency matrix scales as `total_nodes ** 2`, so long prompts can become infeasible even
on large GPUs before model weights, transcoders, allocator overhead, and frontend JSON export are
considered.

Baseline inspected for this work:

- upstream repository: `decoderesearch/circuit-tracer`
- local upstream commit: `eb0e0f9` (`Fix error node_id collision and misparse (issue #78) (#100)`)
- existing CLI subcommands: `attribute`, `start-server`

## What changed

This branch adds a lightweight estimator that can be used without loading a model or downloading
transcoders:

- `circuit_tracer.utils.memory_estimation.estimate_graph_memory(...)`
- `circuit_tracer.utils.estimate_graph_memory(...)`
- `circuit-tracer estimate-memory ...`

The estimator reports:

- feature, error, token, logit, and total node counts,
- dense adjacency tensor size,
- dense boolean mask size,
- approximate graph metadata tensor size,
- conservative graph-pruning peak memory,
- optional fit check against a user-provided memory budget, and
- concrete recommendations for reducing OOM risk.

The estimate intentionally focuses on graph memory, not full execution memory. It does not include
model weights, transcoder weights, CUDA allocator fragmentation, tokenizer/model caches, notebook
overhead, or browser/frontend memory.

## Example

```bash
circuit-tracer estimate-memory \
  --tokens 6000 \
  --layers 26 \
  --max_feature_nodes 7500 \
  --n_logits 10 \
  --dtype float16 \
  --available_memory_gib 80
```

For this long-context configuration, the estimator shows that reconstruction-error nodes dominate
the node count and that dense pruning memory can exceed an 80 GiB device budget after a safety
margin. The practical result is that users can identify infeasible traces before spending time
loading a model, transcoders, and activations.

## Result of the change

This is a quality and usability improvement, not a new attribution algorithm. It gives users an
early answer to:

- "Will this prompt length likely fit?"
- "How much does dtype matter?"
- "How much do feature-node caps matter?"
- "Is this failure likely an inherent dense-graph scaling issue?"

It also creates a clear foundation for future sparse or blockwise graph work: the estimator makes
the dense scaling visible and quantifies when a sparse backend becomes necessary.

## Demo notebook

See `demos/long_context_memory_estimator_demo.ipynb` for an executable walkthrough that compares
short and long prompt configurations and shows the CLI-equivalent Python API.
