This README is here to explain `tl_nnsight_mapping.py` and how to create mappings for new models. In particular, this mapping file implements the following dataclass, which tells `circuit-tracer` where the relevant parts of a model are in NNSight model.

```python
@dataclass
class TransformerLens_NNSight_Mapping:
    """Mapping specifying important locations in NNSight models, as well as mapping from TL Hook Points to NNSight locations"""

    model_architecture: str  # HuggingFace model architecture
    attention_location_pattern: str  # Location of the attention patterns
    layernorm_scale_location_patterns: list[str]  # Locations of the Layernorm denominators
    pre_logit_location: str  # Location immediately before the logits (the location from which we will attribute for logit tokens)
    embed_location: str  # Location of the embedding Module (the location to which we will attribute for embeddings)
    embed_weight: str  # Location of the embedding weight matrix
    unembed_weight: str  # Location of the unembedding weight matrix
    feature_hook_mapping: dict[str, tuple[str, Literal['input', 'output']]]  # Mapping from (TransformerLens Hook) to a tuple representing an NNSight Envoy location, and whether we want its input or output
```

The values of the location / pattern fields are the NNSight locations whose outputs we need to access or freeze. For example, if `embed_location="model.embed_tokens"`, this means that `model.model.embed_tokens` is the location of the embeddings, and `model.model.embed_tokens.output` will fetch its output.

Some of these locations can be found easily by looking at the HuggingFace model implementation. Some, however, are less easy to find. Take attention pattern location, which indicates the tensor that should be accessed to freeze the model's attention pattern; Gemma 2's is `"model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0"`. This location corresponds not just to a nn.Module found in the HuggingFace model, but a specific line in the source code thereof. It's often easiest to find the names of these locations by loading your model into an NNSight `LanguageModel`, and exploring it manually, by delving into the model and its submodules, printing them as you go. If you need to access the source code for a module's forward pass, simply use `.source`; note that accessing the `.source` of a `.source` object, is only possible within a `with model.tracer(input_string):` context. For more on `.source`, see [this page](https://nnsight.net/notebooks/NNsight_0_5_Changes/#Accessing-Intermediate-Values).

Finally, note that the last field, `feature_hook_mapping`, maps from TransformerLens hooks to NNSight locations. In some cases, these are most easily specified via the input of a module rather than its output, so this mapping asks the user to specify which is needed - note that for all other fields, the output of the given point is used. This mapping is necessary because the input / output locations of transcoders are currently specified as TransformerLens hooks in their config files, but this may change. 

Want to know if you got all of the freeze-related hook points (i.e. `attention_location_pattern` and `layernorm_scale_location_patterns`) right? Try running your model through `tests/test_freeze_points_hessian.py`, but note that test is still under construction, and could be made more robust.