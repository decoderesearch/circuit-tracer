from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TransformerLens_NNSight_Mapping:
    """Mapping specifying important locations in NNSight models, as well as mapping from TL Hook Points to NNSight locations"""

    model_architecture: str  # HuggingFace model architecture
    attention_location_pattern: str  # Location of the attention patterns
    layernorm_scale_location_patterns: list[str]  # Location of the Layernorm denominators
    pre_logit_location: str  # Location immediately before the logits (the location from which we will attribute for logit tokens)
    embed_location: str  # Location of the embedding Module (the location to which we will attribute for embeddings)
    embed_weight: str  # Location of the embedding weight matrix
    unembed_weight: str  # Location of the unembedding weight matrix
    feature_hook_mapping: dict[
        str, tuple[str, Literal["input", "output"]]
    ]  # Mapping from (TransformerLens Hook) to a tuple representing an NNSight Envoy location, and whether we want its input or output


# Create an instance with the original configuration values
gemma_2_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="Gemma2ForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.norm.source.self__norm_0.source.torch_rsqrt_0",
    ],
    pre_logit_location="model",
    embed_location="model.embed_tokens",
    embed_weight="model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "ln2.hook_normalized": (
            "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
            "output",
        ),
        "hook_resid_mid": ("model.layers[{layer}].pre_feedforward_layernorm", "input"),
        "hook_mlp_out": ("model.layers[{layer}].post_feedforward_layernorm", "output"),
    },
)

# Create an instance with the original configuration values
gemma_3_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="Gemma3ForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].self_attn.q_norm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].self_attn.k_norm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "model.norm.source.self__norm_0.source.torch_rsqrt_0",
    ],
    pre_logit_location="model",
    embed_location="model.embed_tokens",
    embed_weight="model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "ln2.hook_normalized": (
            "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
            "output",
        ),
        "hook_resid_mid": ("model.layers[{layer}].pre_feedforward_layernorm", "input"),
        "mlp.hook_in": ("model.layers[{layer}].pre_feedforward_layernorm", "output"),
        "hook_mlp_out": ("model.layers[{layer}].post_feedforward_layernorm", "output"),
    },
)

gemma_3_conditional_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="Gemma3ForConditionalGeneration",
    attention_location_pattern="language_model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "language_model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.layers[{layer}].self_attn.q_norm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.layers[{layer}].self_attn.k_norm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
        "language_model.norm.source.self__norm_0.source.torch_rsqrt_0",
    ],
    pre_logit_location="language_model",
    embed_location="language_model.embed_tokens",
    embed_weight="language_model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "ln2.hook_normalized": (
            "language_model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
            "output",
        ),
        "hook_resid_mid": ("language_model.layers[{layer}].pre_feedforward_layernorm", "input"),
        "mlp.hook_in": ("language_model.layers[{layer}].pre_feedforward_layernorm", "output"),
        "hook_mlp_out": ("language_model.layers[{layer}].post_feedforward_layernorm", "output"),
    },
)

# Create an instance with the original configuration values
llama_3_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="LlamaForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "model.layers[{layer}].input_layernorm.source.mean_0",
        "model.layers[{layer}].post_attention_layernorm.source.mean_0",
        "model.norm.source.mean_0",
    ],
    pre_logit_location="model",
    embed_location="model.embed_tokens",
    embed_weight="model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
        "hook_mlp_out": ("model.layers[{layer}].mlp", "output"),
        "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
        "mlp.hook_out": ("model.layers[{layer}].mlp", "output"),
    },
)

# Create an instance with the original configuration values
qwen_3_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="Qwen3ForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "model.layers[{layer}].input_layernorm.source.mean_0",
        "model.layers[{layer}].post_attention_layernorm.source.mean_0",
        "model.norm.source.mean_0",
    ],
    pre_logit_location="model",
    embed_location="model.embed_tokens",
    embed_weight="model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
        "mlp.hook_out": ("model.layers[{layer}].mlp", "output"),
    },
)


gpt_oss_mapping = TransformerLens_NNSight_Mapping(
    model_architecture="GptOssForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
    layernorm_scale_location_patterns=[
        "model.layers[{layer}].input_layernorm.source.mean_0",
        "model.layers[{layer}].post_attention_layernorm.source.mean_0",
        "model.norm.source.mean_0",
    ],
    pre_logit_location="model",
    embed_location="model.embed_tokens",
    embed_weight="model.embed_tokens.weight",
    unembed_weight="lm_head.weight",
    feature_hook_mapping={
        "hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
        "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
        "mlp.hook_out": ("model.layers[{layer}].mlp.source.self_experts_0", "output"),
        "hook_mlp_out": ("model.layers[{layer}].mlp.source.self_experts_0", "output"),
    },
)


def get_mapping(model_architecture: str) -> TransformerLens_NNSight_Mapping:
    """Get the TransformerLens-NNSight mapping for a given model architecture.

    Args:
        model_architecture: The model architecture name (e.g., 'Gemma2ForCausalLM', 'Llama2ForCausalLM')

    Returns:
        TransformerLens_NNSight_Mapping: The mapping configuration for the specified architecture

    Raises:
        ValueError: If the model architecture is not supported
    """
    mappings = {
        mapping.model_architecture: mapping
        for mapping in [
            gemma_2_mapping,
            gemma_3_mapping,
            gemma_3_conditional_mapping,
            llama_3_mapping,
            qwen_3_mapping,
            gpt_oss_mapping,
        ]
    }

    if model_architecture not in mappings:
        supported_architectures = list(mappings.keys())
        raise ValueError(
            f"Unsupported model architecture: {model_architecture}. "
            f"Supported architectures: {supported_architectures}"
        )

    return mappings[model_architecture]


@dataclass
class UnifiedConfig:
    """A unified config class that supports both TransformerLens and NNsight field names."""

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int

    tokenizer_name: str
    model_name: str
    original_architecture: str

    n_key_value_heads: int | None = None
    dtype: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UnifiedConfig":
        """Create from dictionary."""
        return cls(
            n_layers=config_dict["n_layers"],
            d_model=config_dict["d_model"],
            d_head=config_dict["d_head"],
            n_heads=config_dict["n_heads"],
            d_mlp=config_dict["d_mlp"],
            d_vocab=config_dict["d_vocab"],
            tokenizer_name=config_dict["tokenizer_name"],
            model_name=config_dict["model_name"],
            original_architecture=config_dict["original_architecture"],
            n_key_value_heads=config_dict.get("n_key_value_heads"),
            dtype=config_dict.get("dtype"),
        )


def convert_nnsight_config_to_transformerlens(config):
    """Convert NNsight config to TransformerLens config format.

    Args:
        config: NNsight config object
        return_unified: If True, return UnifiedConfig instead of HookedTransformerConfig
    """
    # If already a UnifiedConfig, return as-is
    if isinstance(config, UnifiedConfig):
        return config

    field_mappings = {
        # Basic model dimensions
        "num_hidden_layers": "n_layers",
        "hidden_size": "d_model",
        "head_dim": "d_head",
        "num_attention_heads": "n_heads",
        "intermediate_size": "d_mlp",
        "vocab_size": "d_vocab",
        # Attention parameters
        "num_key_value_heads": "n_key_value_heads",
        # Model metadata
        "torch_dtype": "dtype",
    }
    config_dict = config.to_dict()

    if "original_architecture" not in config_dict:
        config_dict["original_architecture"] = config.architectures[0]
    if "tokenizer_name" not in config_dict:
        config_dict["tokenizer_name"] = config.name_or_path
    if "model_name" not in config_dict:
        config_dict["model_name"] = config.name_or_path

    if "text_config" in config_dict:
        config_dict |= config_dict["text_config"]

    for nnsight_field, transformerlens_field in field_mappings.items():
        if transformerlens_field not in config_dict:
            config_dict[transformerlens_field] = config_dict[nnsight_field]

    return UnifiedConfig.from_dict(config_dict)
