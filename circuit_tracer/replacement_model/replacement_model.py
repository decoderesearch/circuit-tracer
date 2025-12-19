"""
Unified ReplacementModel interface that supports both nnsight and transformerlens backends.
"""

from typing import Literal
import torch

from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

Backend = Literal["nnsight", "transformerlens"]


class ReplacementModel:
    """
    Unified ReplacementModel interface that supports both nnsight and transformerlens backends.

    This class acts as a factory that creates the appropriate backend-specific ReplacementModel
    based on the backend parameter.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        backend: Backend = "transformerlens",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        lazy_encoder: bool = False,
        lazy_decoder: bool = True,
        **kwargs,
    ):
        """Create a ReplacementModel from model name and transcoder config.

        Args:
            model_name (str): The name of the pretrained transformer model
            transcoder_set (str): Either a predefined transcoder set name, or a config file
            backend (Backend): Which backend to use - "nnsight" or "transformerlens"
            device (Optional[torch.device]): Device to load the model on
            dtype (Optional[torch.dtype]): Data type for the model
            lazy_encoder (bool): Whether to lazily load encoder weights (default: False)
            lazy_decoder (bool): Whether to lazily load decoder weights (default: True)
            **kwargs: Additional arguments passed to the backend-specific implementation

        Returns:
            ReplacementModel: The loaded ReplacementModel using the specified backend
        """
        if device is None:
            device = get_default_device()

        transcoders, _ = load_transcoder_from_hub(  # type:ignore
            transcoder_set,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )

        return cls.from_pretrained_and_transcoders(
            model_name=model_name,
            transcoders=transcoders,
            backend=backend,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = "transformerlens",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """Create a ReplacementModel from model name and transcoder objects.

        Args:
            model_name (str): The name of the pretrained transformer model
            transcoders (Union[TranscoderSet, CrossLayerTranscoder]): The transcoder set or
                cross-layer transcoder
            backend (Backend): Which backend to use - "nnsight" or "transformerlens"
            device (Optional[torch.device]): Device to load the model on
            dtype (Optional[torch.dtype]): Data type for the model
            **kwargs: Additional arguments passed to the backend-specific implementation

        Returns:
            ReplacementModel: The loaded ReplacementModel using the specified backend
        """
        if device is None:
            device = get_default_device()

        if backend == "nnsight":
            # Import backend-specific implementations
            from .replacement_model_nnsight import NNSightReplacementModel

            return NNSightReplacementModel.from_pretrained_and_transcoders(
                model_name=model_name,
                transcoders=transcoders,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        elif backend == "transformerlens":
            from .replacement_model_transformerlens import (
                TransformerLensReplacementModel,
            )

            return TransformerLensReplacementModel.from_pretrained_and_transcoders(
                model_name=model_name,
                transcoders=transcoders,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'nnsight' or 'transformerlens'")

    @classmethod
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = "transformerlens",
        **kwargs,
    ):
        """Create a ReplacementModel from a config and transcoder objects.

        Args:
            config: Model configuration (AutoConfig for nnsight, HookedTransformerConfig
                for transformerlens)
            transcoders (Union[TranscoderSet, CrossLayerTranscoder]): The transcoder set or
                cross-layer transcoder
            backend (Backend): Which backend to use - "nnsight" or "transformerlens"
            **kwargs: Additional arguments passed to the backend-specific implementation

        Returns:
            ReplacementModel: The loaded ReplacementModel using the specified backend
        """
        if backend == "nnsight":
            from .replacement_model_nnsight import NNSightReplacementModel

            return NNSightReplacementModel.from_config(
                config=config,
                transcoders=transcoders,
                **kwargs,
            )
        elif backend == "transformerlens":
            from .replacement_model_transformerlens import (
                TransformerLensReplacementModel,
            )

            return TransformerLensReplacementModel.from_config(
                config=config,
                transcoders=transcoders,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'nnsight' or 'transformerlens'")
