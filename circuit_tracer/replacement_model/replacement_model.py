"""
Unified ReplacementModel interface that supports both nnsight and transformerlens backends.
"""

from typing import TYPE_CHECKING, Literal, overload

import torch

from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

if TYPE_CHECKING:
    # Type-only. The real imports stay inside the factory bodies, because each backend pulls in a
    # heavy framework (TransformerLens, nnsight, interp-engine) and asking for one must not import
    # the other two.
    from .replacement_model_interp_engine import InterpEngineReplacementModel
    from .replacement_model_nnsight import NNSightReplacementModel
    from .replacement_model_transformerlens import TransformerLensReplacementModel

    AnyReplacementModel = (
        TransformerLensReplacementModel | NNSightReplacementModel | InterpEngineReplacementModel
    )

Backend = Literal["nnsight", "transformerlens", "interp_engine"]

_BACKENDS = ", ".join(f"'{b}'" for b in ("nnsight", "transformerlens", "interp_engine"))


class ReplacementModel:
    """
    Unified ReplacementModel interface that supports both nnsight and transformerlens backends.

    This class acts as a factory that creates the appropriate backend-specific ReplacementModel
    based on the backend parameter.

    Each factory is overloaded on ``backend`` so that a literal argument -- including the default --
    yields the concrete backend class rather than a union of all three. Without this, reading a
    backend-specific attribute off the result (``model.cfg``, which only TransformerLens has) is a
    type error at every call site, since one member of the union is a plain ``nn.Module`` whose
    ``__getattr__`` returns ``Tensor | Module``.

    The embedding matrices are deliberately *not* backend-specific: every backend exposes
    ``embed_weight`` and ``unembed_weight`` as ``(d_vocab, d_model)``, so code that only needs those
    can stay generic. See ``tests/test_embedding_accessors.py``.
    """

    @classmethod
    @overload
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        backend: Literal["transformerlens"] = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        lazy_encoder: bool = ...,
        lazy_decoder: bool = ...,
        **kwargs,
    ) -> "TransformerLensReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        backend: Literal["nnsight"],
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        lazy_encoder: bool = ...,
        lazy_decoder: bool = ...,
        **kwargs,
    ) -> "NNSightReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        backend: Literal["interp_engine"],
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        lazy_encoder: bool = ...,
        lazy_decoder: bool = ...,
        **kwargs,
    ) -> "InterpEngineReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        backend: Backend = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        lazy_encoder: bool = ...,
        lazy_decoder: bool = ...,
        **kwargs,
    ) -> "AnyReplacementModel": ...

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
    ) -> "AnyReplacementModel":
        """Create a ReplacementModel from model name and transcoder config.

        Args:
            model_name (str): The name of the pretrained transformer model
            transcoder_set (str): Either a predefined transcoder set name, or a config file
            backend (Backend): Which backend to use - "transformerlens", "nnsight" or
                "interp_engine"
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
    @overload
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["transformerlens"] = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        **kwargs,
    ) -> "TransformerLensReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["nnsight"],
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        **kwargs,
    ) -> "NNSightReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["interp_engine"],
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        **kwargs,
    ) -> "InterpEngineReplacementModel": ...

    @classmethod
    @overload
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype = ...,
        **kwargs,
    ) -> "AnyReplacementModel": ...

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = "transformerlens",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "AnyReplacementModel":
        """Create a ReplacementModel from model name and transcoder objects.

        Args:
            model_name (str): The name of the pretrained transformer model
            transcoders (Union[TranscoderSet, CrossLayerTranscoder]): The transcoder set or
                cross-layer transcoder
            backend (Backend): Which backend to use - "transformerlens", "nnsight" or
                "interp_engine"
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
        elif backend == "interp_engine":
            from .replacement_model_interp_engine import InterpEngineReplacementModel

            return InterpEngineReplacementModel.from_pretrained_and_transcoders(
                model_name=model_name,
                transcoders=transcoders,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be one of {_BACKENDS}")

    @classmethod
    @overload
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["transformerlens"] = ...,
        **kwargs,
    ) -> "TransformerLensReplacementModel": ...

    @classmethod
    @overload
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["nnsight"],
        **kwargs,
    ) -> "NNSightReplacementModel": ...

    @classmethod
    @overload
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Literal["interp_engine"],
        **kwargs,
    ) -> "InterpEngineReplacementModel": ...

    @classmethod
    @overload
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = ...,
        **kwargs,
    ) -> "AnyReplacementModel": ...

    @classmethod
    def from_config(
        cls,
        config,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        backend: Backend = "transformerlens",
        **kwargs,
    ) -> "AnyReplacementModel":
        """Create a ReplacementModel from a config and transcoder objects.

        Args:
            config: Model configuration (AutoConfig for nnsight, HookedTransformerConfig
                for transformerlens)
            transcoders (Union[TranscoderSet, CrossLayerTranscoder]): The transcoder set or
                cross-layer transcoder
            backend (Backend): Which backend to use - "transformerlens", "nnsight" or
                "interp_engine"
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
        elif backend == "interp_engine":
            from .replacement_model_interp_engine import InterpEngineReplacementModel

            return InterpEngineReplacementModel.from_config(
                config=config,
                transcoders=transcoders,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be one of {_BACKENDS}")
