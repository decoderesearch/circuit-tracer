import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils.create_graph_files import create_graph_files


def create_fake_clt_model(cfg: GPT2Config):
    """Create a CLT and ReplacementModel with random weights for testing."""
    # Create CLT with 4x expansion
    clt = CrossLayerTranscoder(
        n_layers=cfg.n_layer,
        d_transcoder=cfg.n_embd * 4,
        d_model=cfg.n_embd,
        dtype=torch.float32,
        lazy_decoder=False,
    )

    # Initialize CLT weights
    with torch.no_grad():
        for param in clt.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    # Create model
    hf_model = GPT2LMHeadModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = ReplacementModel.from_hf_model(hf_model, tokenizer, clt)

    # Monkey patch all_special_ids if necessary
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    # Initialize model weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.1, b=0.1)

    return model


def test_create_graph_files(tmp_path):
    """Test that create_graph_files creates output files correctly."""
    # Create minimal config
    cfg = GPT2Config(
        n_layer=2,
        n_embd=8,
        n_ctx=16,
        n_head=2,
        intermediate_size=32,
        activation_function="gelu",
        vocab_size=50,
        model_type="gpt2",
    )

    # Create model
    model = create_fake_clt_model(cfg)

    # Run attribution to get a graph
    prompt = torch.tensor([[0, 1, 2, 3]])
    graph = attribute(prompt, model, max_n_logits=3, desired_logit_prob=0.8, batch_size=16)

    # Set scan for graph
    graph.scan = "test-scan"

    # Create graph files in tmp directory
    slug = "test_graph"
    create_graph_files(
        graph_or_path=graph,
        tokenizer=model.tokenizer,
        slug=slug,
        output_path=str(tmp_path),
        scan="test-scan",
        node_threshold=0.8,
        edge_threshold=0.98,
    )

    # Verify the main graph JSON file was created
    graph_file = tmp_path / f"{slug}.json"
    assert graph_file.exists(), f"Graph file {graph_file} was not created"

    # Verify the metadata file was created
    metadata_file = tmp_path / "graph-metadata.json"
    assert metadata_file.exists(), f"Metadata file {metadata_file} was not created"

    # Load and verify graph JSON structure
    with open(graph_file) as f:
        graph_data = json.load(f)

    assert "metadata" in graph_data
    assert "nodes" in graph_data
    assert "links" in graph_data
    assert "qParams" in graph_data

    # Verify metadata fields
    assert graph_data["metadata"]["slug"] == slug
    assert graph_data["metadata"]["scan"] == "test-scan"
    assert graph_data["metadata"]["node_threshold"] == 0.8
    assert "prompt" in graph_data["metadata"]
    assert "prompt_tokens" in graph_data["metadata"]

    # Verify nodes and links exist
    assert len(graph_data["nodes"]) > 0, "No nodes in graph"
    assert len(graph_data["links"]) > 0, "No links in graph"

    # Verify graph-metadata.json structure
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert "graphs" in metadata, "Metadata should have 'graphs' key"
    assert isinstance(metadata["graphs"], list), "Metadata['graphs'] should be a list"
    assert len(metadata["graphs"]) > 0, "Metadata graphs list is empty"
    assert any(m["slug"] == slug for m in metadata["graphs"]), f"Slug {slug} not found in metadata"


def test_create_graph_files_with_path(tmp_path):
    """Test that create_graph_files works with a path to a saved graph."""
    # Create minimal config
    cfg = GPT2Config(
        n_layer=2,
        n_embd=8,
        n_ctx=16,
        n_head=2,
        intermediate_size=32,
        activation_function="gelu",
        vocab_size=50,
        model_type="gpt2",
    )

    # Create model and graph
    model = create_fake_clt_model(cfg)
    prompt = torch.tensor([[0, 1, 2, 3]])
    graph = attribute(prompt, model, max_n_logits=3, desired_logit_prob=0.8, batch_size=16)
    graph.scan = "test-scan"

    # Save graph to pt file
    graph_pt_path = tmp_path / "test_graph.pt"
    graph.to_pt(str(graph_pt_path))

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create graph files from the saved pt file
    slug = "test_from_path"
    create_graph_files(
        graph_or_path=str(graph_pt_path),
        tokenizer=model.tokenizer,
        slug=slug,
        output_path=str(output_dir),
        scan="test-scan",
        node_threshold=0.8,
        edge_threshold=0.98,
    )

    # Verify files were created
    graph_file = output_dir / f"{slug}.json"
    assert graph_file.exists(), f"Graph file {graph_file} was not created"

    metadata_file = output_dir / "graph-metadata.json"
    assert metadata_file.exists(), f"Metadata file {metadata_file} was not created"

    # Verify the graph data is valid
    with open(graph_file) as f:
        graph_data = json.load(f)

    assert graph_data["metadata"]["slug"] == slug
    assert len(graph_data["nodes"]) > 0
    assert len(graph_data["links"]) > 0
