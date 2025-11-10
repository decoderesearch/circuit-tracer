import argparse
import logging
import os
import time
import warnings


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="CLI for attribution, graph file creation, and server hosting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Attribution subcommand
    attr_parser = subparsers.add_parser("attribute", help="Run attribution analysis on a prompt")

    # Arguments from attribute_batch.py
    attr_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=("Model architecture to use for attribution. Can be inferred from transcoder config."),
    )
    attr_parser.add_argument(
        "-t",
        "--transcoder_set",
        required=True,
        help=(
            "HuggingFace repository ID containing transcoders "
            "(e.g. username/repo-name, username/repo-name@revision)."
        ),
    )
    attr_parser.add_argument("-p", "--prompt", required=True, help="Input prompt text to analyze.")
    attr_parser.add_argument(
        "-o",
        "--graph_output_path",
        help=(
            "Path where to save the attribution graph (.pt file). Required if not "
            "creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights (default: float32).",
    )
    attr_parser.add_argument(
        "--max_n_logits", type=int, default=10, help="Maximum number of logit nodes."
    )
    attr_parser.add_argument(
        "--desired_logit_prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for top logits.",
    )
    attr_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for backward passes."
    )
    attr_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory.",
    )
    attr_parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=7500,
        help="Maximum number of feature nodes.",
    )
    attr_parser.add_argument("--verbose", action="store_true", help="Display progress information.")
    attr_parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help="Enable lazy loading for encoder weights to save memory.",
    )
    attr_parser.add_argument(
        "--lazy-decoder",
        action="store_true",
        default=True,
        help="Enable lazy loading for decoder weights to save memory (default: True).",
    )

    # Arguments for graph creation
    attr_parser.add_argument(
        "--slug",
        type=str,
        help=(
            "Slug for the model metadata (used for graph files). Required if creating "
            "graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--graph_file_dir",
        type=str,
        help=(
            "Path to save the output JSON graph files, and also used as data dir for "
            "server. Required if creating graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Node threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Edge threshold for pruning graph files.",
    )

    # Server arguments
    attr_parser.add_argument(
        "--server",
        action="store_true",
        help="Start a local server to visualize graphs after processing.",
    )
    attr_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")

    # Start-server subcommand
    server_parser = subparsers.add_parser(
        "start-server", help="Start a local server to visualize existing graphs"
    )
    server_parser.add_argument(
        "--graph_file_dir",
        type=str,
        required=True,
        help="Path to the directory containing graph JSON files.",
    )
    server_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")

    args = parser.parse_args()

    if args.command == "attribute":
        run_attribution(args, attr_parser)
    if args.command == "start-server" or args.server:
        run_server(args)


def run_attribution(args, parser):
    # Check if one of slug/graph_file_dir is provided but not the other
    if bool(args.slug) != bool(args.graph_file_dir):
        which_one = "slug" if args.slug else "graph_file_dir"
        missing_one = "graph_file_dir" if args.slug else "slug"
        warnings.warn(
            (
                f"You provided --{which_one} but not --{missing_one}. Both are required "
                "for creating graph files."
            ),
            UserWarning,
        )

    # Determine if we're creating graph files
    create_graph_files_enabled = args.slug is not None and args.graph_file_dir is not None

    # Validate arguments
    if args.server and (not args.slug or not args.graph_file_dir):
        parser.error("Both --slug and --graph_file_dir are required when using --server")

    if not create_graph_files_enabled and not args.graph_output_path:
        parser.error(
            "--graph_output_path is required when not creating graph files "
            "(--slug and --graph_file_dir)"
        )

    # Ensure graph output directory exists if needed
    if create_graph_files_enabled:
        os.makedirs(args.graph_file_dir, exist_ok=True)

    import torch

    dtype = args.dtype
    # Convert short dtype string to long dtype string
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Run attribution
    logging.info(f"Generating attribution graph for model: {args.model}")
    logging.info(f"Loading model with dtype: {dtype}")
    logging.info(f'Input prompt: "{args.prompt}"')
    if args.graph_output_path:
        logging.info(f"Output will be saved to: {args.graph_output_path}")
    logging.info(
        f"Including logits with cumulative probability >= {args.desired_logit_prob} "
        f"(max {args.max_n_logits})"
    )
    logging.info(f"Using batch size of {args.batch_size} for backward passes")

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=args.lazy_encoder,
        lazy_decoder=args.lazy_decoder,
    )
    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    model_instance = ReplacementModel.boot_transformers_and_transcoders(
        args.model, transcoder, dtype=dtype
    )

    logging.info("Running attribution...")
    graph = attribute(
        prompt=args.prompt,
        model=model_instance,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        verbose=args.verbose,
        offload=args.offload,
        max_feature_nodes=args.max_feature_nodes,
    )

    # Save to file if output path specified
    if args.graph_output_path:
        logging.info(f"Saving graph to {args.graph_output_path}")
        graph.to_pt(args.graph_output_path)

    # Create graph files if both slug and graph_file_dir are provided
    if create_graph_files_enabled:
        logging.info(f"Creating graph files with slug: {args.slug}")
        create_graph_files(
            graph_or_path=graph,  # Use the graph object directly
            tokenizer=model_instance.tokenizer,
            slug=args.slug,
            scan=None,  # No scan argument needed
            output_path=args.graph_file_dir,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        logging.info(f"Graph JSON files written to {args.graph_file_dir}")


def run_server(args):
    from circuit_tracer.frontend.local_server import serve

    logging.info(f"Starting server on port {args.port}...")
    logging.info(f"Serving data from: {os.path.abspath(args.graph_file_dir)}")
    server = serve(data_dir=args.graph_file_dir, port=args.port)
    try:
        logging.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        server.stop()


if __name__ == "__main__":
    main()
