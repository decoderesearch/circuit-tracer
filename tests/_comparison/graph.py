from typing import NamedTuple

import torch
from transformer_lens import HookedTransformerConfig


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    active_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    selected_features: torch.Tensor
    activation_values: torch.Tensor
    logit_probabilities: torch.Tensor
    cfg: HookedTransformerConfig
    scan: str | list[str] | None

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        active_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg: HookedTransformerConfig,
        logit_tokens: torch.Tensor,
        logit_probabilities: torch.Tensor,
        selected_features: torch.Tensor,
        activation_values: torch.Tensor,
        scan: str | list[str] | None = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features, transcoder errors,
        tokens, or logits. They are stored in the order [active_features[0], ...,
        active_features[n-1], error[layer0][position0], error[layer0][position1], ...,
        error[layer l - 1][position t-1], tokens[0], ..., tokens[t-1], logits[top-1 logit],
        ..., logits[top-k logit]].

        Args:
            input_string (str): The input string attributed.
            input_tokens (List[str]): The input tokens attributed.
            active_features (torch.Tensor): A tensor of shape (n_active_features, 3)
                containing the indices (layer, pos, feature_idx) of the non-zero features
                of the model on the given input string.
            adjacency_matrix (torch.Tensor): The adjacency matrix. Organized as
                [active_features, error_nodes, embed_nodes, logit_nodes], where there are
                model.cfg.n_layers * len(input_tokens) error nodes, len(input_tokens) embed
                nodes, len(logit_tokens) logit nodes. The rows represent target nodes, while
                columns represent source nodes.
            cfg (HookedTransformerConfig): The cfg of the model.
            logit_tokens (List[str]): The logit tokens attributed from.
            logit_probabilities (torch.Tensor): The probabilities of each logit token, given
                the input string.
            scan (Optional[Union[str,List[str]]], optional): The identifier of the
                transcoders used in the graph. Without a scan, the graph cannot be uploaded
                (since we won't know what transcoders were used). Defaults to None
        """
        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = cfg
        self.n_pos = len(input_tokens)
        self.active_features = active_features
        self.logit_tokens = logit_tokens
        self.logit_probabilities = logit_probabilities
        self.input_tokens = input_tokens
        if scan is None:
            print("Graph loaded without scan to identify it. Uploading will not be possible.")
        self.scan = scan
        self.selected_features = selected_features
        self.activation_values = activation_values

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)

        Args:
            device (_type_): device to send tensors
        """
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.active_features = self.active_features.to(device)
        self.logit_tokens = self.logit_tokens.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)

    def to_pt(self, path: str):
        """Saves the graph at the given path

        Args:
            path (str): The path where the graph will be saved. Should end in .pt
        """
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "active_features": self.active_features,
            "logit_tokens": self.logit_tokens,
            "logit_probabilities": self.logit_probabilities,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "activation_values": self.activation_values,
            "scan": self.scan,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """Load a graph (saved using graph.to_pt) from a .pt file at the given path.

        Args:
            path (str): The path of the Graph to load
            map_location (str, optional): the device to load the graph onto.
                Defaults to 'cpu'.

        Returns:
            Graph: the Graph saved at the specified path
        """
        d = torch.load(path, weights_only=False, map_location=map_location)
        return Graph(**d)


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    # Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
    # and do logit_weights @ B
    # But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
    # as follows:

    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float):
    # Find score threshold that keeps the desired fraction of total influence
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index: int = int(torch.searchsorted(cumulative_score, threshold).item())
    # make sure we don't go out of bounds (only really happens at threshold=1.0)
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative influence scores for each node


def prune_graph(
    graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98
) -> PruneResult:
    """Prunes a graph by removing nodes and edges with low influence on the output logits.

    Args:
        graph: The graph to prune
        node_threshold: Keep nodes that contribute to this fraction of total influence
        edge_threshold: Keep edges that contribute to this fraction of total influence

    Returns:
        Tuple containing:
        - node_mask: Boolean tensor indicating which nodes to keep
        - edge_mask: Boolean tensor indicating which edges to keep
        - cumulative_scores: Tensor of cumulative influence scores for each node
    """

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    # Extract dimensions
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Calculate node influence and apply threshold
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    # Always keep tokens and logits
    node_mask[-n_logits - n_tokens :] = True

    # Create pruned matrix with selected nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0
    # we could also do iterative pruning here (see below)

    # Calculate edge influence and apply threshold
    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)

    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure feature nodes have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # iteratively prune until all nodes missing incoming / outgoing edges are gone
    # (each pruning iteration potentially opens up new candidates for pruning)
    # this should not take more than n_layers + 1 iterations
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        # Ensure feature and error nodes have outgoing edges
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        # Ensure feature nodes have incoming edges
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Calculate cumulative influence scores
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)


def compute_graph_scores(graph: Graph) -> tuple[float, float]:
    """Compute metrics for evaluating how well the graph captures the model's computation.
    This function calculates two complementary scores that measure how much of the model's
    computation flows through interpretable feature nodes versus reconstruction error nodes:
    1. Replacement Score: Measures the fraction of end-to-end influence from input tokens
       to output logits that flows through feature nodes rather than error nodes. This is
       a strict metric that rewards complete explanations where tokens influence logits
       entirely through features.
    2. Completeness Score: Measures the fraction of incoming edges to all nodes (weighted
       by each node's influence on the output) that originate from feature or token nodes
       rather than error nodes. This metric gives partial credit for nodes that are mostly
       explained by features, even if some error influence remains.
    Args:
        graph: The computation graph containing nodes for features, errors, tokens, and logits,
               along with their connections and influence weights.
    Returns:
        tuple[float, float]: A tuple containing:
            - replacement_score: Fraction of token-to-logit influence through features (0-1)
            - completeness_score: Weighted fraction of non-error inputs across all nodes (0-1)
    Note:
        Higher scores indicate better model interpretability, with 1.0 representing perfect
        reconstruction where all computation flows through interpretable features. Lower
        scores indicate more reliance on error nodes, suggesting incomplete feature coverage.
    """
    n_logits = len(graph.logit_tokens)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers
    token_end = error_end + n_tokens

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    non_error_fractions = 1 - normalized_matrix[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()
