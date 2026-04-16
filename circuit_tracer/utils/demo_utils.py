import gc
import html
import json
import urllib.parse
from collections import namedtuple

import torch
from IPython.display import HTML, display

from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.graph import compute_node_influence

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])


def get_unembed_vecs(model, token_ids: list[int], backend: str) -> list[torch.Tensor]:
    """Extract unembedding column vectors for the given token IDs.

    Handles the orientation difference between TransformerLens (d_model, d_vocab)
    and NNSight (d_vocab, d_model) unembedding matrices.

    Args:
        model: A ``ReplacementModel`` instance.
        token_ids: Vocabulary indices whose unembed columns to extract.
        backend: ``"transformerlens"`` or ``"nnsight"``.

    Returns:
        List of 1-D tensors, one per token ID, each of shape ``(d_model,)``.
    """
    unembed = model.unembed.W_U if backend == "transformerlens" else model.unembed_weight
    d_vocab = model.tokenizer.vocab_size
    if unembed.shape[0] == d_vocab:
        return [unembed[tid] for tid in token_ids]
    return [unembed[:, tid] for tid in token_ids]


def cleanup_cuda() -> None:
    """Run garbage collection and free CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def get_top_features(graph, n: int = 10) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Extract the top-N feature nodes from the graph by total multi-hop influence.

    Uses ``compute_node_influence`` to rank features by their total effect
    on *all* logit targets (direct + indirect paths), weighted by each
    target's probability.

    Args:
        graph: A Graph object with ``adjacency_matrix``, ``selected_features``,
            ``active_features``, ``logit_targets``, and ``logit_probabilities``.
        n: Number of top features to return.

    Returns:
        Tuple of (features, scores) where *features* is a list of
        ``(layer, pos, feature_idx)`` tuples and *scores* is the
        corresponding influence values.
    """
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)

    # Build logit weight vector
    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Multi-hop influence across all logit targets
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    feature_influence = node_influence[:n_features]

    top_k = min(n, n_features)
    top_values, top_indices = torch.topk(feature_influence, top_k)

    features = [
        tuple(graph.active_features[graph.selected_features[i]].tolist()) for i in top_indices
    ]
    scores = top_values.tolist()
    return features, scores


def display_top_features_comparison(
    feature_sets: dict[str, list[tuple[int, int, int]]],
    scores_sets: dict[str, list[float]] | None = None,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
):
    """Display top features from multiple attribution configurations side by side.

    Args:
        feature_sets: Mapping from config label to list of ``(layer, pos, feat_idx)`` tuples.
        scores_sets: Optional mapping from config label to list of attribution scores.
            If ``None``, scores are omitted from the display.
        neuronpedia_model: Neuronpedia model slug (e.g. ``"gemma-2-2b"``).
            When provided, feature indices become clickable links.
        neuronpedia_set: Neuronpedia set name (default ``"gemmascope-transcoder-16k"``).
    """
    labels = list(feature_sets.keys())
    colors = ["#2471A3", "#27AE60", "#8E44AD", "#E67E22", "#C0392B", "#16A085"]

    style = """
    <style>
    .features-cmp {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }
    .features-cmp .col {
        flex: 1;
        min-width: 220px;
    }
    .features-cmp .col-header {
        font-weight: bold;
        font-size: 14px;
        padding: 4px 8px;
        border-radius: 3px;
        color: white;
        margin-bottom: 6px;
    }
    .features-cmp table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .features-cmp th, .features-cmp td {
        text-align: left;
        padding: 3px 6px;
        border: 1px solid rgba(150,150,150,0.5);
    }
    .features-cmp th {
        background-color: rgba(200,200,200,0.3);
        font-weight: bold;
    }
    .features-cmp .monospace { font-family: monospace; }
    .features-cmp a.np-link {
        color: inherit;
        text-decoration: none;
        border-bottom: 1px dashed rgba(150,150,150,0.6);
    }
    .features-cmp a.np-link:hover {
        color: #2980B9;
        border-bottom-style: solid;
    }
    </style>
    """

    body = '<div class="features-cmp">'
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        features = feature_sets[label]
        scores = scores_sets.get(label) if scores_sets else None
        body += '<div class="col">'
        body += (
            f'<div class="col-header" style="background-color: {color};">{html.escape(label)}</div>'
        )
        body += "<table><thead><tr><th>#</th><th>Node</th>"
        if scores is not None:
            body += "<th>Score</th>"
        body += "</tr></thead><tbody>"
        for j, (layer, pos, feat_idx) in enumerate(features):
            score_cell = f"<td>{scores[j]:.4f}</td>" if scores is not None else ""
            if neuronpedia_model is not None:
                np_url = (
                    f"https://www.neuronpedia.org/{neuronpedia_model}/"
                    f"{layer}-{neuronpedia_set}/{feat_idx}"
                )
                feat_link = f'<a class="np-link" href="{np_url}" target="_blank" title="View on Neuronpedia">{feat_idx}</a>'
            else:
                feat_link = str(feat_idx)
            node_cell = f'<td class="monospace">({layer},&#8239;{pos},&#8239;{feat_link})</td>'
            body += f"<tr><td>{j + 1}</td>{node_cell}{score_cell}</tr>"
        body += "</tbody></table></div>"
    body += "</div>"

    display(HTML(style + body))


def display_attribution_config(
    token_pairs: list[tuple[str, int]],
    target_pairs: list[tuple[str, CustomTarget]],
) -> None:
    """Display token-mapping and custom-target summary tables.

    Args:
        token_pairs: List of ``(token_str, vocab_id)`` pairs for the Token Mappings table.
        target_pairs: List of ``(kind_label, target)`` pairs for the Attribution Targets
            table, where each ``target`` is a CustomTarget with ``.token_str`` and ``.prob`` attributes.
    """
    th_l = "padding:5px 14px 5px 6px; border-bottom:2px solid #888; text-align:left; white-space:nowrap"
    th_r = "padding:5px 14px 5px 6px; border-bottom:2px solid #888; text-align:right; white-space:nowrap"
    td_l = "padding:4px 14px 4px 6px; border-bottom:1px solid #ddd; text-align:left"
    td_r = "padding:4px 14px 4px 6px; border-bottom:1px solid #ddd; text-align:right"

    # ── Token Mappings ────────────────────────────────────────────────────────
    token_rows = "".join(
        "<tr>"
        "<td style='" + td_l + "'><code>" + html.escape(tok) + "</code></td>"
        "<td style='" + td_r + "'>" + str(vid) + "</td>"
        "</tr>"
        for tok, vid in token_pairs
    )
    display(
        HTML(
            "<b>Token Mappings</b>"
            "<table style='border-collapse:collapse; font-size:0.9em; margin-top:4px'>"
            "<thead><tr>"
            "<th style='" + th_l + "'>Token</th>"
            "<th style='" + th_r + "'>Vocab ID</th>"
            "</tr></thead>"
            "<tbody>" + token_rows + "</tbody>"
            "</table>"
        )
    )

    # ── Attribution Targets ───────────────────────────────────────────────────
    target_rows = "".join(
        "<tr>"
        "<td style='" + td_l + "'>" + html.escape(kind) + "</td>"
        "<td style='" + td_l + "'><code>" + html.escape(tgt.token_str) + "</code></td>"
        "<td style='" + td_r + "'>" + f"{tgt.prob * 100:.3f}%" + "</td>"
        "</tr>"
        for kind, tgt in target_pairs
    )
    display(
        HTML(
            "<b style='margin-top:12px; display:block'>Attribution Targets</b>"
            "<table style='border-collapse:collapse; font-size:0.9em; margin-top:4px'>"
            "<thead><tr>"
            "<th style='" + th_l + "'>Target</th>"
            "<th style='" + th_l + "'>Label</th>"
            "<th style='" + th_r + "'>Probability</th>"
            "</tr></thead>"
            "<tbody>" + target_rows + "</tbody>"
            "</table>"
        )
    )


def display_token_probs(
    logits: torch.Tensor,
    token_ids: list[int],
    labels: list[str],
    title: str = "",
) -> None:
    """Display softmax probabilities for specific tokens as a styled HTML table.

    Probabilities are shown as percentages (3 decimal places) when ≥ 0.001,
    otherwise in scientific notation (2 significant figures).

    Args:
        logits: Raw logits tensor (at least 2-D; last position is used).
        token_ids: Vocabulary indices to display.
        labels: Human-readable label for each token.
        title: Optional heading rendered above the table.
    """
    probs = torch.softmax(logits.squeeze(0)[-1].float(), dim=-1)

    def _fmt(p: float) -> str:
        return f"{p * 100:.3f}%" if p >= 1e-3 else f"{p:.2e}"

    rows = ""
    for i, (tid, label) in enumerate(zip(token_ids, labels)):
        p = probs[tid].item()
        logit_val = logits.squeeze(0)[-1, tid].item()
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        rows += (
            f'<tr class="{row_class}">'
            f'<td class="monospace">{html.escape(label)}</td>'
            f'<td style="text-align:right;">{_fmt(p)}</td>'
            f'<td style="text-align:right;">{logit_val:.4f}</td>'
            f"</tr>\n"
        )

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;border-radius:3px;background:#555;color:white;display:inline-block;">{html.escape(title)}</div>'
        if title
        else ""
    )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:420px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);">Token</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);">Probability</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);">Logit</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))


def display_ablation_chart(
    groups: dict[str, dict[str, float]],
    logit_diffs: dict[str, float] | None = None,
    title: str = "",
    colors: list[str] | None = None,
) -> None:
    """Display ablation results as a grouped bar chart with logit-difference line.

    Args:
        groups: Mapping from group label (e.g. ``"Baseline"``) to a dict
            of ``{token_label: probability}``.
        logit_diffs: Optional mapping from group label to logit difference,
            plotted as a dashed line on a secondary y-axis.
        title: Chart title.
        colors: Bar colours for each token.  Defaults to a built-in palette.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    group_labels = list(groups.keys())
    token_labels = list(next(iter(groups.values())).keys())
    n_groups = len(group_labels)
    n_tokens = len(token_labels)

    if colors is None:
        colors = ["#2471A3", "#E67E22", "#27AE60", "#C0392B", "#8E44AD"][:n_tokens]

    x = np.arange(n_groups)
    width = 0.8 / n_tokens

    fig, ax1 = plt.subplots(figsize=(8, 5.0))

    for i, tok in enumerate(token_labels):
        vals = [groups[g].get(tok, 0) for g in group_labels]
        offset = (i - (n_tokens - 1) / 2) * width
        bars = ax1.bar(
            x + offset,
            vals,
            width * 0.9,
            label=tok,
            color=colors[i],
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    max_prob = max(max(groups[g].get(t, 0) for t in token_labels) for g in group_labels)
    ax1.set_ylim(0, max_prob * 1.4)

    if logit_diffs is not None:
        ax2 = ax1.twinx()
        diff_vals = [logit_diffs.get(g, 0) for g in group_labels]
        ax2.plot(
            x,
            diff_vals,
            "k--o",
            label="Logit diff",
            linewidth=1.5,
            markersize=5,
        )
        ax2.set_ylabel("Logit difference")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    if title:
        ax1.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)]


# Now let's create a version that's more adaptive to dark/light mode
def display_topk_token_predictions(
    sentence,
    original_logits,
    new_logits,
    tokenizer,
    k: int = 5,
    key_tokens: list[tuple[str, int]] | None = None,
):
    """Display top-k token predictions before and after an intervention.

    Adaptive to both dark and light modes using higher-contrast elements
    and CSS variables where possible.

    Args:
        sentence: The input prompt string.
        original_logits: Logits before the intervention.
        new_logits: Logits after the intervention.
        tokenizer: Tokenizer for decoding token IDs.
        k: Number of top tokens to show per section.
        key_tokens: Optional list of ``(token_label, token_id)`` pairs.
            When provided, a third *Key Tokens* table is rendered showing
            the probabilities of these specific tokens in both the original
            and new distributions.
    """

    original_tokens = get_topk(original_logits, tokenizer, k)
    new_tokens = get_topk(new_logits, tokenizer, k)

    # This version uses a technique that will work better in dark mode
    # by using a combination of background colors and border styling
    html = f"""
    <style>
    .token-viz {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 10px;
        max-width: 700px;
    }}
    .token-viz .header {{
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 3px;
        padding: 4px 6px;
        border-radius: 3px;
        color: white;
        display: inline-block;
    }}
    .token-viz .sentence {{
        background-color: rgba(200, 200, 200, 0.2);
        padding: 4px 6px;
        border-radius: 3px;
        border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace;
        margin-bottom: 8px;
        font-weight: 500;
        font-size: 14px;
    }}
    .token-viz table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 8px;
        font-size: 13px;
        table-layout: fixed;
    }}
    .token-viz th {{
        text-align: left;
        padding: 4px 6px;
        font-weight: bold;
        border: 1px solid rgba(150, 150, 150, 0.5);
        background-color: rgba(200, 200, 200, 0.3);
    }}
    .token-viz td {{
        padding: 3px 6px;
        border: 1px solid rgba(150, 150, 150, 0.5);
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .token-viz .token-col {{
        width: 20%;
    }}
    .token-viz .prob-col {{
        width: 15%;
    }}
    .token-viz .dist-col {{
        width: 65%;
    }}
    .token-viz .monospace {{
        font-family: monospace;
    }}
    .token-viz .bar-container {{
        display: flex;
        align-items: center;
    }}
    .token-viz .bar {{
        height: 12px;
        min-width: 2px;
    }}
    .token-viz .bar-text {{
        margin-left: 6px;
        font-weight: 500;
        font-size: 12px;
    }}
    .token-viz .even-row {{
        background-color: rgba(240, 240, 240, 0.1);
    }}
    .token-viz .odd-row {{
        background-color: rgba(255, 255, 255, 0.1);
    }}
    </style>
    
    <div class="token-viz">
        <div class="header" style="background-color: #555555;">Input Sentence:</div>
        <div class="sentence">{sentence}</div>
        
        <div>
            <div class="header" style="background-color: #2471A3;">Original Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Calculate max probability for scaling
    max_prob = max(
        max([prob for _, prob in original_tokens]),
        max([prob for _, prob in new_tokens]),
    )

    # Add rows for original tokens
    for i, (token, prob) in enumerate(original_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        html += f"""
                    <tr class="{row_class}">
                        <td class="monospace token-col" title="{token}">{token}</td>
                        <td class="prob-col" style="text-align: right;">{prob:.3f}</td>
                        <td class="dist-col">
                            <div class="bar-container">
                                <div class="bar" style="background-color: #2471A3; width: {bar_width}%;"></div>
                                <span class="bar-text">{prob * 100:.1f}%</span>
                            </div>
                        </td>
                    </tr>
        """

    # Add new tokens table
    html += f"""
                </tbody>
            </table>
            
            <div class="header" style="background-color: #27AE60;">New Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add rows for new tokens
    for i, (token, prob) in enumerate(new_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        html += f"""
                    <tr class="{row_class}">
                        <td class="monospace token-col" title="{token}">{token}</td>
                        <td class="prob-col" style="text-align: right;">{prob:.3f}</td>
                        <td class="dist-col">
                            <div class="bar-container">
                                <div class="bar" style="background-color: #27AE60; width: {bar_width}%;"></div>
                                <span class="bar-text">{prob * 100:.1f}%</span>
                            </div>
                        </td>
                    </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    """

    # Optional key-tokens section
    if key_tokens:
        orig_probs = torch.softmax(original_logits.squeeze()[-1], dim=-1)
        new_probs = torch.softmax(new_logits.squeeze()[-1], dim=-1)

        html += """
        <div>
            <div class="header" style="background-color: #8E44AD;">Key Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Original</th>
                        <th class="prob-col" style="text-align: right;">New</th>
                        <th class="dist-col">Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        for i, (label, tid) in enumerate(key_tokens):
            p_orig = orig_probs[tid].item()
            p_new = new_probs[tid].item()
            relative = (p_new - p_orig) / max(p_orig, 1e-9)
            sign = "+" if relative >= 0 else ""
            bar_width = int(p_new / max(max_prob, 1e-9) * 100)
            row_class = "even-row" if i % 2 == 0 else "odd-row"
            html += f"""
                    <tr class="{row_class}">
                        <td class="monospace token-col" title="{label}">{label}</td>
                        <td class="prob-col" style="text-align: right;">{p_orig:.4f}</td>
                        <td class="prob-col" style="text-align: right;">{p_new:.4f}</td>
                        <td class="dist-col">
                            <div class="bar-container">
                                <div class="bar" style="background-color: #8E44AD; width: {bar_width}%;"></div>
                                <span class="bar-text">{sign}{relative * 100:.1f}%</span>
                            </div>
                        </td>
                    </tr>
            """
        html += """
                </tbody>
            </table>
        </div>
        """

    html += """
    </div>
    """

    display(HTML(html))


def display_generations_comparison(original_text, pre_intervention_gens, post_intervention_gens):
    """
    Display a comparison of pre-intervention and post-intervention generations
    with the new/continuation text highlighted.
    """
    # Build the HTML with CSS for styling
    html_content = """
    <style>
    .generations-viz {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 12px;
        font-size: 13px;
        max-width: 700px;
    }
    .generations-viz .section-header {
        font-weight: bold;
        font-size: 14px;
        margin: 10px 0 5px 0;
        padding: 4px 6px;
        border-radius: 3px;
        color: white;
        display: block;
    }
    .generations-viz .pre-intervention-header {
        background-color: #2471A3;
    }
    .generations-viz .post-intervention-header {
        background-color: #27AE60;
    }
    .generations-viz .generation-container {
        margin-bottom: 8px;
        padding: 3px;
        border-left: 3px solid rgba(100, 100, 100, 0.5);
    }
    .generations-viz .generation-text {
        background-color: rgba(200, 200, 200, 0.2);
        padding: 6px 8px;
        border-radius: 3px;
        border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace;
        font-weight: 500;
        white-space: pre-wrap;
        line-height: 1.2;
        font-size: 13px;
        overflow-x: auto;
    }
    .generations-viz .base-text {
        color: rgba(100, 100, 100, 0.9);
    }
    .generations-viz .new-text {
        background-color: rgba(255, 255, 0, 0.25);
        font-weight: bold;
        padding: 1px 0;
        border-radius: 2px;
    }
    .generations-viz .pre-intervention-item {
        border-left-color: #2471A3;
    }
    .generations-viz .post-intervention-item {
        border-left-color: #27AE60;
    }
    .generations-viz .generation-number {
        font-weight: bold;
        margin-bottom: 3px;
        color: rgba(70, 70, 70, 0.9);
        font-size: 12px;
    }
    </style>
    
    <div class="generations-viz">
    """

    # Add pre-intervention section
    html_content += """
    <div class="section-header pre-intervention-header">Pre-intervention generations:</div>
    """

    # Add each pre-intervention generation
    for i, gen_text in enumerate(pre_intervention_gens):
        # Split the text to highlight the continuation
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text) :])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)

        html_content += f"""
        <div class="generation-container pre-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """

    # Add post-intervention section
    html_content += """
    <div class="section-header post-intervention-header">Post-intervention generations:</div>
    """

    # Add each post-intervention generation
    for i, gen_text in enumerate(post_intervention_gens):
        # Split the text to highlight the continuation
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text) :])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)

        html_content += f"""
        <div class="generation-container post-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """

    html_content += """
    </div>
    """

    display(HTML(html_content))


def decode_url_features(url: str) -> tuple[dict[str, list[Feature]], list[Feature]]:
    """
    Extract both supernode features and individual singleton features from URL.

    Returns:
        tuple of (supernode_features, singleton_features)
        - supernode_features: dict mapping supernode names to lists of Features
        - singleton_features: list of individual Feature objects
    """
    decoded = urllib.parse.unquote(url)

    parsed_url = urllib.parse.urlparse(decoded)
    query_params = urllib.parse.parse_qs(parsed_url.query)

    # Extract supernodes
    supernodes_json = query_params.get("supernodes", ["[]"])[0]
    supernodes_data = json.loads(supernodes_json)

    supernode_features = {}
    name_counts = {}

    for supernode in supernodes_data:
        name = supernode[0]
        node_ids = supernode[1:]

        # Handle duplicate names by adding counter
        if name in name_counts:
            name_counts[name] += 1
            unique_name = f"{name} ({name_counts[name]})"
        else:
            name_counts[name] = 1
            unique_name = name

        nodes = []
        for node_id in node_ids:
            layer, feature_idx, pos = map(int, node_id.split("_"))
            nodes.append(Feature(layer, pos, feature_idx))

        supernode_features[unique_name] = nodes

    # Extract individual/singleton features from pinnedIds
    pinned_ids_str = query_params.get("pinnedIds", [""])[0]
    singleton_features = []

    if pinned_ids_str:
        pinned_ids = pinned_ids_str.split(",")
        for pinned_id in pinned_ids:
            # Handle both regular format (layer_feature_pos) and E_ format
            if pinned_id.startswith("E_"):
                # E_26865_9 format - embedding layer
                parts = pinned_id[2:].split("_")  # Remove 'E_' prefix
                if len(parts) == 2:
                    feature_idx, pos = map(int, parts)
                    # Use -1 to indicate embedding layer
                    singleton_features.append(Feature(-1, pos, feature_idx))
            else:
                # Regular layer_feature_pos format
                parts = pinned_id.split("_")
                if len(parts) == 3:
                    layer, feature_idx, pos = map(int, parts)
                    singleton_features.append(Feature(layer, pos, feature_idx))

    return supernode_features, singleton_features


# Keep the old function for backward compatibility
def extract_supernode_features(url: str) -> dict[str, list[Feature]]:
    """Legacy function - only extracts supernode features"""
    supernode_features, _ = decode_url_features(url)
    return supernode_features
