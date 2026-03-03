"""Convert neural network modules to computation graphs.

Represents each parameter tensor as an edge in the computation graph,
and each layer/operation as a node, following the Graph Metanetwork paradigm
from Lim et al. (2023) and Kofinas et al. (2024).

The graph representation enables GNNs to process arbitrary neural network
architectures while respecting permutation symmetries of the parameter space.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NNGraph:
    """Graph representation of a neural network.

    Attributes:
        node_features: Tensor of shape (num_nodes, node_feat_dim) with per-node
            features (e.g., layer type encoding, bias indicator).
        edge_index: Tensor of shape (2, num_edges) with source/target node indices.
        edge_features: Tensor of shape (num_edges, edge_feat_dim) with per-edge
            features derived from network parameters (e.g., fan_in, fan_out, stats).
        param_shapes: List of original parameter shapes, one per edge.
        param_refs: List of (module_name, param_name) tuples mapping edges back
            to the original parameters.
        node_types: Optional tensor of node type IDs.
    """
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    param_shapes: List[torch.Size]
    param_refs: List[Tuple[str, str]]
    node_types: Optional[torch.Tensor] = None


# Supported layer types and their integer codes
LAYER_TYPE_MAP = {
    "Linear": 0,
    "Conv1d": 1,
    "Conv2d": 2,
    "Conv3d": 3,
    "BatchNorm1d": 4,
    "BatchNorm2d": 5,
    "LayerNorm": 6,
    "GroupNorm": 7,
    "Embedding": 8,
    "MultiheadAttention": 9,
    "input": 10,
    "output": 11,
    "other": 12,
}

NODE_FEAT_DIM = 16  # base node feature dimensionality
EDGE_FEAT_DIM = 8   # base edge feature dimensionality


def _get_layer_type_id(module: nn.Module) -> int:
    """Return the integer type code for a module."""
    name = type(module).__name__
    return LAYER_TYPE_MAP.get(name, LAYER_TYPE_MAP["other"])


def _compute_param_edge_features(param: torch.Tensor) -> torch.Tensor:
    """Compute statistical features for a parameter tensor.

    Returns a 1-D tensor of EDGE_FEAT_DIM features:
        [numel_log, ndim, mean, std, abs_mean, max, min, norm_log]
    """
    with torch.no_grad():
        flat = param.detach().float().flatten()
        numel_log = torch.tensor(flat.numel()).float().log1p()
        ndim = torch.tensor(param.ndim, dtype=torch.float32)
        mean = flat.mean()
        std = flat.std() if flat.numel() > 1 else torch.tensor(0.0)
        abs_mean = flat.abs().mean()
        max_val = flat.max()
        min_val = flat.min()
        norm_log = flat.norm().log1p()
    return torch.stack([numel_log, ndim, mean, std, abs_mean, max_val, min_val, norm_log])


def _one_hot(idx: int, num_classes: int) -> torch.Tensor:
    vec = torch.zeros(num_classes)
    if 0 <= idx < num_classes:
        vec[idx] = 1.0
    return vec


def nn_to_graph(
    model: nn.Module,
    include_buffers: bool = False,
) -> NNGraph:
    """Convert a PyTorch model to an NNGraph.

    Each module with parameters becomes a node. Each parameter tensor connecting
    two consecutive computation stages becomes an edge. An ``input`` and ``output``
    virtual node are added to anchor the graph.

    Args:
        model: The neural network to convert.
        include_buffers: Whether to include registered buffers as edges.

    Returns:
        An ``NNGraph`` instance.
    """
    num_type_classes = len(LAYER_TYPE_MAP)
    nodes: List[Dict] = []
    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_feats: List[torch.Tensor] = []
    param_shapes: List[torch.Size] = []
    param_refs: List[Tuple[str, str]] = []

    # Virtual input node (index 0)
    nodes.append({"type_id": LAYER_TYPE_MAP["input"], "has_bias": False})

    # Collect parameterized modules in order
    named_modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if list(mod.parameters(recurse=False)) or (
            include_buffers and list(mod.buffers(recurse=False))
        )
    ]

    for name, mod in named_modules:
        type_id = _get_layer_type_id(mod)
        has_bias = any(
            pname == "bias"
            for pname, _ in mod.named_parameters(recurse=False)
        )
        nodes.append({"type_id": type_id, "has_bias": has_bias})

    # Virtual output node
    nodes.append({"type_id": LAYER_TYPE_MAP["output"], "has_bias": False})

    # Build edges: chain consecutive module nodes
    # input -> module_0 -> module_1 -> ... -> module_N -> output
    module_node_ids = list(range(1, len(nodes) - 1))

    for idx, (name, mod) in enumerate(named_modules):
        node_id = module_node_ids[idx]
        src = node_id - 1 if idx == 0 else module_node_ids[idx - 1]
        dst = node_id

        for pname, param in mod.named_parameters(recurse=False):
            edges_src.append(src)
            edges_dst.append(dst)
            edge_feats.append(_compute_param_edge_features(param))
            param_shapes.append(param.shape)
            param_refs.append((name, pname))

        if include_buffers:
            for bname, buf in mod.named_buffers(recurse=False):
                edges_src.append(src)
                edges_dst.append(dst)
                edge_feats.append(_compute_param_edge_features(buf))
                param_shapes.append(buf.shape)
                param_refs.append((name, bname))

    # Edge from last module to output
    if module_node_ids:
        last_mod_id = module_node_ids[-1]
        out_id = len(nodes) - 1
        edges_src.append(last_mod_id)
        edges_dst.append(out_id)
        edge_feats.append(torch.zeros(EDGE_FEAT_DIM))
        param_shapes.append(torch.Size([]))
        param_refs.append(("__output__", ""))

    # Build node features
    node_feat_list = []
    for n in nodes:
        type_vec = _one_hot(n["type_id"], num_type_classes)
        bias_flag = torch.tensor([1.0 if n["has_bias"] else 0.0])
        # Pad to NODE_FEAT_DIM
        feat = torch.cat([type_vec, bias_flag])
        pad_len = NODE_FEAT_DIM - feat.numel()
        if pad_len > 0:
            feat = torch.cat([feat, torch.zeros(pad_len)])
        node_feat_list.append(feat[:NODE_FEAT_DIM])

    node_features = torch.stack(node_feat_list)

    if edge_feats:
        edge_features = torch.stack(edge_feats)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_features = torch.zeros(0, EDGE_FEAT_DIM)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return NNGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        param_shapes=param_shapes,
        param_refs=param_refs,
    )


def build_optimizer_features(
    model: nn.Module,
    graph: NNGraph,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    step: int = 0,
    loss_val: float = 0.0,
) -> torch.Tensor:
    """Build per-edge optimizer input features.

    For each parameter edge in the graph, concatenates:
        - Current parameter statistics (from edge_features)
        - Gradient statistics (mean, std, norm) if grads provided
        - Normalized training step
        - Loss value

    Returns tensor of shape (num_edges, feat_dim).
    """
    num_edges = graph.edge_features.shape[0]
    extra_feats = []

    for i, (mod_name, p_name) in enumerate(graph.param_refs):
        feats = [graph.edge_features[i]]

        # Gradient features
        if grads is not None and mod_name != "__output__":
            full_name = f"{mod_name}.{p_name}" if mod_name else p_name
            g = grads.get(full_name)
            if g is not None:
                gf = g.detach().float().flatten()
                g_mean = gf.mean()
                g_std = gf.std() if gf.numel() > 1 else torch.tensor(0.0)
                g_norm = gf.norm().log1p()
                feats.append(torch.stack([g_mean, g_std, g_norm]))
            else:
                feats.append(torch.zeros(3))
        else:
            feats.append(torch.zeros(3))

        # Step and loss
        step_feat = torch.tensor([step / 10000.0])
        loss_feat = torch.tensor([loss_val])
        feats.append(step_feat)
        feats.append(loss_feat)

        extra_feats.append(torch.cat(feats))

    if extra_feats:
        return torch.stack(extra_feats)
    return torch.zeros(num_edges, EDGE_FEAT_DIM + 5)
