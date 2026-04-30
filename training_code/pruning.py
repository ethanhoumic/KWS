import torch
import torch.nn as nn
from training_code.model import CNNTradFpool3


# =============================================================================
#  Helper: compute L1 norm of each output filter
# =============================================================================

def l1_norm_per_filter(weight: torch.Tensor) -> torch.Tensor:
    """
    weight: (out_ch, in_ch, kH, kW)
    returns: (out_ch,) L1 norm of every output filter
    """
    return weight.detach().abs().sum(dim=(1, 2, 3))


def get_keep_indices(weight: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """
    Return sorted indices of filters to KEEP (highest L1 norm).
    """
    norms = l1_norm_per_filter(weight)
    n_keep = max(1, int(round(len(norms) * (1.0 - prune_ratio))))
    # argsort ascending → take the last n_keep
    keep_idx = torch.argsort(norms, descending=True)[:n_keep]
    keep_idx, _ = torch.sort(keep_idx)          # keep original channel order
    return keep_idx


# =============================================================================
#  Prune a Conv2d layer (output channels only)
# =============================================================================

def prune_conv_out(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new Conv2d with only the kept OUTPUT channels."""
    n_keep = len(keep_idx)
    new_conv = nn.Conv2d(
        in_channels  = conv.in_channels,
        out_channels = n_keep,
        kernel_size  = conv.kernel_size,
        stride       = conv.stride,
        padding      = conv.padding,
        bias         = conv.bias is not None
    )
    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()
    return new_conv


def prune_conv_in(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new Conv2d with only the kept INPUT channels."""
    n_keep = len(keep_idx)
    new_conv = nn.Conv2d(
        in_channels  = n_keep,
        out_channels = conv.out_channels,
        kernel_size  = conv.kernel_size,
        stride       = conv.stride,
        padding      = conv.padding,
        bias         = conv.bias is not None
    )
    new_conv.weight.data = conv.weight.data[:, keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


# =============================================================================
#  Prune Linear input features
# =============================================================================

def prune_linear_in(linear: nn.Linear, keep_idx: torch.Tensor,
                    spatial_size: int) -> nn.Linear:
    """
    After conv2 output channels are pruned, the flattened linear input shrinks.
    spatial_size: number of spatial elements per channel after conv2
                  = H_out * W_out  (must be computed from a dummy forward pass)
    keep_idx: surviving conv2 output channel indices
    """
    # Expanded feature indices: each channel occupies `spatial_size` contiguous slots
    feat_idx = torch.cat([
        torch.arange(ch * spatial_size, (ch + 1) * spatial_size)
        for ch in keep_idx
    ])
    n_keep = len(feat_idx)
    new_linear = nn.Linear(n_keep, linear.out_features,
                           bias=linear.bias is not None)
    new_linear.weight.data = linear.weight.data[:, feat_idx].clone()
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.data.clone()
    return new_linear


# =============================================================================
#  Main pruning function
# =============================================================================

def prune_cnntrad(model: CNNTradFpool3,
                  prune_ratio_conv1: float = 0.5,
                  prune_ratio_conv2: float = 0.5,
                  input_shape: tuple = (1, 1, 101, 40)) -> CNNTradFpool3:
    """
    Perform L1-norm channel pruning on CNNTradFpool3.

    Args:
        model            : trained CNNTradFpool3 instance
        prune_ratio_conv1: fraction of conv1 output channels to remove  (0~1)
        prune_ratio_conv2: fraction of conv2 output channels to remove  (0~1)
        input_shape      : (B, C, T, F) used for spatial size probing

    Returns:
        pruned_model : a new CNNTradFpool3-like nn.Module with smaller conv layers
    """
    model.eval()

    # ------------------------------------------------------------------ #
    #  Step 1 : decide which channels to keep
    # ------------------------------------------------------------------ #
    keep1 = get_keep_indices(model.conv1.weight, prune_ratio_conv1)  # conv1 out
    keep2 = get_keep_indices(model.conv2.weight, prune_ratio_conv2)  # conv2 out

    print(f"conv1 : {model.conv1.out_channels} → {len(keep1)} channels kept")
    print(f"conv2 : {model.conv2.out_channels} → {len(keep2)} channels kept")

    # ------------------------------------------------------------------ #
    #  Step 2 : probe spatial size after pruned conv2
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        x = model.conv1(dummy)
        x = model.relu1(x)
        x = model.conv2(x)
        x = model.relu2(x)
        spatial_size = x.shape[2] * x.shape[3]   # H_out * W_out

    print(f"conv2 spatial size : {spatial_size}")

    # ------------------------------------------------------------------ #
    #  Step 3 : build pruned layers
    # ------------------------------------------------------------------ #
    new_conv1 = prune_conv_out(model.conv1, keep1)          # out: len(keep1)
    new_conv2 = prune_conv_in(
                    prune_conv_out(model.conv2, keep2),
                    keep1)                                  # in: len(keep1), out: len(keep2)
    new_linear = prune_linear_in(model.linear, keep2, spatial_size)

    # ------------------------------------------------------------------ #
    #  Step 4 : assemble pruned model (reuse remaining layers as-is)
    # ------------------------------------------------------------------ #
    pruned = _build_pruned_model(model, new_conv1, new_conv2, new_linear)
    return pruned


def _build_pruned_model(orig, new_conv1, new_conv2, new_linear):
    """Wrap pruned layers back into a module that has the same forward()."""

    class PrunedCNNTrad(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1     = new_conv1
            self.relu1     = orig.relu1
            self.conv2     = new_conv2
            self.relu2     = orig.relu2
            self.dropout   = orig.dropout
            self.linear    = new_linear
            self.dnn       = orig.dnn
            self.relu3     = orig.relu3
            self.classifier = orig.classifier

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.dropout(x)
            x = self.dnn(x)
            x = self.relu3(x)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

    return PrunedCNNTrad()


# =============================================================================
#  Utility : count parameters
# =============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
#  Quick sanity check
# =============================================================================

def verify_output_shape(model: nn.Module,
                        input_shape: tuple = (2, 1, 101, 40),
                        num_classes: int = 12):
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        out = model(dummy)
    assert out.shape == (input_shape[0], num_classes), \
        f"Unexpected output shape: {out.shape}"
    print(f"Output shape OK : {out.shape}")
