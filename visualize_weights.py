"""
=============================================================
  Visualize Weights & Biases of a Llama / Transformer Model
=============================================================

This script shows you EXACTLY what weights and biases look like
inside a neural network (Llama 3.2-1B or any HuggingFace model).

What are Weights & Biases?
--------------------------
- WEIGHTS: Large matrices of numbers that the model "learned" during training.
            They control how information flows between neurons.
            Think of them as the "knowledge" of the model.

- BIASES:  Small vectors added after multiplication.
            They shift the output up or down.
            Think of them as "adjustments" or "offsets".

Together: output = (input x WEIGHT) + BIAS
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "weight_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    """Load the model and tokenizer."""
    print("Loading model (this may take a minute)...")
    print(f"Model: {MODEL_NAME}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def print_model_structure(model):
    """
    Print all layers with their weight shapes and total parameters.
    This shows you the SKELETON of the neural network.
    """
    print("=" * 70)
    print("  MODEL STRUCTURE — All Layers with Weights")
    print("=" * 70)

    total_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        shape = list(param.shape)
        num_params = param.numel()
        total_params += num_params
        layer_info.append((name, shape, num_params))

        # Show first 30 layers
        if len(layer_info) <= 30:
            print(f"  {name:<60s} shape={str(shape):<20s} params={num_params:>12,}")

    if len(layer_info) > 30:
        print(f"  ... and {len(layer_info) - 30} more layers ...")

    print(f"\n  TOTAL PARAMETERS: {total_params:,}")
    print(f"  That's {total_params / 1e9:.2f} BILLION numbers the model learned!\n")
    return layer_info


def visualize_weight_matrix(model):
    """
    Visualization 1: HEATMAP of a weight matrix.

    Each cell = one weight value (a single number).
    Colors show the magnitude:
      - Blue = negative values
      - White = near zero
      - Red  = positive values
    """
    print("─" * 50)
    print("  VIZ 1: Weight Matrix Heatmap")
    print("─" * 50)

    # Get the embedding layer weights (token embeddings)
    embed_weight = model.model.embed_tokens.weight.detach().cpu().numpy()
    print(f"  Layer: embed_tokens (Token Embeddings)")
    print(f"  Shape: {embed_weight.shape}")
    print(f"  → {embed_weight.shape[0]} tokens x {embed_weight.shape[1]} dimensions")
    print(f"  → Each token is represented by {embed_weight.shape[1]} numbers\n")

    # Show a small slice (first 50 tokens x first 64 dimensions)
    slice_data = embed_weight[:50, :64]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(slice_data, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Weight Value')
    ax.set_xlabel('Embedding Dimension (first 64 of 2048)', fontsize=12)
    ax.set_ylabel('Token ID (first 50 of 128,256)', fontsize=12)
    ax.set_title('WEIGHT MATRIX: Token Embeddings\n'
                 'Each row = one token\'s learned representation\n'
                 'Blue = negative, White = zero, Red = positive',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_weight_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def visualize_weight_distribution(model):
    """
    Visualization 2: HISTOGRAM of weight values.

    Shows the DISTRIBUTION of all weight values in a layer.
    Key insight: Most weights are very close to ZERO.
    """
    print("\n" + "─" * 50)
    print("  VIZ 2: Weight Value Distribution")
    print("─" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers_to_plot = [
        ("model.embed_tokens.weight", "Token Embeddings"),
        ("model.layers.0.self_attn.q_proj.weight", "Attention Q (Layer 0)"),
        ("model.layers.0.mlp.gate_proj.weight", "MLP Gate (Layer 0)"),
        ("model.layers.15.self_attn.k_proj.weight", "Attention K (Layer 15)"),
    ]

    for idx, (layer_name, display_name) in enumerate(layers_to_plot):
        ax = axes[idx // 2][idx % 2]

        # Find the parameter
        for name, param in model.named_parameters():
            if name == layer_name:
                values = param.detach().cpu().numpy().flatten()

                ax.hist(values, bins=200, color='steelblue', alpha=0.8, edgecolor='none')
                ax.set_title(f'{display_name}\n{name}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Count')
                ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.7)

                # Add statistics
                stats_text = (f'Mean: {values.mean():.4f}\n'
                              f'Std:  {values.std():.4f}\n'
                              f'Min:  {values.min():.4f}\n'
                              f'Max:  {values.max():.4f}')
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontfamily='monospace')
                break

    fig.suptitle('WEIGHT DISTRIBUTIONS: Most weights cluster near ZERO\n'
                 'Red dashed line = zero',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_weight_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_attention_weights(model):
    """
    Visualization 3: Attention layer weight matrices.

    Q, K, V, O — the four weight matrices in self-attention.
    These control HOW the model "pays attention" to different tokens.
    """
    print("\n" + "─" * 50)
    print("  VIZ 3: Attention Weight Matrices (Q, K, V, O)")
    print("─" * 50)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    attention_parts = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    labels = ['Query (Q)', 'Key (K)', 'Value (V)', 'Output (O)']

    for idx, (part, label) in enumerate(zip(attention_parts, labels)):
        layer_name = f"model.layers.0.self_attn.{part}.weight"
        for name, param in model.named_parameters():
            if name == layer_name:
                data = param.detach().cpu().numpy()
                # Show a 128x128 slice for visibility
                slice_size = min(128, data.shape[0], data.shape[1])
                slice_data = data[:slice_size, :slice_size]

                im = axes[idx].imshow(slice_data, cmap='RdBu_r', aspect='equal',
                                      interpolation='nearest')
                axes[idx].set_title(f'{label}\nShape: {list(data.shape)}',
                                    fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Input dim')
                axes[idx].set_ylabel('Output dim')
                plt.colorbar(im, ax=axes[idx], shrink=0.8)
                break

    fig.suptitle('ATTENTION WEIGHTS (Layer 0): How the model learns to "pay attention"\n'
                 'Each matrix transforms input differently for the attention mechanism',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_layer_norms(model):
    """
    Visualization 4: Layer Normalization weights (these act like biases).

    RMSNorm in Llama doesn't have traditional "bias" terms,
    but the normalization weights serve a similar role.
    """
    print("\n" + "─" * 50)
    print("  VIZ 4: Layer Norm Weights (act like biases)")
    print("─" * 50)

    norm_weights = []
    norm_names = []

    for name, param in model.named_parameters():
        if 'norm' in name.lower() and 'weight' in name:
            norm_weights.append(param.detach().cpu().numpy())
            norm_names.append(name)

    if not norm_weights:
        print("  No norm layers found.")
        return

    # Plot first 6 norm layers
    n_plots = min(6, len(norm_weights))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for idx in range(n_plots):
        ax = axes[idx // 3][idx % 3]
        values = norm_weights[idx]
        x = np.arange(len(values))

        ax.plot(x, values, linewidth=0.5, color='steelblue', alpha=0.8)
        ax.axhline(y=1.0, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_title(norm_names[idx].replace('model.', ''), fontsize=8, fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.tick_params(labelsize=7)

    fig.suptitle('LAYER NORM WEIGHTS: These scale each dimension\n'
                 'Red line = 1.0 (neutral). Deviations = learned adjustments (like biases)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_layer_norm_weights.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_weight_magnitude_across_layers(model):
    """
    Visualization 5: How weight magnitudes change across layers.

    Shows if deeper layers have larger/smaller weights.
    """
    print("\n" + "─" * 50)
    print("  VIZ 5: Weight Magnitude Across All Layers")
    print("─" * 50)

    layer_names = []
    means = []
    stds = []

    for name, param in model.named_parameters():
        values = param.detach().cpu().numpy().flatten()
        layer_names.append(name)
        means.append(np.mean(np.abs(values)))
        stds.append(np.std(values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Mean absolute weight per layer
    ax1.bar(range(len(means)), means, color='steelblue', alpha=0.8, width=1.0)
    ax1.set_ylabel('Mean |Weight|', fontsize=12)
    ax1.set_title('Mean Absolute Weight Value Per Layer\n'
                   'Higher = stronger influence on output',
                   fontsize=13, fontweight='bold')
    ax1.set_xlim(-0.5, len(means) - 0.5)

    # Standard deviation per layer
    ax2.bar(range(len(stds)), stds, color='coral', alpha=0.8, width=1.0)
    ax2.set_xlabel('Layer Index (each bar = one weight tensor)', fontsize=12)
    ax2.set_ylabel('Std Dev of Weights', fontsize=12)
    ax2.set_title('Weight Variance Per Layer\n'
                   'Higher variance = more diverse weight values',
                   fontsize=13, fontweight='bold')
    ax2.set_xlim(-0.5, len(stds) - 0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "5_weight_magnitudes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_raw_weights_sample(model):
    """
    Print actual raw weight numbers so you can SEE what they look like.
    """
    print("\n" + "=" * 70)
    print("  RAW WEIGHT VALUES — What the model actually stores")
    print("=" * 70)

    for name, param in model.named_parameters():
        if 'layers.0.self_attn.q_proj.weight' in name:
            data = param.detach().cpu().numpy()
            print(f"\n  Layer: {name}")
            print(f"  Shape: {data.shape}")
            print(f"  First 5x5 values:\n")
            for i in range(5):
                row = "    [ " + ", ".join(f"{data[i][j]:+.6f}" for j in range(5)) + " ... ]"
                print(row)
            print(f"    ... ({data.shape[0]} rows x {data.shape[1]} columns total)")
            break

    # Show norm weights
    for name, param in model.named_parameters():
        if 'layers.0.input_layernorm.weight' in name:
            data = param.detach().cpu().numpy()
            print(f"\n  Layer: {name}")
            print(f"  Shape: {data.shape} (acts like a bias vector)")
            print(f"  First 20 values:\n")
            vals = ", ".join(f"{data[i]:.4f}" for i in range(20))
            print(f"    [ {vals} ... ]")
            break

    print("\n  KEY INSIGHT: These are just NUMBERS!")
    print("  → Billions of them, learned during training")
    print("  → Small changes in these numbers = different model behavior")
    print("  → Fine-tuning = adjusting these numbers for your data\n")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  WEIGHTS & BIASES VISUALIZER")
    print("  Understanding what's INSIDE a Language Model")
    print("=" * 70 + "\n")

    model, tokenizer = load_model()

    # 1. Show model structure
    print_model_structure(model)

    # 2. Show raw weight numbers
    print_raw_weights_sample(model)

    # 3. Generate visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS...")
    print("=" * 70 + "\n")

    visualize_weight_matrix(model)
    visualize_weight_distribution(model)
    visualize_attention_weights(model)
    visualize_layer_norms(model)
    visualize_weight_magnitude_across_layers(model)

    print("\n" + "=" * 70)
    print(f"  ALL DONE! Check the '{OUTPUT_DIR}/' folder for 5 images:")
    print("=" * 70)
    print(f"""
  1_weight_heatmap.png          → What a weight MATRIX looks like
  2_weight_distributions.png    → How weight VALUES are distributed
  3_attention_weights.png       → Q, K, V, O attention matrices
  4_layer_norm_weights.png      → Normalization weights (like biases)
  5_weight_magnitudes.png       → Weight sizes across ALL layers

  REMEMBER:
  ─────────
  • Weights = matrices of numbers (the model's "knowledge")
  • Biases  = Llama uses RMSNorm weights instead of traditional biases
  • Together they transform input text into output predictions
  • Fine-tuning = adjusting these numbers with YOUR data
""")
