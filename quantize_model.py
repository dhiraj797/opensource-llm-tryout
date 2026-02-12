"""
=============================================================
  Model Quantization — Make Models Smaller & Faster
=============================================================

What is Quantization?
---------------------
Quantization reduces the PRECISION of model weights:

  float32 (32 bits per weight)  →  "Full precision" (default)
  float16 (16 bits per weight)  →  Half the size, minimal quality loss
  int8    (8 bits per weight)   →  Quarter the size, small quality loss
  int4    (4 bits per weight)   →  1/8th the size, noticeable quality loss

Think of it like image compression:
  - RAW photo  = float32 (huge, perfect quality)
  - PNG        = float16 (smaller, nearly perfect)
  - JPEG high  = int8    (much smaller, very good)
  - JPEG low   = int4    (tiny, some artifacts)

Why Quantize?
-------------
  - 4x-8x smaller model files (fits on consumer hardware)
  - 2x-4x faster inference
  - Much less GPU/RAM needed
  - Enables running large models on laptops and phones
"""

import torch
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "quantization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "The future of artificial intelligence is"


def load_model_fp32():
    """Load the model in full float32 precision (baseline)."""
    print("=" * 60)
    print("  Loading model in FLOAT32 (full precision)...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def load_model_fp16():
    """Load the model in float16 (half precision)."""
    print("\n" + "=" * 60)
    print("  Loading model in FLOAT16 (half precision)...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def load_model_bf16():
    """Load the model in bfloat16 (brain floating point)."""
    print("\n" + "=" * 60)
    print("  Loading model in BFLOAT16 (brain floating point)...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def load_model_int8():
    """
    Load the model in INT8 using bitsandbytes.
    This is REAL quantization — weights are stored as 8-bit integers.

    Requires: pip install bitsandbytes
    Note: bitsandbytes requires a CUDA GPU. Skipped on CPU-only systems.
    """
    print("\n" + "=" * 60)
    print("  Loading model in INT8 (8-bit quantization)...")
    print("=" * 60)

    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer

    except ImportError:
        print("  [SKIPPED] bitsandbytes not installed.")
        print("  Install with: pip install bitsandbytes")
        return None, None
    except Exception as e:
        print(f"  [SKIPPED] INT8 requires a CUDA GPU. Error: {e}")
        return None, None


def load_model_int4():
    """
    Load the model in INT4 (4-bit quantization) using bitsandbytes.
    This is the most aggressive quantization — 8x smaller!

    Uses NF4 (Normal Float 4) which is optimized for neural network weights.

    Requires: pip install bitsandbytes + CUDA GPU
    """
    print("\n" + "=" * 60)
    print("  Loading model in INT4 (4-bit quantization)...")
    print("=" * 60)

    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        # NF4 = optimized for neural nets
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,    # Double quantization for extra savings
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer

    except ImportError:
        print("  [SKIPPED] bitsandbytes not installed.")
        print("  Install with: pip install bitsandbytes")
        return None, None
    except Exception as e:
        print(f"  [SKIPPED] INT4 requires a CUDA GPU. Error: {e}")
        return None, None


# ──────────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────────

def get_model_size_mb(model):
    """Calculate model size in memory (MB)."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 * 1024)


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text and measure time."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
        )
    elapsed = time.time() - start_time

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]

    return generated, elapsed, tokens_generated


def simulate_quantization_effects(model):
    """
    Demonstrate quantization by manually converting weights to lower precision
    and showing the difference. This works on CPU without special libraries.
    """
    print("\n" + "=" * 60)
    print("  SIMULATING QUANTIZATION EFFECTS (works on CPU)")
    print("=" * 60)

    # Pick a weight tensor to analyze
    for name, param in model.named_parameters():
        if 'layers.0.self_attn.q_proj.weight' in name:
            original = param.detach().cpu().float()
            break

    print(f"\n  Analyzing: layers.0.self_attn.q_proj.weight")
    print(f"  Shape: {list(original.shape)}")
    print(f"  Original dtype: float32 (32 bits per weight)\n")

    results = {}

    # --- Float32 (baseline) ---
    fp32_weights = original.numpy()
    results['float32'] = {
        'bits': 32,
        'weights': fp32_weights,
        'size_factor': 1.0,
    }

    # --- Float16 ---
    fp16_weights = original.half().float().numpy()  # Convert to fp16 and back
    fp16_error = np.abs(fp32_weights - fp16_weights)
    results['float16'] = {
        'bits': 16,
        'weights': fp16_weights,
        'error': fp16_error,
        'size_factor': 0.5,
        'max_error': fp16_error.max(),
        'mean_error': fp16_error.mean(),
    }

    # --- Simulated INT8 ---
    # Scale to [-127, 127] range and round to integers
    scale_8 = original.abs().max() / 127.0
    int8_quantized = torch.round(original / scale_8).clamp(-127, 127)
    int8_dequantized = (int8_quantized * scale_8).numpy()
    int8_error = np.abs(fp32_weights - int8_dequantized)
    results['int8'] = {
        'bits': 8,
        'weights': int8_dequantized,
        'error': int8_error,
        'size_factor': 0.25,
        'max_error': int8_error.max(),
        'mean_error': int8_error.mean(),
    }

    # --- Simulated INT4 ---
    # Scale to [-7, 7] range and round to integers
    scale_4 = original.abs().max() / 7.0
    int4_quantized = torch.round(original / scale_4).clamp(-7, 7)
    int4_dequantized = (int4_quantized * scale_4).numpy()
    int4_error = np.abs(fp32_weights - int4_dequantized)
    results['int4'] = {
        'bits': 4,
        'weights': int4_dequantized,
        'error': int4_error,
        'size_factor': 0.125,
        'max_error': int4_error.max(),
        'mean_error': int4_error.mean(),
    }

    # Print comparison table
    print("  ┌─────────────┬──────┬───────────┬────────────┬────────────┐")
    print("  │ Precision   │ Bits │ Size Mult │ Mean Error │ Max Error  │")
    print("  ├─────────────┼──────┼───────────┼────────────┼────────────┤")
    print("  │ float32     │  32  │   1.00x   │  baseline  │  baseline  │")
    for dtype in ['float16', 'int8', 'int4']:
        r = results[dtype]
        print(f"  │ {dtype:<11s} │  {r['bits']:>2d}  │   "
              f"{r['size_factor']:.2f}x   │ {r['mean_error']:>10.7f} │ {r['max_error']:>10.7f} │")
    print("  └─────────────┴──────┴───────────┴────────────┴────────────┘")

    return results


def visualize_quantization_comparison(results):
    """
    Create visualizations comparing different quantization levels.
    """
    print("\n" + "─" * 50)
    print("  Generating quantization comparison plots...")
    print("─" * 50)

    # --- Plot 1: Weight distributions at different precisions ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dtypes = ['float32', 'float16', 'int8', 'int4']
    colors = ['steelblue', 'green', 'orange', 'red']

    for idx, (dtype, color) in enumerate(zip(dtypes, colors)):
        ax = axes[idx // 2][idx % 2]
        weights = results[dtype]['weights'].flatten()
        bits = results[dtype]['bits']

        ax.hist(weights, bins=200, color=color, alpha=0.8, edgecolor='none')
        ax.set_title(f'{dtype.upper()} ({bits}-bit)\n'
                     f'Size: {results[dtype]["size_factor"]:.0%} of original',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')

        stats = (f'Mean: {weights.mean():.5f}\n'
                 f'Std:  {weights.std():.5f}')
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='monospace')

    fig.suptitle('QUANTIZATION: Weight Distributions at Different Precisions\n'
                 'Notice how INT4 has fewer distinct values (more "binned")',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_quantization_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 2: Quantization error heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    error_dtypes = ['float16', 'int8', 'int4']
    slice_size = 64

    for idx, dtype in enumerate(error_dtypes):
        error = results[dtype]['error'][:slice_size, :slice_size]
        im = axes[idx].imshow(error, cmap='hot', aspect='equal', interpolation='nearest')
        axes[idx].set_title(f'{dtype.upper()} Quantization Error\n'
                            f'Mean: {results[dtype]["mean_error"]:.7f}',
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Column')
        axes[idx].set_ylabel('Row')
        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    fig.suptitle('QUANTIZATION ERROR: Difference from Original float32 Weights\n'
                 'Brighter = more error (information lost during compression)',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_quantization_errors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 3: Side-by-side weight heatmaps ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    slice_size = 64

    for idx, (dtype, color) in enumerate(zip(dtypes, colors)):
        weights = results[dtype]['weights'][:slice_size, :slice_size]
        im = axes[idx].imshow(weights, cmap='RdBu_r', aspect='equal',
                              interpolation='nearest',
                              vmin=results['float32']['weights'][:slice_size, :slice_size].min(),
                              vmax=results['float32']['weights'][:slice_size, :slice_size].max())
        axes[idx].set_title(f'{dtype.upper()} ({results[dtype]["bits"]}-bit)',
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Column')
        axes[idx].set_ylabel('Row')
        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    fig.suptitle('WEIGHT MATRICES: Same Layer at Different Precisions\n'
                 'INT4 shows visible "blockiness" from reduced precision',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_weight_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 4: Size and Error summary bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Model size comparison
    sizes = [results[d]['size_factor'] * 100 for d in dtypes]
    bars1 = ax1.bar(dtypes, sizes, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Model Size (% of original)', fontsize=12)
    ax1.set_title('Model Size Reduction', fontsize=13, fontweight='bold')
    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{size:.0f}%', ha='center', va='bottom', fontweight='bold')

    # Error comparison
    errors = [0] + [results[d]['mean_error'] for d in ['float16', 'int8', 'int4']]
    bars2 = ax2.bar(dtypes, errors, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Quantization Error (vs float32)', fontsize=13, fontweight='bold')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    fig.suptitle('QUANTIZATION TRADE-OFF: Size vs Accuracy',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_size_vs_error.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def compare_generation_quality(model_fp32, tokenizer):
    """
    Show how quantization affects actual text generation quality.
    We simulate by adding quantization noise to demonstrate the effect.
    """
    print("\n" + "=" * 60)
    print("  TEXT GENERATION — Comparing Quality")
    print("=" * 60)

    print(f'\n  Prompt: "{PROMPT}"\n')

    # Generate with fp32
    text, elapsed, n_tokens = generate_text(model_fp32, tokenizer, PROMPT)
    tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0

    print(f"  FLOAT32 (full precision):")
    print(f"  ─────────────────────────")
    print(f"  Output: {text}")
    print(f"  Time: {elapsed:.2f}s | Tokens: {n_tokens} | Speed: {tokens_per_sec:.1f} tok/s")

    print(f"\n  NOTE: For INT8 and INT4 generation, you need:")
    print(f"  → A CUDA GPU (NVIDIA)")
    print(f"  → pip install bitsandbytes")
    print(f"  → The load_model_int8() and load_model_int4() functions above")
    print(f"\n  On a GPU, you'd typically see:")
    print(f"  ┌─────────────┬───────────┬──────────────────────────────────┐")
    print(f"  │ Precision   │ Speed     │ Quality                          │")
    print(f"  ├─────────────┼───────────┼──────────────────────────────────┤")
    print(f"  │ float32     │ 1x (base) │ Perfect (baseline)               │")
    print(f"  │ float16     │ ~2x       │ Nearly identical to float32      │")
    print(f"  │ int8        │ ~2-3x     │ Very close, minor differences    │")
    print(f"  │ int4        │ ~3-4x     │ Good but some quality loss       │")
    print(f"  └─────────────┴───────────┴──────────────────────────────────┘")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MODEL QUANTIZATION EXPLORER")
    print("  See how reducing precision makes models smaller & faster")
    print("=" * 60 + "\n")

    # Step 1: Load the full precision model
    model_fp32, tokenizer = load_model_fp32()
    fp32_size = get_model_size_mb(model_fp32)
    print(f"\n  Model size in memory (float32): {fp32_size:.1f} MB")
    print(f"  Total parameters: {sum(p.numel() for p in model_fp32.parameters()):,}")

    # Step 2: Simulate quantization effects on weights
    results = simulate_quantization_effects(model_fp32)

    # Step 3: Generate visualizations
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS...")
    print("=" * 60)
    visualize_quantization_comparison(results)

    # Step 4: Compare generation quality
    compare_generation_quality(model_fp32, tokenizer)

    # Step 5: Try loading in different precisions
    print("\n" + "=" * 60)
    print("  LOADING MODEL IN DIFFERENT PRECISIONS")
    print("=" * 60)

    try:
        model_fp16, _ = load_model_fp16()
        fp16_size = get_model_size_mb(model_fp16)
        print(f"  Model size in memory (float16): {fp16_size:.1f} MB")
        print(f"  Savings: {(1 - fp16_size / fp32_size) * 100:.1f}% smaller!")
        del model_fp16
    except Exception as e:
        print(f"  [SKIPPED] float16: {e}")

    try:
        model_bf16, _ = load_model_bf16()
        bf16_size = get_model_size_mb(model_bf16)
        print(f"  Model size in memory (bfloat16): {bf16_size:.1f} MB")
        print(f"  Savings: {(1 - bf16_size / fp32_size) * 100:.1f}% smaller!")
        del model_bf16
    except Exception as e:
        print(f"  [SKIPPED] bfloat16: {e}")

    # INT8 and INT4 need a GPU
    model_int8, _ = load_model_int8()
    if model_int8:
        int8_size = get_model_size_mb(model_int8)
        print(f"  Model size in memory (int8): {int8_size:.1f} MB")
        print(f"  Savings: {(1 - int8_size / fp32_size) * 100:.1f}% smaller!")
        del model_int8

    model_int4, _ = load_model_int4()
    if model_int4:
        int4_size = get_model_size_mb(model_int4)
        print(f"  Model size in memory (int4): {int4_size:.1f} MB")
        print(f"  Savings: {(1 - int4_size / fp32_size) * 100:.1f}% smaller!")
        del model_int4

    # Final summary
    print("\n" + "=" * 60)
    print(f"  ALL DONE! Check the '{OUTPUT_DIR}/' folder for plots:")
    print("=" * 60)
    print(f"""
  1_quantization_distributions.png  → Weight distributions at each precision
  2_quantization_errors.png         → Error heatmaps (what's lost)
  3_weight_comparison.png           → Side-by-side weight matrices
  4_size_vs_error.png               → Size vs accuracy trade-off

  KEY TAKEAWAYS:
  ──────────────
  • float32 → float16:  50% smaller, almost no quality loss
  • float32 → int8:     75% smaller, very small quality loss
  • float32 → int4:     87% smaller, some quality loss but still usable
  • INT8/INT4 need bitsandbytes + CUDA GPU for real quantization
  • float16/bfloat16 work on CPU too!

  TO QUANTIZE FOR REAL (on a GPU):
  ────────────────────────────────
  pip install bitsandbytes accelerate
  # Then use load_model_int8() or load_model_int4() functions
""")
