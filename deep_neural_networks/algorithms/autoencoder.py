"""
🔄 AUTOENCODERS — Compress, Reconstruct, Denoise, Detect Anomalies

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What an autoencoder is and why the input = the target (unsupervised!)
2. The encoder-bottleneck-decoder architecture and why the bottleneck matters
3. How to build an autoencoder from scratch (NumPy) and with Keras
4. How a denoising autoencoder learns signal vs noise
5. How autoencoders detect anomalies without any labeled examples
6. What the latent space looks like — and why similar things cluster there
7. How autoencoders connect unsupervised learning (Part 3) to deep learning (Part 4)

================================================================================
📺 RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
⭐ ABSOLUTE MUST WATCH:
   - Andrej Karpathy: "Autoencoders" (from CS231n Stanford)
     https://www.youtube.com/watch?v=nTt_ajul7Io
   - StatQuest: "Autoencoders"
     https://www.youtube.com/watch?v=qiUEgSCyY5o

Also Recommended:
   - 3Blue1Brown: "Neural networks" playlist (for architecture intuition)
     https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   - Lex Fridman: "Autoencoders and Variational Autoencoders"
     https://www.youtube.com/watch?v=9zKuYvjFFS8

================================================================================
OVERVIEW
================================================================================
An autoencoder is a neural network trained to copy its input to its output —
but with a twist: it must pass through a narrow BOTTLENECK.

That bottleneck forces the network to learn a compressed representation
of the data. It has to figure out what's essential and what's noise.

This makes autoencoders useful for:
  - Compression & reconstruction (images, audio, text)
  - Denoising (clean up blurry/noisy images)
  - Anomaly detection (unusual = high reconstruction error)
  - Feature learning (bottleneck = compact features for downstream ML)
  - Generative models (VAE — Variational Autoencoders)

The key insight: it's UNSUPERVISED.
You need NO labels. The data teaches itself.

================================================================================
CONNECTIONS TO EARLIER PARTS
================================================================================
  Part 3 (Unsupervised): PCA also compresses to a bottleneck!
    PCA:         Linear compression (eigenvectors)
    Autoencoder: Non-linear compression (learned neural network)
    Autoencoder >> PCA for complex data like images

  Part 3 (Anomaly Detection): Isolation Forest, LOF — statistical methods
    Autoencoder-based anomaly detection: train on normal → abnormal = high error
    Works even when anomalies have complex structure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "autoencoder")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("🔄 AUTOENCODERS")
print("   Compress → Reconstruct → Denoise → Detect Anomalies")
print("=" * 70)
print()

# ============================================================================
# SECTION 1: THE CONCEPT — ARCHITECTURE DIAGRAM
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: What Is an Autoencoder?")
print("=" * 70)
print()
print("CORE IDEA:")
print("-" * 60)
print("  Normal neural network:  Input → Layers → Prediction")
print("  Autoencoder:            Input → COMPRESS → RECONSTRUCT → Input")
print()
print("  The network learns to encode the ESSENCE of the data.")
print("  Then decode it back to the original.")
print()
print("  Input = Target (no labels needed — fully unsupervised)")
print()
print("ARCHITECTURE:")
print("-" * 60)
print()
print("  Original     Encoder          Bottleneck    Decoder       Reconstructed")
print("  (784 pixels) (compress)       (32 values)   (expand)      (784 pixels)")
print()
print("    ┌───┐    ┌────────────┐    ┌─────────┐  ┌────────────┐    ┌───┐")
print("    │ X │───▶│ 784→256    │───▶│   32    │─▶│  32→256    │───▶│ X̂ │")
print("    │   │    │ 256→128    │    │(latent  │  │  256→784   │    │   │")
print("    └───┘    │ 128→32     │    │ space)  │  │            │    └───┘")
print("             └────────────┘    └─────────┘  └────────────┘")
print("             (information        (forced         (reconstruct")
print("              compressed)        bottleneck)      from code)")
print()
print("  BOTTLENECK = the compressed representation = LATENT CODE")
print("  32 numbers describe an entire 784-pixel image!")
print()
print("TRAINING SIGNAL:")
print("-" * 60)
print("  Loss = MSE(X, X̂)  = mean((original - reconstructed)²)")
print()
print("  If reconstruction is good: loss is LOW → weights preserved")
print("  If reconstruction is poor: loss is HIGH → weights updated")
print()
print("  The network has ONE job: reconstruct the input.")
print("  To do that through 32 numbers, it MUST learn what matters.")
print()

# ============================================================================
# VISUALIZATION 1: CONCEPTUAL ARCHITECTURE DIAGRAM (not a graph — a diagram!)
# ============================================================================
print("Generating Visualization 1: Conceptual Architecture Diagram...")

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#0f0f1a')

# --- Panel 1: Architecture block diagram ---
ax = axes[0]
ax.set_facecolor('#0f0f1a')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Autoencoder Architecture', color='white', fontsize=14, fontweight='bold', pad=15)

def draw_layer_block(ax, x, y, width, height, n_units, label, color, text_color='white'):
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.1", facecolor=color, edgecolor='white',
                           linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x, y + 0.15, label, ha='center', va='center', color=text_color,
            fontsize=9, fontweight='bold')
    ax.text(x, y - 0.35, f'({n_units})', ha='center', va='center', color=text_color,
            fontsize=8, alpha=0.9)

def draw_arrow(ax, x1, y1, x2, y2, color='#aaaaaa'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Input
draw_layer_block(ax, 1.0, 5.0, 1.4, 4.5, 784, 'INPUT\nImage', '#2c5f8a')
# Encoder layers
draw_layer_block(ax, 2.8, 5.0, 1.2, 3.5, 256, 'Dense\n256', '#1e7a4e')
draw_layer_block(ax, 4.2, 5.0, 1.0, 2.5, 128, 'Dense\n128', '#1e7a4e')
# Bottleneck
draw_layer_block(ax, 5.5, 5.0, 1.0, 1.2, 32,  'Bottleneck\n32', '#8b2252', text_color='white')
# Decoder layers
draw_layer_block(ax, 6.8, 5.0, 1.0, 2.5, 128, 'Dense\n128', '#7a5a1e')
draw_layer_block(ax, 8.2, 5.0, 1.2, 3.5, 256, 'Dense\n256', '#7a5a1e')
# Output
draw_layer_block(ax, 9.8, 5.0, 1.0, 4.5, 784, 'OUTPUT\nReconst.', '#2c5f8a')

# Arrows between layers
for x1, x2 in [(1.7, 2.2), (3.4, 3.7), (4.7, 5.0), (6.0, 6.3), (7.3, 7.7), (8.8, 9.3)]:
    draw_arrow(ax, x1, 5.0, x2, 5.0)

# Brace labels
ax.annotate('', xy=(5.0, 8.8), xytext=(1.0, 8.8),
            arrowprops=dict(arrowstyle='<->', color='#1e7a4e', lw=2))
ax.text(3.0, 9.1, 'ENCODER', ha='center', color='#4ec990', fontsize=11, fontweight='bold')

ax.annotate('', xy=(9.8, 8.8), xytext=(5.8, 8.8),
            arrowprops=dict(arrowstyle='<->', color='#7a5a1e', lw=2))
ax.text(7.8, 9.1, 'DECODER', ha='center', color='#d4a843', fontsize=11, fontweight='bold')

ax.annotate('', xy=(6.0, 1.2), xytext=(5.0, 1.2),
            arrowprops=dict(arrowstyle='<->', color='#cc3377', lw=2))
ax.text(5.5, 0.7, 'LATENT\nSPACE', ha='center', color='#ee66aa', fontsize=9, fontweight='bold')

# Loss annotation
ax.annotate('Loss = MSE(Input, Output)\n= how well we reconstructed',
            xy=(5.5, 1.8), fontsize=9, ha='center', color='#ffdd88',
            bbox=dict(boxstyle='round', facecolor='#333322', alpha=0.9))

ax.text(5.0, 9.7, 'Input feeds directly into Loss — NO LABELS NEEDED',
        ha='center', color='#aaaaaa', fontsize=9, style='italic')

# --- Panel 2: What the bottleneck forces the model to learn ---
ax2 = axes[1]
ax2.set_facecolor('#0f0f1a')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Why the Bottleneck Matters', color='white', fontsize=14, fontweight='bold', pad=15)

# Draw funnel shape
from matplotlib.patches import Polygon
funnel_top = Polygon([[1,9],[5,9],[5,5.5],[3,4.5],[3,4.5],[1,5.5]], closed=True,
                      facecolor='#1a3a5c', edgecolor='#4488cc', lw=2, alpha=0.7)
funnel_bot = Polygon([[5,4.5],[5.5,4.5],[5.5,4.5],[5.5,4.5],[5,5.0],[5,4.5]], closed=True,
                      facecolor='#8b2252', edgecolor='#cc4488', lw=2, alpha=0.9)
ax2.add_patch(funnel_top)

# Concepts going in (what the encoder compresses)
concepts_in = [
    (2.5, 8.5, '🔲 Edges & Corners', '#88bbff'),
    (2.5, 7.7, '📐 Shapes & Curves', '#88ffbb'),
    (2.5, 6.9, '🎨 Textures', '#ffbb88'),
    (2.5, 6.1, '📍 Positions', '#ff88bb'),
    (2.5, 5.3, '📏 Proportions', '#bbff88'),
]
for x, y, text, color in concepts_in:
    ax2.text(x, y, text, ha='center', va='center', color=color, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

# Bottleneck label
ax2.text(5.0, 4.75, '◉\n32\nnumbers', ha='center', va='center', color='white',
         fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='circle', facecolor='#8b2252', edgecolor='#ff44aa', lw=2))
ax2.text(5.0, 3.7, 'LATENT CODE', ha='center', color='#ee66aa', fontsize=10, fontweight='bold')
ax2.text(5.0, 3.2, '"The essence of the image"', ha='center', color='#aaaaaa', fontsize=8, style='italic')

# Comparison: PCA vs Autoencoder
ax2.text(5.0, 2.3, 'COMPARISON: PCA vs Autoencoder', ha='center', color='white',
         fontsize=10, fontweight='bold')
rows = [
    ('',          'PCA (Part 3)',    'Autoencoder'),
    ('Compression','Linear rotation','Non-linear network'),
    ('Learns',    'Eigenvectors',   'Arbitrary features'),
    ('Best for',  'Simple structure','Complex (images)'),
    ('Speed',     'Very fast',      'Needs training'),
]
for i, (label, pca, ae) in enumerate(rows):
    y_row = 1.9 - i * 0.35
    color = '#cccccc' if i > 0 else '#88aaff'
    ax2.text(1.5, y_row, label,  ha='left', color=color, fontsize=8)
    ax2.text(5.0, y_row, pca,    ha='center', color='#88bbff', fontsize=8)
    ax2.text(8.5, y_row, ae,     ha='center', color='#88ffcc', fontsize=8)

ax2.axhline(1.75, color='#444466', lw=1, xmin=0.1, xmax=0.9)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VIS_DIR, '01_architecture_concept.png'), dpi=300,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("  ✅ Saved: 01_architecture_concept.png")
print()

# ============================================================================
# SECTION 2: AUTOENCODER FROM SCRATCH (NUMPY)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Autoencoder from Scratch (NumPy)")
print("=" * 70)
print()
print("Building a small autoencoder manually to see every weight and gradient.")
print("We'll use a toy 2D→1D→2D bottleneck on synthetic data.")
print()

# Toy data: 2D points forming a diagonal line (a 1D manifold)
from sklearn.datasets import make_blobs
n_samples = 300
t = np.linspace(0, 2*np.pi, n_samples)
X_toy = np.column_stack([np.cos(t) + 0.1*np.random.randn(n_samples),
                          np.sin(t) + 0.1*np.random.randn(n_samples)])

print(f"  Toy data: {X_toy.shape}  (points on a noisy circle)")
print(f"  Bottleneck: 2D → 1D → 2D")
print()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

class SimpleAutoencoder:
    """Minimal autoencoder: 2D → 1D bottleneck → 2D (all sigmoid activations)"""
    def __init__(self, input_dim=2, latent_dim=1, lr=0.01):
        self.lr = lr
        # Encoder weights: input → latent
        self.W_enc = np.random.randn(input_dim, latent_dim) * 0.1
        self.b_enc = np.zeros(latent_dim)
        # Decoder weights: latent → output
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)

    def encode(self, X):
        return np.tanh(X @ self.W_enc + self.b_enc)

    def decode(self, Z):
        return Z @ self.W_dec + self.b_dec   # linear output layer

    def forward(self, X):
        Z = self.encode(X)
        X_hat = self.decode(Z)
        return Z, X_hat

    def train_step(self, X):
        Z, X_hat = self.forward(X)
        # MSE loss
        loss = np.mean((X - X_hat) ** 2)
        # Backprop
        d_out = -2 * (X - X_hat) / len(X)              # dL/dX_hat
        self.W_dec -= self.lr * Z.T @ d_out             # dL/dW_dec
        self.b_dec -= self.lr * d_out.mean(axis=0)
        d_Z = d_out @ self.W_dec.T * (1 - Z**2)        # tanh derivative
        self.W_enc -= self.lr * X.T @ d_Z
        self.b_enc -= self.lr * d_Z.mean(axis=0)
        return loss

ae_scratch = SimpleAutoencoder(input_dim=2, latent_dim=1, lr=0.05)

losses = []
epochs = 500
print(f"  Training {epochs} epochs...")
for epoch in range(epochs):
    loss = ae_scratch.train_step(X_toy)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"    Epoch {epoch:4d} | Loss: {loss:.6f}")

Z_toy = ae_scratch.encode(X_toy)
X_hat_toy = ae_scratch.decode(Z_toy)
final_loss = np.mean((X_toy - X_hat_toy)**2)

print()
print(f"  Final reconstruction loss: {final_loss:.6f}")
print(f"  Input shape:  {X_toy.shape}   (2D)")
print(f"  Latent shape: {Z_toy.shape}   (1D — compressed!)")
print(f"  Output shape: {X_hat_toy.shape}   (2D reconstructed)")
print()
print("  Key insight: we stored a circle in ONE number per point!")
print("  That 1 number = the 'angle' on the circle (the intrinsic dimensionality)")
print()

# ============================================================================
# SECTION 3: KERAS AUTOENCODER ON MNIST
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Autoencoder with Keras on MNIST")
print("=" * 70)
print()

KERAS_AE = None
X_test_sample = None
X_hat_sample = None
X_noisy_sample = None
X_hat_denoised = None
Z_train_2d = None
y_train_colors = None
training_history = None
anomaly_errors = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    tf.random.set_seed(42)

    print(f"  TensorFlow {tf.__version__} loaded ✅")
    print()

    # Load MNIST
    (X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train_raw.astype('float32') / 255.0
    X_test  = X_test_raw.astype('float32')  / 255.0
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat  = X_test.reshape(-1, 784)

    print(f"  MNIST loaded: {X_train_flat.shape} train, {X_test_flat.shape} test")
    print(f"  Each image: 28×28 = 784 pixels, normalized to [0, 1]")
    print()

    # Build Autoencoder
    print("  Building autoencoder: 784 → 256 → 128 → 32 → 128 → 256 → 784")
    print()

    # Encoder
    enc_input = keras.Input(shape=(784,), name='encoder_input')
    x = layers.Dense(256, activation='relu', name='enc_dense1')(enc_input)
    x = layers.Dense(128, activation='relu', name='enc_dense2')(x)
    bottleneck = layers.Dense(32,  activation='relu', name='bottleneck')(x)
    encoder = Model(enc_input, bottleneck, name='encoder')

    # Decoder
    dec_input = keras.Input(shape=(32,), name='decoder_input')
    x = layers.Dense(128, activation='relu', name='dec_dense1')(dec_input)
    x = layers.Dense(256, activation='relu', name='dec_dense2')(x)
    dec_output = layers.Dense(784, activation='sigmoid', name='decoder_output')(x)
    decoder = Model(dec_input, dec_output, name='decoder')

    # Full autoencoder
    ae_input  = keras.Input(shape=(784,))
    ae_output = decoder(encoder(ae_input))
    autoencoder = Model(ae_input, ae_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    total_params = autoencoder.count_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder.count_params():,}")
    print(f"  Decoder parameters: {decoder.count_params():,}")
    print()

    # Train
    print("  Training autoencoder (input = target, no labels!)...")
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=0)]
    history = autoencoder.fit(
        X_train_flat, X_train_flat,    # <-- input IS the target
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        callbacks=cb,
        verbose=1
    )
    training_history = history.history

    # Reconstruct test images
    X_hat_test = autoencoder.predict(X_test_flat, verbose=0)
    test_mse = np.mean((X_test_flat - X_hat_test)**2)
    print(f"\n  Test reconstruction MSE: {test_mse:.6f}")
    print(f"  Average pixel error:     {np.sqrt(test_mse)*255:.2f} / 255")
    print()

    X_test_sample = X_test_flat[:10]
    X_hat_sample  = X_hat_test[:10]
    KERAS_AE = autoencoder

    # ============================================================================
    # SECTION 4: DENOISING AUTOENCODER
    # ============================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Denoising Autoencoder")
    print("=" * 70)
    print()
    print("  Train with noisy input → clean output as target")
    print("  Teaches the model to separate SIGNAL from NOISE")
    print()

    noise_factor = 0.3
    X_noisy_train = np.clip(X_train_flat + noise_factor * np.random.randn(*X_train_flat.shape), 0, 1)
    X_noisy_test  = np.clip(X_test_flat  + noise_factor * np.random.randn(*X_test_flat.shape),  0, 1)

    # Denoising AE (same architecture, different training target)
    dae_input  = keras.Input(shape=(784,))
    dae_output = decoder(encoder(dae_input))
    denoising_ae = Model(dae_input, dae_output, name='denoising_ae')
    denoising_ae.compile(optimizer='adam', loss='mse')

    print("  Training denoising autoencoder (noisy input → clean target)...")
    denoising_ae.fit(
        X_noisy_train, X_train_flat,   # noisy in, clean out
        epochs=10, batch_size=256,
        validation_split=0.1,
        callbacks=cb, verbose=1
    )

    X_hat_denoised = denoising_ae.predict(X_noisy_test[:10], verbose=0)
    X_noisy_sample = X_noisy_test[:10]

    denoising_mse = np.mean((X_test_flat[:10] - X_hat_denoised)**2)
    noisy_mse     = np.mean((X_test_flat[:10] - X_noisy_test[:10])**2)
    print(f"\n  Noisy vs clean MSE (before denoising):  {noisy_mse:.6f}")
    print(f"  Denoised vs clean MSE (after denoising): {denoising_mse:.6f}")
    print(f"  Improvement factor: {noisy_mse/denoising_mse:.1f}×  ✅")

    # ============================================================================
    # SECTION 5: ANOMALY DETECTION WITH AUTOENCODERS
    # ============================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Anomaly Detection with Autoencoders")
    print("=" * 70)
    print()
    print("  Strategy: train ONLY on digit '0'")
    print("  Test on all digits — high reconstruction error = anomaly")
    print()

    mask_0 = y_train == 0
    X_zeros_train = X_train_flat[mask_0]
    print(f"  Training on {len(X_zeros_train):,} examples of digit '0' only")

    ae_anomaly_input  = keras.Input(shape=(784,))
    enc_anom = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(32,  activation='relu'),
    ], name='enc_anom')
    dec_anom = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(784, activation='sigmoid'),
    ], name='dec_anom')
    ae_anomaly_out = dec_anom(enc_anom(ae_anomaly_input))
    ae_anomaly = Model(ae_anomaly_input, ae_anomaly_out)
    ae_anomaly.compile(optimizer='adam', loss='mse')

    ae_anomaly.fit(X_zeros_train, X_zeros_train,
                   epochs=10, batch_size=128,
                   validation_split=0.1, callbacks=cb, verbose=1)

    print("\n  Testing reconstruction error per digit class:")
    print("  (Low error = 'looks like 0'; High error = anomaly)")
    print()

    anomaly_errors = {}
    for digit in range(10):
        mask = y_test == digit
        X_digit = X_test_flat[mask[:1000]][:200]
        X_hat_d = ae_anomaly.predict(X_digit, verbose=0)
        mse = np.mean((X_digit - X_hat_d)**2)
        anomaly_errors[digit] = mse
        marker = " ← trained on this" if digit == 0 else (" ← ANOMALY" if mse > anomaly_errors[0] * 2 else "")
        print(f"    Digit {digit}: MSE = {mse:.6f}{marker}")

    print()
    print("  Digits that look unlike '0' have much higher reconstruction error.")
    print("  No labels needed — unsupervised anomaly detection!")

    # ============================================================================
    # SECTION 6: 2D LATENT SPACE VISUALIZATION
    # ============================================================================
    print("\n" + "=" * 70)
    print("SECTION 6: The Latent Space — What the Bottleneck Learns")
    print("=" * 70)
    print()
    print("  Build a 2D bottleneck to visualize the latent space directly")
    print()

    enc2_input = keras.Input(shape=(784,))
    enc2_x     = layers.Dense(128, activation='relu')(enc2_input)
    enc2_out   = layers.Dense(2,   activation='linear', name='latent_2d')(enc2_x)
    dec2_input = keras.Input(shape=(2,))
    dec2_x     = layers.Dense(128, activation='relu')(dec2_input)
    dec2_out   = layers.Dense(784, activation='sigmoid')(dec2_x)
    encoder2   = Model(enc2_input, enc2_out)
    decoder2   = Model(dec2_input, dec2_out)
    ae2_out    = decoder2(encoder2(enc2_input))
    ae2        = Model(enc2_input, ae2_out)
    ae2.compile(optimizer='adam', loss='mse')

    print("  Training 2D bottleneck autoencoder...")
    ae2.fit(X_train_flat, X_train_flat,
            epochs=15, batch_size=256,
            validation_split=0.1, callbacks=cb, verbose=1)

    # Encode a sample to 2D
    n_vis = 3000
    Z_train_2d = encoder2.predict(X_train_flat[:n_vis], verbose=0)
    y_train_colors = y_train[:n_vis]

    print(f"\n  2D latent space computed for {n_vis:,} digits")
    print(f"  Latent shape: {Z_train_2d.shape}")
    print()
    print("  Without ANY labels during training, the autoencoder")
    print("  naturally separates digit classes in latent space!")
    print()

except ImportError:
    print()
    print("  ⚠️  TensorFlow not installed — cannot run Keras sections.")
    print("  Install: pip install tensorflow")
    print()
    print("  The code above shows the exact patterns to follow.")
    print("  Once installed: python3 autoencoder.py  (everything runs)")
    print()

# ============================================================================
# VISUALIZATION 2: RECONSTRUCTIONS (ORIGINAL / NOISY / DENOISED)
# ============================================================================
print("\n" + "Generating Visualization 2: Reconstructions Grid...")

fig, axes = plt.subplots(3, 10, figsize=(20, 7))
fig.patch.set_facecolor('#0f0f1a')
fig.suptitle('Autoencoder Reconstructions: Original  ·  Noisy  ·  Denoised',
             color='white', fontsize=14, fontweight='bold', y=1.01)

row_labels = ['Original\n(input)', 'Noisy\n(+noise)', 'Denoised\n(output)']
row_colors = ['#88ccff', '#ff8888', '#88ffbb']

for col in range(10):
    for row, (label, color) in enumerate(zip(row_labels, row_colors)):
        ax = axes[row, col]
        ax.set_facecolor('#0f0f1a')

        if row == 0 and X_test_sample is not None:
            img = X_test_sample[col].reshape(28, 28)
        elif row == 1 and X_noisy_sample is not None:
            img = X_noisy_sample[col].reshape(28, 28)
        elif row == 2 and X_hat_denoised is not None:
            img = X_hat_denoised[col].reshape(28, 28)
        else:
            img = np.random.rand(28, 28) * 0.3 + 0.1

        ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
        ax.axis('off')

        if col == 0:
            ax.set_ylabel(label, color=color, fontsize=10, fontweight='bold',
                         rotation=0, labelpad=60, va='center')

for col in range(10):
    axes[0, col].set_title(f'#{col+1}', color='#aaaaaa', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '02_reconstructions.png'), dpi=300,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("  ✅ Saved: 02_reconstructions.png")

# ============================================================================
# VISUALIZATION 3: LATENT SPACE + ANOMALY DETECTION SCORE
# ============================================================================
print("Generating Visualization 3: Latent Space and Anomaly Scores...")

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.patch.set_facecolor('#0f0f1a')

# Panel 1: Toy autoencoder training loss
ax = axes[0]
ax.set_facecolor('#111122')
ax.plot(losses, color='#4ec9b0', lw=2.5, label='Reconstruction loss')
ax.fill_between(range(len(losses)), losses, alpha=0.2, color='#4ec9b0')
ax.set_xlabel('Epoch', color='white', fontsize=11)
ax.set_ylabel('MSE Loss', color='white', fontsize=11)
ax.set_title('From-Scratch Autoencoder\nTraining Loss (2D→1D→2D)', color='white',
             fontsize=12, fontweight='bold')
ax.tick_params(colors='white')
ax.spines[:].set_color('#444466')
ax.legend(facecolor='#222233', labelcolor='white', fontsize=10)
ax.text(len(losses)*0.6, max(losses)*0.7,
        f'Final loss:\n{losses[-1]:.5f}', color='#88ffcc',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

# Panel 2: 2D latent space colored by digit class
ax2 = axes[1]
ax2.set_facecolor('#111122')
if Z_train_2d is not None:
    cmap = plt.cm.get_cmap('tab10', 10)
    for digit in range(10):
        mask = y_train_colors == digit
        ax2.scatter(Z_train_2d[mask, 0], Z_train_2d[mask, 1],
                   c=[cmap(digit)], s=8, alpha=0.6, label=str(digit))
    ax2.legend(title='Digit', title_fontsize=9, fontsize=8,
               facecolor='#222233', labelcolor='white',
               ncol=2, loc='upper right',
               bbox_to_anchor=(1.0, 1.0))
    ax2.set_title('2D Latent Space\n(No labels used in training!)', color='white',
                 fontsize=12, fontweight='bold')
    ax2.text(0.05, 0.05, 'Similar digits\ncluster together\nautomatically',
             transform=ax2.transAxes, color='#ffdd88', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#222211', alpha=0.85))
else:
    # Placeholder when TF not available
    ax2.text(0.5, 0.5, 'Install TensorFlow\nto see 2D latent space\n\npip install tensorflow',
             ha='center', va='center', transform=ax2.transAxes,
             color='#aaaaaa', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#222233', alpha=0.8))
    ax2.set_title('2D Latent Space\n(TensorFlow required)', color='white',
                 fontsize=12, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#444466')
ax2.set_xlabel('Latent dim 1', color='white')
ax2.set_ylabel('Latent dim 2', color='white')

# Panel 3: Anomaly detection scores per digit
ax3 = axes[2]
ax3.set_facecolor('#111122')
if anomaly_errors is not None:
    digits = list(anomaly_errors.keys())
    errors = list(anomaly_errors.values())
    bar_colors = ['#4ec9b0' if d == 0 else '#ff6666' for d in digits]
    bars = ax3.bar(digits, errors, color=bar_colors, edgecolor='white',
                   linewidth=0.8, alpha=0.85)
    ax3.axhline(y=anomaly_errors[0]*2, color='#ffdd44', lw=2, ls='--',
               label='Anomaly threshold (2× baseline)')
    ax3.set_title('Anomaly Detection\n(Trained only on digit "0")', color='white',
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Digit class', color='white')
    ax3.set_ylabel('Reconstruction MSE', color='white')
    ax3.legend(facecolor='#222233', labelcolor='white', fontsize=9)
    ax3.text(0, anomaly_errors[0]*1.1, '← Normal\n   (low error)',
             color='#4ec9b0', fontsize=9, ha='center')
    for bar, d, e in zip(bars, digits, errors):
        if d != 0 and e > anomaly_errors[0] * 2:
            ax3.text(bar.get_x() + bar.get_width()/2, e + 0.0001, '⚠',
                    ha='center', va='bottom', color='#ffdd44', fontsize=12)
else:
    ax3.text(0.5, 0.5, 'Install TensorFlow\nto see anomaly detection\n\npip install tensorflow',
             ha='center', va='center', transform=ax3.transAxes,
             color='#aaaaaa', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#222233', alpha=0.8))
    ax3.set_title('Anomaly Detection\n(TensorFlow required)', color='white',
                 fontsize=12, fontweight='bold')
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#444466')
ax3.set_xticks(range(10))

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VIS_DIR, '03_latent_space_and_anomaly.png'), dpi=300,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("  ✅ Saved: 03_latent_space_and_anomaly.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Autoencoders")
print("=" * 70)
print()
print("WHAT YOU BUILT:")
print("-" * 60)
print("  1. From-scratch autoencoder (NumPy)")
print("     2D circle → 1D latent → 2D reconstruction")
print()
print("  2. Standard autoencoder (Keras, MNIST)")
print("     784 → 32 → 784, MSE loss, no labels needed")
print()
print("  3. Denoising autoencoder")
print("     Noisy input → clean output (removes noise)")
print()
print("  4. Anomaly detection")
print("     Train on class 0 only → high error = anomaly")
print()
print("  5. 2D latent space")
print("     32D → 2D → see digit clusters form automatically")
print()
print("KEY INSIGHT:")
print("-" * 60)
print("  The autoencoder is the bridge between unsupervised learning")
print("  (Part 3) and deep learning (Part 4).")
print()
print("  PCA:         Linear compression    (eigenvectors)")
print("  Autoencoder: Non-linear compression (neural network)")
print()
print("  Both find compressed representations without labels.")
print("  Autoencoders learn more complex structure.")
print()
print("WHAT'S NEXT:")
print("-" * 60)
print("  Variational Autoencoders (VAEs): not just compress,")
print("  but GENERATE new examples from the latent space.")
print("  The conceptual ancestor of modern image generation.")
print("  (Explored in Part 8: LLMs section)")
print()
print("=" * 70)
print("Visualizations saved to:", VIS_DIR)
print("  01_architecture_concept.png — architecture + bottleneck diagram")
print("  02_reconstructions.png      — original / noisy / denoised grid")
print("  03_latent_space_and_anomaly.png — latent space + anomaly scores")
print("=" * 70)
