"""
🔁 NLP — Math Foundation 4: RNN Intuition
==========================================

Learning Objectives:
  1. Understand WHY sequence order matters (BoW fails for context)
  2. Grasp the RNN computational graph: hidden state as "memory"
  3. Implement a Vanilla RNN forward pass from scratch in numpy
  4. Understand the vanishing gradient problem in deep/long sequences
  5. Understand how LSTM gates solve the vanishing gradient problem
  6. Compare RNN, LSTM, GRU architectures and when to use each
  7. Visualize hidden state dynamics across timesteps

YouTube Resources:
  ⭐ Andrej Karpathy — The Unreasonable Effectiveness of RNNs
     https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  ⭐ StatQuest — RNNs clearly explained https://www.youtube.com/watch?v=AsNTP8Kwu80
  ⭐ StatQuest — LSTMs clearly explained https://www.youtube.com/watch?v=YCzL96nL7j0

Time Estimate: 60 min
Difficulty: Intermediate
Prerequisites: 03_word_embeddings.py, DNNs Part 3 (backpropagation)
Key Concepts: hidden state, vanishing gradient, LSTM gates (forget, input, output), GRU
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

_VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "04_rnn_intuition")
os.makedirs(_VISUALS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🔁 NLP MATH FOUNDATION 4: RNN INTUITION")
print("=" * 70)
print()
print("So far: text = bag of tokens (order doesn't matter)")
print("Problem: order DOES matter for meaning!")
print()
print("  'The dog bit the man'   vs  'The man bit the dog'")
print("  BoW: identical! Both contain {dog, bit, man, the}")
print("  Meaning: completely different!")
print()
print("  'It was not bad'  vs  'It was bad, not good'")
print("  The word 'not' changes meaning based on POSITION")
print()
print("We need a model that reads left-to-right, maintaining MEMORY of what")
print("it has seen — exactly like a human reading a sentence.")
print()
print("That model is the Recurrent Neural Network (RNN).")
print()


# ======================================================================
# SECTION 1: What Is an RNN?
# ======================================================================
print("=" * 70)
print("SECTION 1: THE RECURRENT NEURAL NETWORK")
print("=" * 70)
print()
print("Key idea: a HIDDEN STATE that gets updated at each timestep.")
print()
print("  At each word (timestep t):")
print("  h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
print("  y_t = W_hy × h_t + b_y")
print()
print("  Where:")
print("  x_t   = input at timestep t (word embedding)")
print("  h_t   = hidden state at t ('memory' of words 1..t)")
print("  h_t-1 = hidden state from previous timestep")
print("  W_hh  = recurrent weight matrix (hidden → hidden)")
print("  W_xh  = input weight matrix (input → hidden)")
print("  W_hy  = output weight matrix (hidden → output)")
print()
print("  Critical: W_hh, W_xh, W_hy are SHARED across all timesteps!")
print("  The SAME weights process word 1, word 2, ..., word 100.")
print("  This is how the RNN can handle sequences of ANY length.")
print()
print("  The 'unrolled' view:")
print()
print("  x_1  x_2  x_3  x_4  x_5")
print("   ↓    ↓    ↓    ↓    ↓")
print("  [RNN]→[RNN]→[RNN]→[RNN]→[RNN]")
print("  h_0  h_1   h_2   h_3   h_4   h_5")
print("                              ↓")
print("                           output")
print()


# ======================================================================
# SECTION 2: Vanilla RNN From Scratch
# ======================================================================
print("=" * 70)
print("SECTION 2: VANILLA RNN FORWARD PASS FROM SCRATCH")
print("=" * 70)
print()


class VanillaRNN:
    """
    Vanilla (Elman) RNN — forward pass.
    Input:  sequence of vectors x_t ∈ R^input_size
    Output: sequence of hidden states h_t ∈ R^hidden_size
            and final output y ∈ R^output_size
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight initialization (Xavier)
        scale_h = np.sqrt(1.0 / hidden_size)
        scale_i = np.sqrt(1.0 / input_size)

        self.W_xh = np.random.randn(input_size,  hidden_size) * scale_i   # input → hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_h   # hidden → hidden
        self.b_h  = np.zeros(hidden_size)

        self.W_hy = np.random.randn(hidden_size, output_size) * scale_h   # hidden → output
        self.b_y  = np.zeros(output_size)

    def forward(self, x_sequence, h0=None):
        """
        Forward pass through the RNN.

        Args:
            x_sequence: list of T input vectors, each shape (input_size,)
            h0:         initial hidden state (default: zeros)

        Returns:
            outputs:      list of T output vectors
            hidden_states: list of T+1 hidden states (h0, h1, ..., hT)
        """
        T = len(x_sequence)
        h = h0 if h0 is not None else np.zeros(self.hidden_size)

        hidden_states = [h.copy()]
        outputs       = []

        for t in range(T):
            x_t = x_sequence[t]

            # Update hidden state: h_t = tanh(W_xh @ x_t + W_hh @ h_t-1 + b_h)
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h.copy())

            # Output (often only at final timestep for classification)
            y_t = h @ self.W_hy + self.b_y
            outputs.append(y_t)

        return outputs, hidden_states

    def n_params(self):
        return (self.input_size * self.hidden_size +
                self.hidden_size * self.hidden_size +
                self.hidden_size +
                self.hidden_size * self.output_size +
                self.output_size)


# Demo: process a sentence character by character
sentence  = "the cat sat"
vocab_chr = sorted(set(sentence))
chr2idx   = {c: i for i, c in enumerate(vocab_chr)}
V_chr     = len(vocab_chr)

INPUT_SIZE  = V_chr       # one-hot character
HIDDEN_SIZE = 8
OUTPUT_SIZE = V_chr       # predict next character

rnn = VanillaRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
print(f"  Vanilla RNN on character sequence: {sentence!r}")
print(f"  Character vocab: {vocab_chr}  ({V_chr} characters)")
print(f"  Architecture: input={INPUT_SIZE}d → hidden={HIDDEN_SIZE}d → output={OUTPUT_SIZE}d")
print(f"  Parameters: {rnn.n_params():,}")
print()

# One-hot encode the sentence
x_seq = []
for char in sentence:
    onehot = np.zeros(V_chr)
    onehot[chr2idx[char]] = 1.0
    x_seq.append(onehot)

outputs, hidden_states = rnn.forward(x_seq)

print(f"  Sequence length T = {len(sentence)} characters")
print(f"  Hidden states: {len(hidden_states)} (including h_0)")
print(f"  Each hidden state shape: {hidden_states[0].shape}")
print()
print(f"  Hidden state evolution (first 4 dims) across timesteps:")
print(f"  {'t':>4}  {'char':>6}  {'h[0]':>7}  {'h[1]':>7}  {'h[2]':>7}  {'h[3]':>7}")
print(f"  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
for t, (char, h) in enumerate(zip(sentence, hidden_states[1:])):
    print(f"  {t:>4}  {char!r:>6}  {h[0]:>7.4f}  {h[1]:>7.4f}  {h[2]:>7.4f}  {h[3]:>7.4f}")
print()
print("  Notice: the hidden state changes with each character — it's accumulating context!")
print()


# ======================================================================
# SECTION 3: The Vanishing Gradient Problem
# ======================================================================
print("=" * 70)
print("SECTION 3: THE VANISHING GRADIENT PROBLEM")
print("=" * 70)
print()
print("Training an RNN requires Backpropagation Through Time (BPTT).")
print()
print("  For a sequence of length T, the gradient at t=1 requires")
print("  multiplying T weight matrices together:")
print()
print("  ∂L/∂h_0 = (∂L/∂h_T) × W_hh × W_hh × ... × W_hh")
print("              ─────────────────────────────────")
print("                        T times!")
print()
print("  Two cases:")
print()
print("  1. ||W_hh|| < 1:")
print("     Gradient shrinks exponentially: 0.9^50 ≈ 0.0052")
print("     → VANISHING GRADIENTS: early timesteps get near-zero gradient")
print("     → RNN 'forgets' long-range dependencies")
print()
print("  2. ||W_hh|| > 1:")
print("     Gradient grows exponentially: 1.1^50 ≈ 117.4")
print("     → EXPLODING GRADIENTS: training unstable (NaN)")
print("     → Fixed with gradient clipping: clip(grad, -threshold, threshold)")
print()

# Demonstrate gradient shrinkage numerically
print("  Simulating gradient flow through T timesteps:")
print()
print(f"  {'T':>5}  {'W_hh=0.9':<20}  {'W_hh=1.0':<20}  {'W_hh=1.1'}")
print(f"  {'─'*5}  {'─'*20}  {'─'*20}  {'─'*20}")

for T in [1, 5, 10, 25, 50, 100]:
    g_vanish = 0.9 ** T
    g_stable = 1.0 ** T
    g_explod = 1.1 ** T
    print(f"  {T:>5}  {g_vanish:<20.6f}  {g_stable:<20.6f}  {g_explod:.2f}")

print()
print("  At T=50: gradient for W_hh=0.9 is 0.0052 — practically zero!")
print("  The RNN cannot learn that 'it' refers to something from 50 words ago.")
print()
print("  Real-world problem: 'The cat, which was sleeping in the warm,")
print("  sunny, comfortable corner of the living room all afternoon, finally woke up.'")
print("  → 'woke' must agree with 'cat' — but 20+ words apart!")
print("  → Vanilla RNN fails this, LSTM succeeds")
print()


# ======================================================================
# SECTION 4: LSTM — Long Short-Term Memory
# ======================================================================
print("=" * 70)
print("SECTION 4: LSTM — SOLVING THE VANISHING GRADIENT")
print("=" * 70)
print()
print("LSTM (Hochreiter & Schmidhuber, 1997) adds a CELL STATE c_t")
print("— a 'conveyor belt' that runs straight through the sequence with only")
print("small linear interactions — making gradients flow without shrinking.")
print()
print("LSTM has 3 GATES (values between 0 and 1) that control information flow:")
print()
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  FORGET GATE:  how much of c_{t-1} to keep                 │")
print("  │  f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)                │")
print("  │  f_t = 0 → completely forget   f_t = 1 → completely keep   │")
print("  │                                                             │")
print("  │  INPUT GATE:   how much of new info to add                  │")
print("  │  i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)                │")
print("  │  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)    (candidate)    │")
print("  │                                                             │")
print("  │  CELL UPDATE:  update the cell state                        │")
print("  │  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                         │")
print("  │       (forget old)   +  (add new)                          │")
print("  │                                                             │")
print("  │  OUTPUT GATE:  what to expose as hidden state               │")
print("  │  o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)                │")
print("  │  h_t = o_t ⊙ tanh(c_t)                                     │")
print("  └─────────────────────────────────────────────────────────────┘")
print()
print("  The key gradient path: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t")
print("  Gradient flows through ADDITION — not multiplication!")
print("  → Gradients DON'T vanish through the cell state highway")
print()


class LSTMCell:
    """Single LSTM cell — forward pass only."""

    def __init__(self, input_size, hidden_size):
        self.D = hidden_size
        # All 4 gates computed in one matrix for efficiency
        # Stacked: [forget, input, gate (g), output] — each D neurons
        scale = np.sqrt(1.0 / hidden_size)
        self.W  = np.random.randn(input_size + hidden_size, 4 * hidden_size) * scale
        self.b  = np.zeros(4 * hidden_size)
        self.b[hidden_size:2*hidden_size] = 1.0  # bias forget gate → 1 (remember by default)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x, h_prev, c_prev):
        """One LSTM timestep."""
        combined = np.concatenate([x, h_prev])
        gates    = combined @ self.W + self.b

        D = self.D
        f = self.sigmoid(gates[:D])          # forget gate
        i = self.sigmoid(gates[D:2*D])       # input gate
        g = np.tanh(gates[2*D:3*D])          # cell gate (candidate)
        o = self.sigmoid(gates[3*D:])        # output gate

        c = f * c_prev + i * g              # cell state update
        h = o * np.tanh(c)                  # hidden state

        return h, c, {"f": f, "i": i, "g": g, "o": o}


# Demo LSTM forward pass
lstm_cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
h, c = np.zeros(HIDDEN_SIZE), np.zeros(HIDDEN_SIZE)

print(f"  LSTM forward pass on same sentence: {sentence!r}")
print()
print(f"  {'t':>4}  {'char':>6}  {'f_mean':>8}  {'i_mean':>8}  {'o_mean':>8}  {'|h|':>8}  {'|c|':>8}")
print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

all_cell_states = [c.copy()]
for t, (char, x_t) in enumerate(zip(sentence, x_seq)):
    h, c, gates = lstm_cell.forward(x_t, h, c)
    all_cell_states.append(c.copy())
    print(f"  {t:>4}  {char!r:>6}  {gates['f'].mean():>8.4f}  {gates['i'].mean():>8.4f}"
          f"  {gates['o'].mean():>8.4f}  {np.linalg.norm(h):>8.4f}  {np.linalg.norm(c):>8.4f}")

print()
print("  Gates interpretation:")
print("  • forget ≈ 0.5-0.7: LSTM is selectively remembering past")
print("  • input  ≈ 0.5: LSTM adds moderate new information each step")
print("  • output controls what to expose to downstream layers")
print()


# ======================================================================
# SECTION 5: GRU — Gated Recurrent Unit
# ======================================================================
print("=" * 70)
print("SECTION 5: GRU — SIMPLER ALTERNATIVE TO LSTM")
print("=" * 70)
print()
print("GRU (Cho et al., 2014): simpler than LSTM with only 2 gates.")
print("Merges cell state and hidden state into one. Often equally good.")
print()
print("  RESET GATE:  r_t = sigmoid(W_r · [h_{t-1}, x_t])")
print("               Controls how much of h_{t-1} to use for candidate h")
print()
print("  UPDATE GATE: z_t = sigmoid(W_z · [h_{t-1}, x_t])")
print("               Controls mix of old h and new candidate h")
print()
print("  Candidate:   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])")
print()
print("  Final:       h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t")
print("                     (keep old)            (add new)")
print()
print("  GRU vs LSTM:")
print("  ┌─────────────────┬──────────────┬──────────────┐")
print("  │ Property        │ LSTM         │ GRU          │")
print("  ├─────────────────┼──────────────┼──────────────┤")
print("  │ Gates           │ 3 (f, i, o)  │ 2 (r, z)     │")
print("  │ Parameters      │ 4 × more     │ 3 × more     │")
print("  │ Memory          │ Cell + hidden│ Hidden only  │")
print("  │ Training speed  │ Slower       │ Faster       │")
print("  │ Long sequences  │ Better       │ Competitive  │")
print("  │ Small data      │ May overfit  │ Often better │")
print("  └─────────────────┴──────────────┴──────────────┘")
print()
print("  Rule of thumb:")
print("  • Try GRU first (faster to train, fewer params)")
print("  • Switch to LSTM if GRU underfits on long sequences")
print("  • Modern NLP: Transformer attention often outperforms both")
print()


# ======================================================================
# SECTION 6: RNN for Sequence Tasks
# ======================================================================
print("=" * 70)
print("SECTION 6: WHAT CAN RNNs DO?")
print("=" * 70)
print()
print("Different 'modes' of RNN input-output:")
print()
print("  1. Many-to-one (Sequence Classification):")
print("     Input:  'I love this movie'  [T words → T hidden states]")
print("     Output: final h_T → Dense → 'Positive' / 'Negative'")
print("     Used for: sentiment analysis, spam detection")
print()
print("  2. One-to-many (Sequence Generation):")
print("     Input:  seed token")
print("     Output: h_1 → word_1, h_2 → word_2, ...")
print("     Used for: text generation, image captioning")
print()
print("  3. Many-to-many, equal length (Sequence Labeling):")
print("     Input:  'John Smith visited New York'  [5 words]")
print("     Output: 'PER   PER   O       LOC LOC'  [5 labels]")
print("     Used for: POS tagging, NER, token classification")
print()
print("  4. Many-to-many, different length (Seq2Seq):")
print("     Input:  'How are you?' [encoder RNN]")
print("     Output: 'Fine, thanks' [decoder RNN]")
print("     Used for: machine translation, summarization")
print()
print("  5. Bidirectional RNN (BiRNN):")
print("     Forward RNN:  reads left → right")
print("     Backward RNN: reads right → left")
print("     h_t = [h_forward_t ; h_backward_t]  (concatenated)")
print("     Used when: future context matters (NER, translation)")
print("     Not when:  autoregressive generation (can't peek ahead)")
print()


# ======================================================================
# SECTION 7: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Vanishing gradient demonstration ---
print("Generating: Vanishing gradient demonstration...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("The Vanishing Gradient Problem in RNNs",
             fontsize=14, fontweight="bold")

# Gradient magnitude vs timestep
timesteps = np.arange(1, 101)
w_values  = [0.8, 0.9, 1.0, 1.1, 1.2]
colors_v  = ["#E74C3C", "#E67E22", "#2ECC71", "#F1C40F", "#3498DB"]

for w, color in zip(w_values, colors_v):
    grads = w ** timesteps
    # Clip for display
    grads = np.clip(grads, 1e-8, 1e6)
    axes[0].semilogy(timesteps, grads, linewidth=2, color=color, label=f"W={w:.1f}")

axes[0].axhline(y=1e-3, color="gray", linestyle="--", alpha=0.6, label="~zero (1e-3)")
axes[0].axhline(y=1e3,  color="gray", linestyle=":",  alpha=0.6, label="~explode (1e3)")
axes[0].set_xlabel("Timesteps ago"); axes[0].set_ylabel("Gradient Magnitude (log)")
axes[0].set_title("Gradient Magnitude vs Distance\n(how far gradient travels back)", fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].fill_between(timesteps, 1e-3, 1e-8, alpha=0.1, color="red", label="Dead zone")
axes[0].set_ylim(1e-8, 1e7)

# Hidden state evolution comparison: RNN vs LSTM
# Simulate hidden state norms over time for both
T_sim  = 20
rnn_norm  = []
lstm_norm = []

h_rnn  = np.zeros(HIDDEN_SIZE)
h_lstm = np.zeros(HIDDEN_SIZE)
c_lstm = np.zeros(HIDDEN_SIZE)

for t in range(T_sim):
    x_t = np.random.randn(INPUT_SIZE) * 0.3
    # RNN update
    h_rnn = np.tanh(x_t @ rnn.W_xh + h_rnn @ rnn.W_hh + rnn.b_h)
    rnn_norm.append(np.linalg.norm(h_rnn))
    # LSTM update
    h_lstm, c_lstm, _ = lstm_cell.forward(x_t, h_lstm, c_lstm)
    lstm_norm.append(np.linalg.norm(h_lstm))

axes[1].plot(rnn_norm,  "r-o", linewidth=2, markersize=6, label="Vanilla RNN ||h_t||")
axes[1].plot(lstm_norm, "b-s", linewidth=2, markersize=6, label="LSTM ||h_t||")
axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("||h_t|| (hidden state norm)")
axes[1].set_title("Hidden State Evolution\nRNN vs LSTM", fontsize=11, fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# LSTM gate activity visualization
lstm_cell2 = LSTMCell(4, 6)
h_g, c_g = np.zeros(6), np.zeros(6)

# Simulate with "sentiment" context (positive then negative)
inputs_g = (
    [np.array([1, 0, 0, 0])] * 5 +   # positive signal
    [np.array([0, 0, 0, 1])] * 5 +   # negative signal
    [np.array([0, 1, 0, 0])] * 5     # neutral
)

gate_history = {"f": [], "i": [], "g_abs": [], "o": []}

for x_g in inputs_g:
    h_g, c_g, gates_g = lstm_cell2.forward(x_g, h_g, c_g)
    gate_history["f"].append(gates_g["f"].mean())
    gate_history["i"].append(gates_g["i"].mean())
    gate_history["g_abs"].append(abs(gates_g["g"]).mean())
    gate_history["o"].append(gates_g["o"].mean())

T_g = len(inputs_g)
axes[2].plot(gate_history["f"],     "r-",  linewidth=2, label="Forget gate (f)")
axes[2].plot(gate_history["i"],     "b-",  linewidth=2, label="Input gate (i)")
axes[2].plot(gate_history["o"],     "g-",  linewidth=2, label="Output gate (o)")
axes[2].plot(gate_history["g_abs"], "m--", linewidth=2, label="|Cell gate (g)|")
axes[2].axvspan(0,  5, alpha=0.08, color="#2ECC71", label="Positive context")
axes[2].axvspan(5,  10, alpha=0.08, color="#E74C3C", label="Negative context")
axes[2].axvspan(10, 15, alpha=0.08, color="#3498DB", label="Neutral context")
axes[2].set_xlabel("Timestep"); axes[2].set_ylabel("Gate Activation (mean)")
axes[2].set_title("LSTM Gate Activations Over Time\n(context shifts at t=5,10)",
                  fontsize=11, fontweight="bold")
axes[2].legend(fontsize=8, loc="upper right"); axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "vanishing_gradient_and_lstm.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: vanishing_gradient_and_lstm.png")


# --- PLOT 2: RNN Architecture diagrams ---
print("Generating: RNN architecture comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RNN Architecture Family: Vanilla RNN → LSTM → GRU",
             fontsize=14, fontweight="bold")

arch_configs = [
    {
        "title": "Vanilla RNN",
        "subtitle": "h_t = tanh(W_xh·x + W_hh·h_{t-1})",
        "boxes": [
            ("x_t", 0.5, 0.10, "#3498DB"),
            ("tanh", 0.5, 0.40, "#E74C3C"),
            ("h_t", 0.5, 0.72, "#2ECC71"),
        ],
        "arrows": [(0.5, 0.18, 0.5, 0.32), (0.5, 0.52, 0.5, 0.64)],
        "recurrent": True,
        "note": "1 activation\nSimple but\nvanishing gradient",
    },
    {
        "title": "LSTM",
        "subtitle": "3 gates: forget (f), input (i), output (o)",
        "boxes": [
            ("x_t", 0.5, 0.08, "#3498DB"),
            ("f gate\nsigmoid", 0.20, 0.38, "#E74C3C"),
            ("i gate\nsigmoid", 0.50, 0.38, "#9B59B6"),
            ("o gate\nsigmoid", 0.80, 0.38, "#E67E22"),
            ("Cell c_t", 0.5, 0.60, "#F39C12"),
            ("h_t", 0.5, 0.82, "#2ECC71"),
        ],
        "note": "3 gates\nSolves vanishing\ngradient perfectly",
    },
    {
        "title": "GRU",
        "subtitle": "2 gates: reset (r), update (z)",
        "boxes": [
            ("x_t", 0.5, 0.08, "#3498DB"),
            ("r gate\nsigmoid", 0.30, 0.38, "#E74C3C"),
            ("z gate\nsigmoid", 0.70, 0.38, "#9B59B6"),
            ("h̃_t\ntanh", 0.50, 0.60, "#F39C12"),
            ("h_t", 0.5, 0.82, "#2ECC71"),
        ],
        "note": "2 gates\nFewer params\nFaster training",
    },
]

for ax, cfg in zip(axes, arch_configs):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, 0.97, cfg["subtitle"], ha="center", va="top", fontsize=8.5,
            color="#555", style="italic")

    for label, x, y, color in cfg["boxes"]:
        rect = mpatches.FancyBboxPatch((x - 0.12, y - 0.06), 0.24, 0.12,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor="white",
                                       linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")

    if cfg.get("arrows"):
        for (x0, y0, x1, y1) in cfg["arrows"]:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    if cfg.get("recurrent"):
        ax.annotate("", xy=(0.72, 0.40), xytext=(0.72, 0.72),
                    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2))
        ax.annotate("", xy=(0.58, 0.40), xytext=(0.72, 0.40),
                    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2))
        ax.text(0.82, 0.56, "h_{t-1}", fontsize=8, color="#9B59B6", fontweight="bold")

    ax.text(0.5, 0.05, cfg["note"], ha="center", va="bottom",
            fontsize=9, color="#333", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "architecture_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: architecture_comparison.png")


# --- PLOT 3: Hidden state dynamics + RNN modes ---
print("Generating: Hidden state dynamics and RNN modes...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("RNN Hidden State Dynamics and Task Modes",
             fontsize=14, fontweight="bold")

# Hidden state heatmap over sequence
h_all = np.array(hidden_states[1:])   # T × D

im = axes[0].imshow(h_all.T, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
axes[0].set_yticks(range(HIDDEN_SIZE))
axes[0].set_yticklabels([f"h[{i}]" for i in range(HIDDEN_SIZE)], fontsize=9)
axes[0].set_xticks(range(len(sentence)))
axes[0].set_xticklabels(list(sentence), fontsize=11)
axes[0].set_xlabel("Character (timestep)")
axes[0].set_ylabel("Hidden State Dimension")
axes[0].set_title(f"RNN Hidden State: {sentence!r}\n"
                  "Each column = h_t after reading that character", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=axes[0], shrink=0.8, label="Activation")

# RNN task modes diagram
modes = [
    ("Many-to-One\n(Classification)", "Sentiment:\n'I love this' → Positive", "#3498DB"),
    ("One-to-Many\n(Generation)",     "Generate:\n'<start>' → 'The cat ...'", "#E74C3C"),
    ("Many-to-Many\n(Seq Labeling)",  "NER:\n'John London' → 'PER LOC'",     "#2ECC71"),
    ("Seq2Seq\n(Translation)",        "MT:\n'Hello' → 'Bonjour'",            "#9B59B6"),
]

ax2 = axes[1]
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.set_title("RNN Task Modes", fontsize=12, fontweight="bold")

bw2, bh2 = 0.22, 0.14
ys = [0.80, 0.60, 0.40, 0.18]

for (title, example, color), y in zip(modes, ys):
    rect = mpatches.FancyBboxPatch((0.02, y - bh2/2), 0.96, bh2,
                                   boxstyle="round,pad=0.01",
                                   facecolor=color, alpha=0.15,
                                   edgecolor=color, linewidth=2)
    ax2.add_patch(rect)
    ax2.text(0.12, y, title, ha="center", va="center",
             fontsize=9, fontweight="bold", color=color)
    ax2.text(0.60, y, example, ha="center", va="center",
             fontsize=9, color="#333")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "hidden_state_dynamics.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: hidden_state_dynamics.png")


print()
print("=" * 70)
print("NLP MATH FOUNDATION 4: RNN INTUITION COMPLETE!")
print("=" * 70)
print()
print("What you learned:")
print("  ✓ Why sequence order matters ('not bad' ≠ 'bad not' in BoW)")
print("  ✓ RNN: hidden state h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1})")
print("  ✓ Shared weights across timesteps → works for any sequence length")
print("  ✓ Vanishing gradient: W_hh^T → 0 exponentially for W_hh < 1")
print("  ✓ LSTM: 3 gates (forget/input/output) + cell state highway")
print("  ✓ GRU: 2 gates (reset/update) — faster, fewer params, competitive")
print("  ✓ RNN modes: many-to-one, one-to-many, many-to-many, seq2seq")
print()
print("3 Visualizations saved to: ../visuals/04_rnn_intuition/")
print("  1. vanishing_gradient_and_lstm.png — gradient decay + LSTM gates")
print("  2. architecture_comparison.png     — RNN vs LSTM vs GRU diagrams")
print("  3. hidden_state_dynamics.png       — hidden state heatmap + task modes")
print()
print("🎉 All 4 NLP Math Foundations Complete!")
print()
print("Next: Algorithm 1 → Text Classification Pipeline")
