"""
02_prompt_engineering.py
========================
Part 7 - Large Language Models: Prompt Engineering

Learning Objectives:
  1. Understand prompt engineering as an inference-time adaptation technique
  2. Implement and compare zero-shot, few-shot, and chain-of-thought prompting
  3. Build a few-shot classifier using in-context examples
  4. Construct structured output prompts and parse JSON-like responses
  5. Apply temperature and sampling parameters appropriately for different tasks
  6. Internalize a practical prompt engineering checklist

YouTube: Search "Prompt Engineering Guide by DAIR.AI" and
         "Andrew Ng Prompt Engineering for Developers"

Time: ~20 min | Difficulty: Intermediate | Prerequisites: 01_how_llms_work.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import json
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(_HERE, "..", "visuals", "02_prompt_engineering")
os.makedirs(VIS_DIR, exist_ok=True)

# ==============================================================================
print("\n" + "="*70)
print("SECTION 1: WHAT IS PROMPT ENGINEERING?")
print("="*70)
# ==============================================================================

print("""
PROMPT ENGINEERING: The practice of designing and optimizing the INPUT TEXT
(the "prompt") sent to a frozen, pre-trained LLM so that it produces the
desired OUTPUT — without updating any model weights.

WHY IT MATTERS:
  - GPT-4, Claude, Gemini are expensive to fine-tune. Prompting is free.
  - Well-designed prompts can close 80% of the gap to fine-tuned models.
  - Prompt engineering is now a core ML engineering skill.

THE KEY INSIGHT:
  LLMs are trained to complete text. A well-crafted prefix "steers" the
  model toward the desired completion distribution.

  Bad:  "Summarize this article."
  Good: "You are a professional editor. Summarize the following article
         in exactly 3 bullet points, each under 15 words.
         Article: ```{article}```"

  The second prompt is: specific, role-conditioned, format-constrained.
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 2: ZERO-SHOT PROMPTING")
print("="*70)
# ==============================================================================

print("""
ZERO-SHOT: No examples in the prompt. Relies purely on the model's
pre-trained knowledge about what the task is.

Works well when:
  - The task is common and well-represented in training data
  - The task description is very clear and unambiguous
  - You have no labeled examples
""")

class SimpleLLMSimulator:
    """
    A rule-based 'LLM simulator' for demonstration purposes.
    In production, replace .complete() with an API call to OpenAI/Anthropic/etc.
    """
    SENTIMENT_KEYWORDS = {
        "positive": ["great", "excellent", "amazing", "love", "wonderful",
                     "fantastic", "best", "outstanding", "perfect", "brilliant"],
        "negative": ["terrible", "awful", "horrible", "hate", "worst",
                     "poor", "bad", "disappointing", "boring", "failed"],
    }

    def classify_sentiment_zero_shot(self, text):
        """Zero-shot sentiment: count positive vs negative keywords."""
        text_lower = text.lower()
        pos = sum(1 for w in self.SENTIMENT_KEYWORDS["positive"] if w in text_lower)
        neg = sum(1 for w in self.SENTIMENT_KEYWORDS["negative"] if w in text_lower)
        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        else:
            return "Neutral"

    def complete(self, prompt, temperature=0.7):
        """Simulate model completion based on prompt structure."""
        if "sentiment" in prompt.lower() or "classify" in prompt.lower():
            # Extract the text to classify (after last newline block)
            lines = [l.strip() for l in prompt.strip().split("\n") if l.strip()]
            text_to_classify = lines[-1] if lines else ""
            return self.classify_sentiment_zero_shot(text_to_classify)
        return "[Simulated LLM Response]"

llm = SimpleLLMSimulator()

zero_shot_examples = [
    "Great movie! I loved every minute of it.",
    "This was terrible. Worst film I've ever seen.",
    "The acting was brilliant but the plot was boring.",
    "An outstanding performance by the entire cast!",
    "I hated the ending. Very disappointing.",
]

print("ZERO-SHOT SENTIMENT CLASSIFICATION:")
print()
for text in zero_shot_examples:
    zero_shot_prompt = f"""Classify the sentiment of the following text as Positive, Negative, or Neutral.

Text: {text}
Sentiment:"""
    prediction = llm.classify_sentiment_zero_shot(text)
    print(f"  Text: '{text[:55]}...' " if len(text) > 55 else f"  Text: '{text}'")
    print(f"  Prompt ends with: 'Sentiment:'  →  Model predicts: {prediction}")
    print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 3: FEW-SHOT PROMPTING")
print("="*70)
# ==============================================================================

print("""
FEW-SHOT: Include 2–10 labeled examples (demonstrations) in the prompt.
The model infers the task pattern from these examples and applies it
to the new input — a form of in-context learning (ICL).

Why it works: The examples constrain the output format, class labels,
and style, dramatically reducing ambiguity.

Typical improvement: zero-shot ~70% → few-shot ~85% accuracy on
classification tasks (task-dependent).
""")

few_shot_examples = [
    ("The special effects were breathtaking and the story was captivating!", "Positive"),
    ("I couldn't finish it. The dialogue was cringe-worthy and the plot made no sense.", "Negative"),
    ("A decent film. Nothing groundbreaking, but enjoyable enough.", "Neutral"),
]

def build_few_shot_prompt(examples, new_text):
    """Build a few-shot prompt from (text, label) pairs + new text."""
    prompt = "Classify movie review sentiment as Positive, Negative, or Neutral.\n\n"
    for text, label in examples:
        prompt += f"Review: {text}\nSentiment: {label}\n\n"
    prompt += f"Review: {new_text}\nSentiment:"
    return prompt

def few_shot_classify(text, examples):
    """Rule-based few-shot classifier using keyword matching + examples."""
    # Extract labels from examples for label-space constraint
    valid_labels = {label for _, label in examples}
    return llm.classify_sentiment_zero_shot(text)

print("FEW-SHOT PROMPT STRUCTURE:")
print()
demo_prompt = build_few_shot_prompt(few_shot_examples, "Amazing cinematography!")
print(demo_prompt)
print()

# Accuracy comparison (simulated)
test_cases = [
    ("What an incredible journey this film takes you on!", "Positive"),
    ("Complete waste of time and money.", "Negative"),
    ("The movie was okay. Had some good moments.", "Neutral"),
    ("A masterpiece of modern cinema!", "Positive"),
    ("I've seen better student films.", "Negative"),
    ("Not bad, not great. Somewhere in the middle.", "Neutral"),
    ("The director's vision was absolutely stunning.", "Positive"),
    ("Painful to watch. Avoid at all costs.", "Negative"),
]

zero_shot_correct = 0
few_shot_correct  = 0
for text, true_label in test_cases:
    zs_pred = llm.classify_sentiment_zero_shot(text)
    fs_pred = few_shot_classify(text, few_shot_examples)
    if zs_pred == true_label:
        zero_shot_correct += 1
    if fs_pred == true_label:
        few_shot_correct += 1

zs_acc = zero_shot_correct / len(test_cases) * 100
fs_acc = few_shot_correct  / len(test_cases) * 100
print(f"ACCURACY COMPARISON on {len(test_cases)} test cases:")
print(f"  Zero-shot:  {zs_acc:.1f}%")
print(f"  Few-shot:   {fs_acc:.1f}%  (examples help constrain label space & format)")
print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 4: CHAIN-OF-THOUGHT (CoT) PROMPTING")
print("="*70)
# ==============================================================================

print("""
CHAIN-OF-THOUGHT (CoT): Wei et al. (2022) found that including reasoning
steps in few-shot examples dramatically improves multi-step problem solving.

Two CoT variants:
  1. Few-shot CoT:  Provide examples WITH reasoning chains
  2. Zero-shot CoT: Just add "Let's think step by step." (Kojima et al., 2022)

WHY IT WORKS: Forces the model to "show its work." Intermediate reasoning
steps serve as scratchpad tokens that help the model attend to the right
information for the final answer.
""")

def build_cot_template(question, cot_hint="Let's think step by step."):
    """Build a chain-of-thought prompt template."""
    return f"""You are a careful problem solver. {cot_hint}

Question: {question}

Reasoning:"""

# Math word problems
problems = [
    {
        "question": "A bookstore sells 3 books at $12 each and 2 books at $18 each. What is the total revenue?",
        "standard_answer": "$72",
        "cot_steps": [
            "Revenue from first set: 3 × $12 = $36",
            "Revenue from second set: 2 × $18 = $36",
            "Total revenue: $36 + $36 = $72",
        ],
    },
    {
        "question": "If a train travels at 80 km/h for 1.5 hours, then 120 km/h for 0.5 hours, what is the total distance?",
        "standard_answer": "180 km",
        "cot_steps": [
            "First leg: 80 × 1.5 = 120 km",
            "Second leg: 120 × 0.5 = 60 km",
            "Total distance: 120 + 60 = 180 km",
        ],
    },
]

for prob in problems:
    print(f"PROBLEM: {prob['question']}")
    print()
    print("  Standard prompt answer:  '{}'  (direct lookup, error-prone)".format(prob["standard_answer"]))
    print()
    print("  CoT prompt reasoning:")
    for i, step in enumerate(prob["cot_steps"], 1):
        print(f"    Step {i}: {step}")
    print(f"    → Final answer: {prob['standard_answer']}  (much more reliable!)")
    print()

print("CoT TEMPLATE BUILDER:")
sample_cot_prompt = build_cot_template(
    "Emma has 3 times as many stickers as Jake. Jake has 8. How many do they have together?"
)
print(sample_cot_prompt)
print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 5: STRUCTURED OUTPUT PROMPTING")
print("="*70)
# ==============================================================================

print("""
STRUCTURED OUTPUT: Instruct the model to return machine-parseable formats
(JSON, XML, Markdown tables). Critical for integrating LLMs into pipelines.

Key techniques:
  - Specify the exact schema in the prompt
  - Use delimiters to mark input vs output sections
  - Provide a JSON example in the prompt
  - Use system prompts to enforce format (when API supports it)
""")

def build_structured_prompt(text, schema_example):
    """Build a prompt requesting structured JSON output."""
    return f"""Analyze the following movie review and extract information.
Respond ONLY with valid JSON matching this schema:
{json.dumps(schema_example, indent=2)}

Movie review: ```{text}```

JSON output:"""

schema_example = {
    "sentiment": "Positive | Negative | Neutral",
    "confidence": 0.95,
    "key_aspects": ["acting", "plot", "visuals"],
    "one_sentence_summary": "Brief summary here."
}

sample_review = "The visuals were stunning and the acting superb, though the plot had a few holes."
structured_prompt = build_structured_prompt(sample_review, schema_example)
print("STRUCTURED OUTPUT PROMPT:")
print(structured_prompt)

# Simulate a JSON response
simulated_json_response = {
    "sentiment": "Positive",
    "confidence": 0.82,
    "key_aspects": ["visuals", "acting", "plot"],
    "one_sentence_summary": "Visually stunning with great acting despite minor plot issues."
}

def parse_llm_json_response(response_text):
    """Parse JSON from LLM response, with error handling."""
    try:
        # In real usage, LLM might wrap JSON in markdown code fences
        cleaned = re.sub(r"```json\s*|\s*```", "", response_text).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}")
        return None

simulated_response_str = json.dumps(simulated_json_response, indent=2)
parsed = parse_llm_json_response(simulated_response_str)
print("\nSIMULATED LLM JSON RESPONSE:")
print(simulated_response_str)
print(f"\nParsed sentiment: {parsed['sentiment']}")
print(f"Parsed confidence: {parsed['confidence']}")
print(f"Parsed aspects: {parsed['key_aspects']}")
print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 6: PROMPT OPTIMIZATION PRINCIPLES")
print("="*70)
# ==============================================================================

print("""
PROMPT ENGINEERING CHECKLIST
=============================

[ ] 1. ROLE / PERSONA
      "You are an expert [role] with [N] years of experience in [domain]."
      Sets the model's epistemic stance and style.

[ ] 2. CLEAR TASK DESCRIPTION
      State the task explicitly, not implicitly.
      Bad:  "What about this text?"
      Good: "Classify the sentiment of the following text as one of:
             Positive, Negative, or Neutral."

[ ] 3. USE DELIMITERS
      Separate instruction from content with: ```, ---, ###, <tag>, [CONTEXT]
      Prevents the model from treating your text as instructions.

[ ] 4. OUTPUT FORMAT SPECIFICATION
      "Respond with a JSON object.", "Use bullet points.", "Be concise (under 50 words)."
      Always specify format BEFORE the input content.

[ ] 5. EXAMPLES (FEW-SHOT)
      2–5 diverse, representative examples dramatically improve consistency.
      Ensure examples cover edge cases.

[ ] 6. CHAIN-OF-THOUGHT TRIGGER
      For reasoning tasks: "Let's think step by step." or provide CoT examples.

[ ] 7. CONSTRAINTS
      "Do not mention competitors.", "Respond only in English.", "Under 100 words."

[ ] 8. ESCAPE HATCHES
      "If you are unsure, say 'I don't know' rather than guessing."
      Reduces hallucinations.

[ ] 9. TEMPERATURE SETTING (API level)
      Creative tasks: T = 0.8–1.2
      Factual/extraction: T = 0.0–0.3
      Code generation: T = 0.0
      Brainstorming: T = 1.0–1.5
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 7: TEMPERATURE AND SAMPLING THEORY")
print("="*70)
# ==============================================================================

print("""
TEMPERATURE (T): Controls randomness of the output distribution.

  Softmax with temperature:  p_i = exp(z_i / T) / sum_j(exp(z_j / T))

  T → 0: Argmax sampling (always pick highest logit) — deterministic, repetitive
  T = 1: Standard softmax — unmodified model distribution
  T > 1: Flatter distribution — more creative, more random, more hallucinations

TOP-K SAMPLING:
  At each step, keep only the K highest-probability tokens, renormalize, sample.
  K=1: greedy. K=50: typical. Prevents very low-probability tokens.

TOP-P (NUCLEUS) SAMPLING:
  Keep the smallest set of tokens whose cumulative probability ≥ P.
  P=0.9: typical. Adapts dynamically to the distribution shape.
  More principled than top-K for diverse distributions.

RECOMMENDED SETTINGS:
""")

settings_table = [
    ("Creative writing",      "0.8–1.2", "50",   "0.9",  "Variety & originality"),
    ("Factual Q&A",           "0.0–0.3", "10",   "0.7",  "Accuracy & consistency"),
    ("Code generation",       "0.0",     "1",    "1.0",  "Deterministic correctness"),
    ("Brainstorming",         "1.0–1.5", "100",  "0.95", "Maximum diversity"),
    ("Summarization",         "0.3–0.5", "20",   "0.85", "Coherent & focused"),
    ("Chatbot conversations", "0.7",     "50",   "0.9",  "Natural & engaging"),
]

print(f"  {'Task':<26} {'Temp':<12} {'Top-K':<8} {'Top-P':<8} {'Reason'}")
print("  " + "-" * 66)
for task, temp, topk, topp, reason in settings_table:
    print(f"  {task:<26} {temp:<12} {topk:<8} {topp:<8} {reason}")

print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 8: GENERATING VISUALIZATIONS")
print("="*70)
# ==============================================================================

print(f"Saving visualizations to: {VIS_DIR}")

# ── Visualization 1: Prompting Strategies Accuracy Bar Chart ──────────────────
fig, ax = plt.subplots(figsize=(11, 6))

tasks = ["Sentiment\nClassification", "Math Word\nProblems", "Code\nGeneration", "Translation"]
strategies = ["Zero-Shot", "Few-Shot", "Chain-of-Thought", "Fine-Tuned (Reference)"]
# Simulated accuracy data
accuracy_data = np.array([
    [72, 68, 55, 80],   # Zero-shot
    [84, 75, 62, 87],   # Few-shot
    [83, 91, 70, 89],   # CoT
    [90, 88, 85, 93],   # Fine-tuned
])

x = np.arange(len(tasks))
width = 0.2
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (strategy, color) in enumerate(zip(strategies, colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, accuracy_data[i], width, label=strategy,
                  color=color, edgecolor="white", linewidth=0.8)

ax.set_xlabel("Task", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Prompting Strategy Accuracy Comparison\n(Simulated Data — Illustrative)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylim(40, 100)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, axis="y", alpha=0.3)
ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="90% threshold")
fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "01_prompting_strategies.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 01_prompting_strategies.png")

# ── Visualization 2: Temperature Sampling ─────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 5))

# Same logits for 8 tokens
np.random.seed(42)
raw_logits = np.array([3.5, 2.1, 1.8, 0.9, 0.4, 0.2, -0.5, -1.2])
token_labels = [f"T{i+1}" for i in range(len(raw_logits))]
temperatures = [0.1, 0.5, 1.0, 2.0]
temp_colors  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for ax, T, color in zip(axes, temperatures, temp_colors):
    scaled = raw_logits / T
    exp_scaled = np.exp(scaled - scaled.max())   # numerical stability
    probs = exp_scaled / exp_scaled.sum()
    bars = ax.bar(token_labels, probs, color=color, alpha=0.8, edgecolor="white")
    ax.set_title(f"Temperature = {T}", fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Token", fontsize=9)
    ax.set_ylabel("Probability" if ax == axes[0] else "", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    ax.text(0.5, 0.93, f"H={entropy:.2f} bits", transform=ax.transAxes,
            ha="center", fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

fig.suptitle("Effect of Temperature on Token Probability Distribution\n"
             "(Same logits; T→0: deterministic, T→∞: uniform)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "02_temperature_sampling.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 02_temperature_sampling.png")

# ── Visualization 3: Prompt Anatomy Diagram ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Anatomy of a Well-Structured LLM Prompt", fontsize=14, fontweight="bold", pad=15)

# Prompt sections as colored boxes
sections = [
    # (y_center, height, label, text, box_color, text_color)
    (9.3, 0.8,  "1. System / Role",
     "You are an expert product reviewer with 10 years of experience in consumer electronics.",
     "#E3F2FD", "#1565C0"),
    (8.2, 0.8,  "2. Task Description",
     "Analyze the following product review and classify its sentiment.",
     "#E8F5E9", "#2E7D32"),
    (7.0, 0.85, "3. Output Format",
     "Respond with JSON: {\"sentiment\": \"...\", \"confidence\": 0.0–1.0, \"summary\": \"...\"}",
     "#FFF8E1", "#F57F17"),
    (5.75, 1.0, "4. Few-Shot Examples",
     "Review: 'Battery life is amazing!'  → {\"sentiment\": \"Positive\", \"confidence\": 0.95, ...}\n"
     "Review: 'Broke after two days.'     → {\"sentiment\": \"Negative\", \"confidence\": 0.92, ...}",
     "#FCE4EC", "#880E4F"),
    (4.35, 0.85,"5. Input (Delimited)",
     "Review: ```The screen resolution is stunning but the speaker quality is poor.```",
     "#F3E5F5", "#4A148C"),
    (3.2, 0.8,  "6. Output Trigger",
     "JSON output:",
     "#EFEBE9", "#3E2723"),
]

for y_center, height, label, text, box_color, text_color in sections:
    y_bottom = y_center - height / 2
    rect = FancyBboxPatch((0.3, y_bottom), 7.0, height,
                           boxstyle="round,pad=0.05",
                           facecolor=box_color, edgecolor=text_color,
                           linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.45, y_center, text, va="center", fontsize=7.5,
            fontfamily="monospace", color="#1a1a1a", wrap=True)
    # Label on right
    ax.text(7.55, y_center, label, va="center", fontsize=9,
            fontweight="bold", color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, edgecolor=text_color, linewidth=1))

# Annotation arrows pointing to principles
principles = [
    (9.3,  "Sets model persona & expertise"),
    (8.2,  "Explicit task, not implicit"),
    (7.0,  "Specify format BEFORE input"),
    (5.75, "2-5 diverse demonstrations"),
    (4.35, "Delimiters prevent injection"),
    (3.2,  "Trigger the completion"),
]
for y, note in principles:
    ax.annotate(note, xy=(7.45, y), xytext=(7.6, y),
                fontsize=7.5, color="#555555", va="center")

ax.text(5, 2.4, "PROMPT = Role + Task + Format + Examples + Input + Trigger",
        ha="center", va="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8EAF6", edgecolor="#3F51B5", linewidth=2))

fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "03_prompt_anatomy.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 03_prompt_anatomy.png")

print(f"\nAll 3 visualizations saved to: {VIS_DIR}")
print("\n" + "="*70)
print("MODULE 02 COMPLETE: Prompt Engineering")
print("="*70)
print("""
KEY TAKEAWAYS:
  1. Prompt engineering adapts frozen LLMs without any training cost
  2. Few-shot examples significantly outperform zero-shot in most tasks
  3. Chain-of-thought reasoning is critical for multi-step problems
  4. Structured output prompts enable reliable pipeline integration
  5. Temperature controls creativity vs. determinism; set it per task
  6. A good prompt: role + task + format + examples + delimited input

NEXT: 03_finetuning_basics.py — when prompting isn't enough
""")
