# GLM-Image Blog: Evaluation Metrics Explained

This note summarizes the evaluation metrics **listed on the GLM-Image blog** and explains what each metric measures and how it is typically computed. The blog’s tables explicitly list metrics such as **NED**, **CLIPScore**, **Word Accuracy (2–5 regions & average)**, **LongText-Bench (EN/ZH)**, and **OneIG_Bench (Alignment/Text/Reasoning/Style/Diversity/Overall)**.

> Some benchmark columns on the blog are **reported without formulas**. Where the blog does not define a formula, I provide the *standard/commonly used* definition and note it explicitly.

---

## 1) NED (Normalized Edit Distance)

**What it evaluates**  
String similarity between the OCR-extracted text and the target text. In text-rendering tasks, NED reflects how close the generated text is to the desired words.

**How it is calculated**  
A common definition normalizes Levenshtein (edit) distance by the maximum string length, producing a score in \([0, 1]\), where 0 indicates identical strings and 1 indicates completely different strings.  
Typical form used in text-evaluation:

$$
\mathrm{NED}(x,y) = \frac{\mathrm{Levenshtein}(x,y)}{\max(|x|,|y|)}
$$

If you see NED reported as a **similarity** instead of a distance, it may be converted as \(1-\mathrm{NED}\). (The blog does not specify which variant is used.)

---

## 2) CLIPScore

**What it evaluates**  
Text–image semantic alignment using CLIP embeddings. It measures how well the generated image matches the prompt content.

**How it is calculated**  
The metric uses cosine similarity between CLIP image and text embeddings:

$$
\mathrm{CLIPScore}(I, C) = \max(100 \cdot \cos(E_I, E_C), 0)
$$

where \(E_I\) is the image embedding and \(E_C\) is the text embedding.

---

## 3) Word Accuracy

**What it evaluates**  
How many **words** in the OCR output exactly match the target words. This is commonly used in OCR and text-rendering evaluation.

**How it is calculated**  
A typical definition is:

$$
\mathrm{WordAccuracy} = \frac{\text{# correct words}}{\text{# total target words}}
$$

In OCR evaluation literature, word-level accuracy is closely related to **word error rate (WER)**. WER is derived from Levenshtein distance over words (substitutions, insertions, deletions), and word accuracy is often computed as \(1 - \mathrm{WER}\) (when normalized by the reference length).

---

## 4) Word Accuracy by “Regions” (2–5 regions) + Average

**What it evaluates**  
The blog’s CVTG-2k table reports Word Accuracy for images containing **2, 3, 4, or 5 text regions**, plus an average.

**How it is calculated (likely)**  
Compute Word Accuracy on each subset of images with exactly \(k\) text regions (where each region contains its own target text), then average across those subsets:

$$
\mathrm{WordAcc}_{k} = \frac{\#\text{correct words in images with }k\text{ regions}}{\#\text{target words in those images}}
$$

The **“average”** is likely the mean of WordAcc\(_2\)–WordAcc\(_5\). The blog does not provide the exact formula, so confirm with the benchmark’s official documentation.

---

## 5) LongText-Bench (EN / ZH)

**What it evaluates**  
Long text rendering accuracy for English and Chinese prompts. The GLM-Image blog reports a single score for each language split.

**How it is calculated (not specified on the blog)**  
LongText-Bench typically uses OCR-based text comparison metrics such as NED/Word Accuracy on long text strings. The blog does not state the exact formula, so you should consult the benchmark’s official documentation for exact details.

---

## 6) OneIG-Bench (OneIG_EN): Alignment / Text / Reasoning / Style / Diversity / Overall

The GLM-Image blog reports the following OneIG_EN columns: **Alignment, Text, Reasoning, Style, Diversity, Overall**.  
Below are the **metric intents and typical calculation approaches** from the OneIG-Bench description:

### Alignment
**What it evaluates**  
Semantic matching between the generated image and prompt, aggregated over categories like general objects, portrait, and stylization.

**How it is calculated**  
OneIG-Bench describes **question-based scoring** using VLMs (e.g., GPT-4o, Qwen2.5-VL) to measure prompt compliance.

### Text
**What it evaluates**  
Accuracy of rendered text vs. ground truth.

**How it is calculated**  
OneIG-Bench uses **Edit Distance**, **Completion Rate**, and **Word Accuracy** for the text-rendering subset.  
(The exact combination/weighting is defined in the benchmark code.)

### Reasoning
**What it evaluates**  
Visual reasoning/knowledge correctness (e.g., diagrams, multi-step relations).

**How it is calculated**  
OneIG-Bench uses **LLM2CLIP similarity scoring** between a generated reference answer and the image.

### Style
**What it evaluates**  
Stylization fidelity for prompts that request a specific visual style.

**How it is calculated**  
OneIG-Bench mentions **CSD** and **fine-tuned CLIP embeddings** for style scoring.

### Diversity
**What it evaluates**  
Variation across multiple generations for the same prompt set.

**How it is calculated**  
OneIG-Bench uses **DreamSim pairwise similarity** (lower similarity → higher diversity).

### Overall
**What it evaluates**  
A combined score across the above dimensions. The blog reports a single overall column.

**How it is calculated**  
Typically a weighted or averaged aggregate across Alignment/Text/Reasoning/Style/Diversity; consult OneIG-Bench for the exact weighting.

---

## Notes for Implementation

- **OCR-dependent metrics (NED, Word Accuracy, Completion Rate)** require a robust OCR pipeline. Small OCR errors can dominate scores, so use consistent OCR settings across models.
- **CLIPScore** is sensitive to the chosen CLIP backbone and prompt formatting; report the model variant used.
- **OneIG-Bench** metrics are multi-dimensional; report each dimension separately in addition to the Overall score.

---

## Source of Metric List (GLM-Image Blog)

The GLM-Image blog tables explicitly list: **NED, CLIPScore, Word Accuracy (2–5 regions & average), LongText-Bench (EN/ZH), and OneIG_EN metrics (Alignment/Text/Reasoning/Style/Diversity/Overall).**
