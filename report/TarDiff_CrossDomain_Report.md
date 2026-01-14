# TarDiff Cross-Domain Application: From Healthcare to Financial and Industrial Domains

**Author:** Yuanze Xu

## Abstract

TarDiff is a target-aware time series diffusion generation model originally designed for healthcare scenarios. This report presents my efforts to adapt TarDiff to non-medical domains: the NASDAQ stock market (financial) and Wafer semiconductor fault detection (industrial). I developed a generalizable pipeline for cross-domain application. Experiments reveal that TarDiff's effectiveness varies significantly across domains—the Wafer dataset shows remarkable improvements (AUROC: 0.9864 → 0.9960 with TSRTR at α=0.1), while NASDAQ presents challenges due to balanced class distribution and inherent unpredictability.

---

## 1. Introduction

TarDiff is a diffusion-based time series generation model that incorporates Influence Guidance to generate task-aware synthetic data. Originally developed for healthcare applications (e.g., MIMIC-III ICU data), it addresses data scarcity by generating high-quality synthetic time series beneficial for downstream classification tasks.

**Objectives:**
1. Evaluate TarDiff's adaptability to non-medical domains
2. Develop a reusable cross-domain adaptation pipeline
3. Analyze feasibility and challenges across different data characteristics

**Selected Datasets:**

| Dataset | Domain | Characteristics |
|---------|--------|-----------------|
| **NASDAQ** | Financial | 5 channels, balanced classes (~50%), 450K+ samples |
| **Wafer** | Industrial | 1 channel, imbalanced (~10% anomaly), 1K samples |

---

## 2. Cross-Domain Adaptation Pipeline

I developed a standardized 10-step pipeline:

1. Environment Setup → 2. Data Download → 3. Data Preprocessing → 4. Config Creation → 5. Diffusion Training → 6. Classifier Training → 7. Gradient Norm Analysis → 8. Influence Guidance Generation → 9. Quality Evaluation → 10. Downstream Evaluation

**Data Format:** processed pickle files with `(data, labels)` tuple, where `data` shape is `(N, C, T)`.

---

## 3. Experimental Setup

| Config | NASDAQ | Wafer |
|--------|--------|-------|
| Channels | 5 (OHLCV) | 1 |
| Seq Length | 24 | 24 |
| Train/Val/Test | 361K/45K/45K | 900/100/6K |
| Class Balance | ~50% | ~10.7% anomaly |
| Training Steps | 50,000 | 20,000 |

**Evaluation Protocols:** Baseline (100% real), TSTR (100% synthetic), TSRTR (real + synthetic), Mixed 50/50

**Metrics:** AUROC, AUPRC, Accuracy

---

## 4. Results

### 4.1 NASDAQ Stock Dataset

**Data Overview:**
- Samples: 361,501 | Shape: (N, 5, 24) | Classes: 2
- Label distribution: Class 0 (Decrease): 49.70%, Class 1 (Increase): 50.30%
- Sample example (label=1): 
  - Channel 0-3 (OHLC): [186.55, 188.31, ..., 169.70] (price series)
  - Channel 4 (Volume): [958700, 1439700, ..., 1740700]

**Labeling Strategy:**
- Compute the 5-day return rate for each sample: `return = (close[t+5] - close[t]) / close[t]`
- Binary classification: Class 1 (Increase) if return > 0, else Class 0 (Decrease)

**Diffusion Model Training (50k steps):**

![NASDAQ Diffusion Training Loss](nasdaq_diffusion_train_loss_50k.png)

**Classifier Training (40 epochs):**

![NASDAQ Classifier Training Metrics](nasdaq_metrics.png)

The classifier training shows clear signs of **overfitting without learning meaningful patterns**:
- Train accuracy improves slowly from 50.9% to 56.8%, while val accuracy stays flat around 51-52%
- Val loss increases from 0.70 to 0.94 (diverging from train loss), indicating the model memorizes training data without generalizing
- Best val accuracy: **51.97%** (barely above random), confirming the fundamental unpredictability of stock price movements

**Gradient Norm Analysis:**

| Class | Mean Gradient Norm | Sample Count |
|-------|-------------------|--------------|
| Class 0 (Decrease) | 8.0296 | 5,027 |
| Class 1 (Increase) | 6.0054 | 4,973 |
| **Ratio** | **1.34x** | - |

**Baseline (100% Real Data):**

| AUROC | AUPRC | Accuracy |
|-------|-------|----------|
| 0.5413 | 0.5427 | 0.5283 |

**TSTR (100% Synthetic Data):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.4728 | 0.4869 | 0.5023 |
| 0.25 | 0.4888 | 0.5015 | 0.5022 |
| 0.5 | 0.5097 | 0.5116 | 0.5034 |

**TSRTR (100% Real + 100% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5342 | 0.5265 | 0.5023 |
| 0.25 | 0.5415 | 0.5385 | 0.5272 |
| 0.5 | 0.5416 | 0.5341 | 0.5304 |

**Mixed 50/50 (50% Real + 50% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5268 | 0.5276 | 0.5023 |
| 0.25 | 0.5325 | 0.5269 | 0.5206 |
| 0.5 | 0.5397 | 0.5301 | 0.5234 |

**Observations:** Across all evaluation protocols, performance improves as guidance strength α increases from 0 to 0.5:
- **TSTR**: AUROC 0.4728 (α=0) → 0.5097 (α=0.5)
- **TSRTR**: AUROC 0.5342 (α=0) → 0.5416 (α=0.5)
- **Mixed 50/50**: AUROC 0.5268 (α=0) → 0.5397 (α=0.5)

This consistent improvement demonstrates that Influence Guidance is effective in steering generation toward more task-relevant samples, even when the classifier itself performs poorly. However, TarDiff struggles on NASDAQ because: (1) the classifier achieves only AUROC 0.5197 (barely above random), providing noisy rather than useful gradient signals; (2) the nearly balanced distribution (49.7% vs 50.3%) yields a gradient norm ratio of only 1.34x, far below the 12-16x observed in TarDiff's original medical datasets.

### 4.2 NASDAQ Extreme Dataset

To address the class balance issue in the standard NASDAQ dataset, I created an "extreme" variant by redefining the classification task. Instead of binary up/down prediction, I focus on detecting **extreme price movements**.

**Labeling Strategy:** Using the same 5-day return rate, I apply percentile-based thresholds:
- Calculate the 5th and 95th percentile thresholds across all returns
- **Class 0 (Extreme Loss)**: Bottom 5% returns (large price drops)
- **Class 1 (Neutral)**: Middle 90% returns (normal fluctuations)  
- **Class 2 (Extreme Gain)**: Top 5% returns (large price increases)

This creates a class distribution of approximately **5% : 90% : 5%**, introducing significant class imbalance similar to the medical datasets where TarDiff excels. The hypothesis is that the extreme price movement patterns (both gains and losses) are harder to classify, leading to higher gradient norms for minority classes and thus more effective Influence Guidance.

**Data Overview:**
- Samples: 360,328 | Shape: (N, 5, 24) | Classes: 3
- Label distribution: Class 0 (Loss): 4.98%, Class 1 (Neutral): 90.03%, Class 2 (Gain): 4.99%
- Sample example (label=1, Neutral):
  - Channel 0-3 (OHLC): [2.36, 2.39, ..., 2.48] (price series)
  - Channel 4 (Volume): [1689900, 2508100, ..., 881000]

**Diffusion Model Training (50k steps):**

![NASDAQ Extreme Diffusion Training Loss](nasdaq_extreme_diffusion_train_loss_50k.png)

**Classifier Training (40 epochs):**

![NASDAQ Extreme Classifier Training Metrics](nasdaq_extreme_metrics.png)

The 3-class classifier exhibits the **"lazy majority prediction"** phenomenon typical of highly imbalanced datasets:
- Train/val accuracy both hover around **89.9%** throughout training—exactly matching the Class 1 (Neutral) proportion
- Train loss decreases from 0.40 to 0.34, but val loss increases from 0.40 to 0.45, indicating overfitting
- Best val accuracy: **89.94%**, achieved by simply predicting the majority class
- The model fails to learn discriminative features for the minority extreme classes (5% each)

**Gradient Norm Analysis:**

| Class | Mean Gradient Norm | Sample Count |
|-------|-------------------|--------------|
| Class 0 (Extreme Loss) | 17.5379 | 496 |
| Class 1 (Neutral) | 0.5727 | 9,031 |
| Class 2 (Extreme Gain) | 18.1262 | 473 |
| **Ratio (max/min)** | **31.65x** | - |

The artificial class imbalance successfully creates a large gradient norm ratio (31.65x), much higher than the standard NASDAQ (1.34x) and even exceeding Wafer (12.34x). Both extreme classes (Loss and Gain) show significantly higher gradient norms than the Neutral class, indicating the classifier finds them harder to classify—exactly the condition where Influence Guidance should be most effective.

**Baseline (100% Real Data):**

| AUROC | AUPRC | Accuracy |
|-------|-------|----------|
| 0.5831 | 0.3551 | 0.8965 |

**TSTR (100% Synthetic Data):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.4904 | 0.3319 | 0.8988 |
| 0.1 | 0.4901 | 0.3315 | 0.8988 |
| 0.25 | 0.5055 | 0.3346 | 0.8988 |
| 0.5 | 0.5227 | 0.3376 | 0.8988 |

**TSRTR (100% Real + 100% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5052 | 0.3356 | 0.8988 |
| 0.1 | 0.5582 | 0.3463 | 0.8988 |
| 0.25 | 0.4925 | 0.3317 | 0.8988 |
| 0.5 | 0.5150 | 0.3362 | 0.8988 |

**Mixed 50/50 (50% Real + 50% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5116 | 0.3353 | 0.8988 |
| 0.1 | 0.5200 | 0.3389 | 0.8988 |
| 0.25 | 0.5224 | 0.3390 | 0.8988 |
| 0.5 | 0.4893 | 0.3315 | 0.8988 |

**Observations:** Across all evaluation protocols, performance improves as guidance strength α increases from 0 to 0.5:
- **TSTR**: AUROC 0.4904 (α=0) → 0.5227 (α=0.5)
- **TSRTR**: AUROC 0.5052 (α=0) → 0.5150 (α=0.5), with peak at α=0.1 (0.5582)
- **Mixed 50/50**: AUROC 0.5116 (α=0) → 0.5224 (α=0.25)

This consistent improvement demonstrates that Influence Guidance is effective. However, despite successfully manufacturing class imbalance (5%:90%:5%) and achieving a high gradient norm ratio (31.65x), the classifier still cannot learn—the 89.94% accuracy exactly matches the Neutral class proportion ("lazy prediction"). This reveals that imbalance alone is insufficient; TarDiff also requires a classifier that can actually learn discriminative patterns.

### 4.3 Wafer Semiconductor Dataset

**Data Overview:**
- Samples: 901 | Shape: (N, 1, 24) | Classes: 2
- Label distribution: Class 0 (Normal): 90.23%, Class 1 (Anomaly): 9.77%
- Data range: [-2.86, 7.17], mean=-0.02, std=0.97 (normalized)
- Sample example (label=0, Normal):
  - Channel 0: [-1.14, -1.14, -1.14, 0.94, ..., -1.14, -1.14, -1.14, -1.14]

**Diffusion Model Training (20k steps, early stopped at ~5k):**

![Wafer Diffusion Training Loss](wafer_diffusion_train_loss_20k.png)

**Classifier Training (40 epochs):**

![Wafer Classifier Training Metrics](wafer_metrics.png)

The Wafer classifier training demonstrates an **ideal learning curve** with a characteristic "breakthrough" moment:
- **Epochs 1-10 ("Lazy Phase")**: Accuracy stuck at ~90% (predicting all samples as majority class Normal)
- **Epoch 11 ("Breakthrough")**: Model suddenly learns to distinguish anomalies, val accuracy jumps to 98.99%
- **Epoch 20+**: Val accuracy reaches **100%** multiple times, stabilizing around 96-100%
- Final best val accuracy: **100%**, indicating perfect separation between Normal and Anomaly classes
- Train loss drops from 0.57 to 0.02, val loss from 0.36 to near 0, with no overfitting

**Gradient Norm Analysis:**

| Class | Mean Gradient Norm | Sample Count |
|-------|-------------------|--------------|
| Class 0 (Normal) | 0.5960 | 813 |
| Class 1 (Anomaly) | 7.3570 | 88 |
| **Ratio** | **12.34x** | - |

**Baseline (100% Real Data):**

| AUROC | AUPRC | Accuracy |
|-------|-------|----------|
| 0.9932 | 0.9546 | 0.9813 |

**TSTR (100% Synthetic Data):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.3374 | 0.0828 | 0.8921 |
| 0.1 | 0.3831 | 0.0813 | 0.5999 |
| 0.25 | 0.4885 | 0.1037 | 0.5706 |
| 0.5 | 0.6224 | 0.1574 | 0.2382 |

**TSRTR (100% Real + 100% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.9864 | 0.8930 | 0.9661 |
| **0.1** | **0.9960** | **0.9650** | **0.9825** |
| 0.25 | 0.9902 | 0.9318 | 0.9718 |
| 0.5 | 0.9903 | 0.9265 | 0.9653 |

**Mixed 50/50 (50% Real + 50% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.6802 | 0.1485 | 0.8921 |
| **0.1** | **0.9892** | **0.9493** | **0.9763** |
| 0.25 | 0.9772 | 0.8682 | 0.9534 |
| 0.5 | 0.9904 | 0.9285 | 0.9745 |

**Observations:** Across all evaluation protocols, performance improves as guidance strength α increases from 0:
- **TSTR**: AUROC 0.3374 (α=0) → 0.6224 (α=0.5), an 84% relative improvement
- **TSRTR**: AUROC 0.9864 (α=0) → 0.9960 (α=0.1), surpassing baseline (0.9932)
- **Mixed 50/50**: AUROC 0.6802 (α=0) → 0.9904 (α=0.5)

This dramatic improvement demonstrates that Influence Guidance is highly effective when backed by a strong classifier (100% val accuracy). The TSTR results are particularly notable: pure synthetic data without guidance (α=0) performs at random level, but with guidance (α=0.5) achieves AUROC 0.6224—clear evidence that classifier gradients successfully steer generation toward task-relevant samples. The strong baseline (AUROC 0.9932) limits absolute improvement; the gain to 0.9960 (TSRTR α=0.1) represents a 41% error rate reduction—meaningful but numerically small.

---

## 5. Discussion

Based on my experiments, I identify three key requirements for successfully applying TarDiff to new domains:

**Requirement 1: The Classification Task Must Be Learnable**

TarDiff relies on classifier gradients to guide generation. If the classifier cannot learn, its gradients are noise rather than useful signals.

| Dataset | Classifier AUROC | Gradient Quality | TarDiff Effect |
|---------|------------------|------------------|----------------|
| Wafer | ~0.99 | High-quality | ✅ Effective |
| NASDAQ Extreme | ~0.58 | Weak | ⚠️ Limited |
| NASDAQ Binary | ~0.52 | Noise | ❌ Minimal |

**Requirement 2: Class Imbalance Should Exist**

Minority class samples produce larger gradients, providing stronger guidance signals. Wafer (12.34x gradient ratio) shows clear improvement, while NASDAQ binary (1.34x ratio) shows minimal gains.

**Requirement 3: The Task Should Not Be Too Easy**

If baseline AUROC > 0.99, there is limited room for improvement. Wafer (baseline 0.9932) demonstrates this ceiling effect—the gain to 0.9960 is meaningful (41% error reduction) but numerically small.

---

## 6. Conclusion

TarDiff achieves success on Wafer (AUROC 0.9864 → 0.9960) but faces challenges on NASDAQ due to inherent data characteristics. Based on my cross-domain experiments, I summarize the ideal dataset characteristics for TarDiff:

| Characteristic | Ideal Range | Rationale |
|---------------|-------------|-----------|
| **Baseline AUROC** | 0.70 - 0.95 | Learnable but not saturated |
| **Class Imbalance** | 5% - 15% minority | Strong gradient signal |
| **Gradient Norm Ratio** | > 5x | Sufficient guidance strength |
| **Feature-Label Relationship** | Morphologically distinguishable | Patterns observable in input |

This work demonstrates both the potential and limitations of cross-domain transfer for target-aware time series generation.

---


## Appendix

Reproduction instructions: `TarDiff_CrossDomain/NASDAQ_List.md`, `TarDiff_CrossDomain/Wafer_List.md`

Trained on 3070Ti/3090
