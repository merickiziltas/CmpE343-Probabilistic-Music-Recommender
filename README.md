# 🎵 Tunes Duel: A Probabilistic Simulation of Music Recommendation Systems

![Python](https://img.shields.io/badge/Language-Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Course](https://img.shields.io/badge/Course-CmpE_343_Probabilistic_Systems-purple?style=for-the-badge)
![LaTeX](https://img.shields.io/badge/Report-LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white)
![Concept](https://img.shields.io/badge/Concept-Statistical_Modeling_%26_MLE-orange?style=for-the-badge)

---

## 📌 Executive Summary

This repository contains the complete implementation and statistical analysis for **CmpE 343 (Introduction to Probability & Statistics for CS)** project at **Boğaziçi University**. 

The project explores music recommendation systems (**Tunes Duel ♪**) through rigorous probabilistic modeling, Maximum Likelihood Estimation (MLE), non-parametric hypothesis testing, patience-adaptive recommender design, and large-scale Monte Carlo evaluation across 3,000 simulated user sessions.

Full academic paper source code is available in [`docs/report.tex`](docs/report.tex).

---

## 🌟 Four-Phase Technical Architecture

```
                                  PROBABILISTIC PIPELINE
                                  
  [Data: Tracks & Ratings] ---> [Part 1: Laplace Conditional & Bayes]
                                            │
                                            ▼
  [User Group Hypotheses]  <--- [Part 2: Beta-Geometric MLE & Mann-Whitney U]
                                            │
                                            ▼
  [Tune Duel API Entry]    <--- [Part 3: Model A (Matcher) vs Model B (Sampler)]
                                            │
                                            ▼
                                [Part 4: Monte Carlo 3000 Trials]
```

### Part 1: Conditional Probability & Bayesian Inversion
- **Global & Binned Conditional Probabilities ($P(5\star \mid F)$):** Computes smoothed probabilities across 18 audio features (artist, release year, popularity, danceability, acousticness, mood, genre) using Laplace smoothing:
  $$P(5\star \mid F = f) = \frac{\text{Count}(5\star, F=f) + 1}{\text{Count}(F=f) + 5}$$
- **Exhaustive Feature Interactions ($P(5\star \mid F_1, F_2)$):** Discovers joint feature combinations (e.g., Not-relaxed $\times$ Jazz yields $P(5\star) = 0.688$).
- **Bayesian Inversion ($P(F \mid 5\star)$):** Applies Bayes' rule to determine dominant attributes of 5-star rated tracks (e.g., Electronic genre 96.0%, Female vocals 72.0%, Non-aggressive mood 94.1%).
- **Personal vs Group Divergence:** Analyzes preference heterogeneity across team members' rating histories.

### Part 2: User Variability & Survival Modeling ($T_u$)
- Models waiting rounds $T_u$ until user $u$ discovers their first 5-star track.
- **Geometric Model (Homogeneous Assumption):** $\hat{p} = \frac{1}{\bar{T}} = 0.2707$ ($\text{Log-Likelihood} = -627.75$).
- **Beta-Geometric Model (Heterogeneous MLE):** Allows individual probabilities $p_u \sim \text{Beta}(\alpha, \beta)$. Maximizes log-beta likelihood using adaptive gradient search:
  $$\hat{\alpha} = 3.04, \quad \hat{\beta} = 5.57, \quad \text{Log-Likelihood} = -611.18$$
  *(Log-likelihood improvement of +16.56 conclusively proves user heterogeneity).*
- **Hypothesis Testing (Mann-Whitney U Test):** Tests whether popular music lovers find 5-star tracks faster than niche lovers:
  - Popular Music Lovers: $\bar{T}_A = 2.92$ rounds
  - Niche Music Lovers: $\bar{T}_B = 4.48$ rounds
  - $p\text{-value} = 0.0030$ (Highly statistically significant 35% reduction in waiting time).

### Part 3: Recommender System Design
- **Model A (Deterministic Feature Matcher):** Aggregates user sentiment across 18 features weighted by global $P(5\star \mid F)$ confidence and 5-year temporal release window smoothing.
- **Model B (Hybrid Patience-Aware Sampler - Competition Model):** Adapts the exploration-exploitation trade-off based on empirical user patience:
  $$\Delta_u = T_u - 3.6942$$
  Impatient users ($\Delta_u < 0$) receive focused exploitation, while patient users ($\Delta_u > 0$) receive flattened probabilistic sampling for broader music discovery.

### Part 4: Monte Carlo Evaluation
- **Simulation Setup:** 3,000 independent trials (300 users $\times$ 10 seed trials).
- **Evaluation Metrics:** Hit@5, Average Rating, Time-to-5$\star$, and 95% Welch Confidence Intervals.

---

## 📂 Repository Structure

```
CmpE343-Probabilistic-Music-Recommender/
├── src/
│   ├── part1.py           # Conditional Probability Modeling & Bayes' Rule
│   ├── part2.py           # User Variability Modeling (Geometric & Beta-Geometric MLE)
│   ├── part3.py           # Recommender Engine (Model A & Model B implementation)
│   ├── part4.py           # Monte Carlo Simulation Harness (3,000 trials & CIs)
│   ├── part4_cf.py        # Collaborative Filtering Model Extension
│   └── recommender.py     # Tune Duel Competition API & Interactive CLI Test
├── data/
│   ├── tracks.csv         # Track metadata & audio feature extractions
│   ├── ratings.csv        # Global user rating interactions
│   └── group.csv          # Team member session ratings
├── docs/
│   ├── report.tex         # Complete LaTeX Academic Paper Source
│   ├── project_report.pdf # Project Report
│   └── Project_Description.md# Course project specification
├── requirements.txt       # Python dependencies (pandas, numpy, scipy)
├── .gitattributes         # LF line ending configuration
└── .gitignore             # Excludes bytecode, venvs, and raw log dumps
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 1. Execute Statistical Modeling Pipeline
Run individual parts from project root:
```bash
python src/part1.py
python src/part2.py
python src/part3.py
python src/part4.py
```

### 2. Interactive Recommender CLI Test
Launch the interactive CLI to test song search and personalized recommendations:
```bash
python src/recommender.py
```

---

## 👨‍💻 Authors (Group 108-301-110)

- **Ahmet Meriç Kızıltaş** (2022400225)

*Department of Computer Engineering, Boğaziçi University*
