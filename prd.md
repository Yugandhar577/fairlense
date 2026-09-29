# Product Requirements Document (PRD)

## FairLens: A Comparative Framework for Bias Detection and Mitigation in Machine Learning Models

| | |
|---|---|
| **Document Owner** | Team FairLens (Yugandhar Paulbudhe, Vedanti Raut, Palash Sahuji, Sara Ansingkar, Raunak Shah) |
| **Guide** | Prof. Dhananjay Bhagat |
| **Institution** | Vishwakarma Institute of Technology, Pune — Dept. of CS & AI |
| **Course** | Ethical and Responsible AI, TY B.Tech (AI), Div. E |
| **Academic Year** | 2026–2027 |
| **Version** | 1.0 |
| **Status** | Draft |

---

## 1. Purpose

FairLens is a research and educational framework that lets its users train standard machine learning classifiers on the UCI Adult Income dataset, measure how those classifiers treat different demographic groups, apply bias mitigation techniques, and compare predictive performance against fairness before and after mitigation. The goal of this document is to define **what** the product must do and **why**, so that the functional specification and technical design can translate it into **how**.

FairLens is not a production fairness-auditing product; it is a reproducible, transparent, small-scale experimental framework intended to demonstrate responsible-AI principles in practice and to serve as a foundation for a future research paper/publication.

---

## 2. Background & Problem Statement

Machine learning systems are increasingly used to support decisions in recruitment, lending, education, and social services. These systems are typically evaluated using conventional metrics such as accuracy, precision, recall, and F1-score. A model can score well on all of these while still producing systematically different outcomes for different demographic groups (e.g., by gender or age), because it has learned historical or representational biases present in its training data.

There is no single, universally correct definition of "fairness" — different fairness criteria (demographic parity, equal opportunity, equalized odds, disparate impact) can disagree about whether the same model is fair, and improving one criterion can worsen another or reduce accuracy. Existing toolkits (AI Fairness 360, Fairlearn) and surveys provide the building blocks for fairness analysis, but there is a gap for a **compact, reproducible, side-by-side comparative study** that applies these concepts consistently across multiple standard classifiers on one dataset and explicitly quantifies the accuracy–fairness trade-off.

---

## 3. Goals & Objectives

### 3.1 Product Goals
1. Provide a working pipeline that trains multiple classifiers on a public dataset and evaluates them for demographic bias.
2. Quantify bias using multiple, well-established fairness metrics rather than a single number.
3. Implement and evaluate bias mitigation techniques, and make the resulting trade-offs (fairness gained vs. accuracy lost) visible and interpretable.
4. Present all of the above through an accessible dashboard so that a non-specialist user can explore the results interactively.
5. Produce a defensible, well-documented academic artifact (code + report) suitable for course submission and potential publication.

### 3.2 Success Criteria (Definition of Done)
- All three models (Logistic Regression, Decision Tree, Random Forest) are trained and evaluated on the Adult Income dataset.
- At least four fairness metrics (demographic parity, equal opportunity, equalized odds, disparate impact) are computed per model, per sensitive attribute (gender, age group).
- At least two mitigation techniques (reweighing, threshold adjustment) are implemented and produce measurable before/after comparisons.
- A functioning Streamlit dashboard allows model selection, metric viewing, mitigation toggling, and baseline-vs-mitigated comparison.
- A responsible-AI risk assessment (fairness, privacy, transparency, accountability, safety, human oversight) is documented and reflected in the product's design and disclaimers.
- All deliverables listed in Section 9 are complete and reproducible from the submitted codebase.

---

## 4. Target Users / Personas

| Persona | Description | Needs from FairLens |
|---|---|---|
| **Course Evaluator / Guide** | Faculty assessing the project for technical rigor and responsible-AI understanding | Clear metrics, reproducibility, documented ethical analysis |
| **ML Practitioner / Student Researcher** | Someone learning or teaching fairness concepts | Ability to compare models and mitigation techniques side-by-side |
| **Curious Analyst / Demo Viewer** | A non-technical or semi-technical viewer during project demonstration | Simple, visual, self-explanatory dashboard with plain-language explanations |

FairLens is explicitly **not** designed for: real-world deployment on live/PII data, use as an autonomous decision-making system, or use in high-stakes production settings.

---

## 5. Scope

### 5.1 In Scope
- Binary classification fairness analysis using the UCI Adult Income dataset (48,842 instances, 14 features).
- Sensitive attributes: **gender** and **age group** (derived).
- Models: Logistic Regression, Decision Tree, Random Forest.
- Fairness metrics: demographic parity, equal opportunity, equalized odds, disparate impact.
- Mitigation techniques: reweighing (pre-processing) and threshold adjustment (post-processing); sampling-based mitigation as a stretch goal.
- Performance metrics: accuracy, precision, recall, F1-score, ROC-AUC.
- A Streamlit-based demonstration dashboard.
- Documentation: technical report, ethical/responsible-AI risk assessment, reproducible experiment configuration.

### 5.2 Out of Scope
- Development of new/novel ML algorithms.
- Collection or use of personally identifiable real-world data.
- Deployment as an autonomous or high-stakes production decision system.
- Determining whether any specific real individual has personally experienced discrimination.
- Support for datasets other than Adult Income (unless added later as an extension).
- Real-time or streaming inference; the system is a batch analysis/demo tool.

---

## 6. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | System shall ingest and preprocess the Adult Income dataset (missing-value handling, categorical encoding, numerical scaling, age-group binning, train/test split). | Must |
| FR-2 | System shall train Logistic Regression, Decision Tree, and Random Forest classifiers on the preprocessed data. | Must |
| FR-3 | System shall compute standard performance metrics (accuracy, precision, recall, F1, ROC-AUC) for each trained model. | Must |
| FR-4 | System shall compute demographic parity, equal opportunity, equalized odds, and disparate impact for each model, broken down by gender and by age group. | Must |
| FR-5 | System shall apply reweighing as a pre-processing mitigation technique and retrain affected models. | Must |
| FR-6 | System shall apply threshold adjustment as a post-processing mitigation technique on trained model outputs. | Must |
| FR-7 | System shall recompute performance and fairness metrics after each mitigation technique and allow direct before/after comparison. | Must |
| FR-8 | Dashboard shall allow the user to select a model to inspect. | Must |
| FR-9 | Dashboard shall display demographic distribution of the dataset/predictions. | Must |
| FR-10 | Dashboard shall display fairness metrics per selected model and sensitive attribute. | Must |
| FR-11 | Dashboard shall allow the user to toggle/apply a mitigation technique and view its effect. | Must |
| FR-12 | Dashboard shall visualize the accuracy–fairness trade-off (e.g., baseline vs. mitigated, across models). | Must |
| FR-13 | System shall support sampling-based mitigation as an additional technique. | Should (stretch) |
| FR-14 | System shall log/export experiment configurations and results for reproducibility. | Should |
| FR-15 | Dashboard shall present plain-language disclaimers about the tool's intended use as a research/decision-support tool, not an autonomous decision-maker. | Must |

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Reproducibility** | Given the same dataset and configuration (random seed, split ratio), results must be reproducible across runs. |
| **Interpretability** | All metrics shown must be labeled clearly with plain-language explanations of what they mean, given the target audience includes non-specialists. |
| **Performance** | Full pipeline (preprocessing → training all 3 models → metric computation) should complete within a few minutes on a standard laptop CPU, given dataset size (~49K rows). |
| **Usability** | Dashboard must be usable without ML expertise; navigation should require no more than a few clicks to reach any comparison view. |
| **Maintainability** | Code shall be modular (data, models, fairness, mitigation, UI as separate components) to support future extension to other datasets/models. |
| **Privacy** | No personally identifiable or re-identifiable data shall be used or stored; dataset is public and anonymized. |
| **Transparency** | All sensitive attributes, fairness definitions, and mitigation methods used shall be explicitly documented and visible to the dashboard user. |
| **Portability** | System shall run locally via Python/Streamlit without requiring cloud infrastructure or external paid services. |

---

## 8. Assumptions & Constraints

**Assumptions**
- The Adult Income dataset is representative enough for demonstrating fairness concepts, despite its known age (1994 US Census data) and limitations.
- Gender in the dataset is recorded as a binary attribute; this is a known dataset limitation, not a design choice endorsed by the team.
- Users of the dashboard have at least basic familiarity with ML terms (accuracy, precision, etc.), though metrics will still be explained.

**Constraints**
- Academic timeline: project must be completed within the semester (see Section 10).
- Team must use open-source, free tools only (Scikit-learn, Fairlearn/AIF360, Pandas, NumPy, Matplotlib/Seaborn, Streamlit).
- Computational resources limited to student laptops / free-tier environments (Jupyter/VS Code); no large-scale compute assumed.
- Deliverable must satisfy VIT Pune's ERA course synopsis/report formatting and content requirements.

---

## 9. Deliverables

1. Working FairLens fairness-analysis prototype (codebase).
2. Source code and technical documentation.
3. Preprocessed experimental dataset and reproducible experiment configuration.
4. Baseline and bias-mitigated machine learning models.
5. Test results containing predictive performance and fairness metrics.
6. Accuracy–fairness comparison graphs and tables.
7. Ethical-risk and responsible-AI analysis document.
8. Project presentation and live demonstration (Streamlit dashboard).

---

## 10. High-Level Timeline

| Phase | Activities | Timeframe |
|---|---|---|
| Phase 1 | Literature survey & problem definition | Week 1 |
| Phase 2 | Design & data collection/preprocessing | Week 2 |
| Phase 3 | Implementation: model development, fairness metrics, mitigation | Week 3 |
| Phase 4 | Testing & evaluation, dashboard build-out | Week 4 |
| Phase 5 | Documentation & final demonstration | Week 4+ |

---

## 11. Risks & Mitigations (Product-Level)

| Risk | Impact | Mitigation |
|---|---|---|
| Fairness metrics conflict with each other (improving one worsens another) | Confusing or seemingly inconclusive results | Present multiple metrics transparently as an intentional trade-off analysis, not a single "fair/unfair" verdict |
| Mitigation reduces accuracy significantly | Perceived as reducing model "usefulness" | Explicitly frame and visualize the accuracy–fairness trade-off as the core research contribution |
| Dataset limitations (age, binary gender, US-specific) | Limits generalizability | Document dataset limitations explicitly in report and dashboard disclaimers |
| Users misinterpret dashboard output as a real-world fairness certification | Ethical/reputational risk | Mandatory disclaimers (FR-15); framed strictly as a research/decision-support tool |
| Timeline slippage given multiple mitigation techniques planned | Incomplete deliverables | Treat sampling-based mitigation (FR-13) as a stretch goal, not a blocker |

---

## 12. Out-of-the-Box Success Metrics (Project Evaluation)

- Successful training and evaluation of all 3 required models.
- Demonstrable, quantified bias detected in at least one baseline model on at least one sensitive attribute.
- Demonstrable reduction in fairness disparity after mitigation, with corresponding accuracy trade-off reported.
- Positive faculty evaluation on responsible-AI depth (ART: Accountability, Responsibility, Transparency) per course rubric.
- Fully reproducible results from the submitted repository.

---

## 13. References

Refer to `References` section of the project synopsis (Bellamy et al. 2018 — AI Fairness 360; Das, Stanton & Wallace 2023; Hort et al. 2023; Calegari et al. 2023; Weerts et al. 2023 — Fairlearn), IEEE style, to be maintained consistently across all project documents.
