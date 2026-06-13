# Literature & References

## Core papers used in this simulator

- **Robertson, Hollmann, Awad & Hutter (2024)** — *FairPFN: Transformers Can do Counterfactual Fairness.*
  Introduces counterfactual fairness via Prior-Fitted Networks (PFNs) and in-context learning.
  Used for: counterfactual path mechanics, fairness evaluation.
  `https://doi.org/10.48550/arXiv.2407.05732`

- **Kozodoi, Lessmann, Alamgir, Moreira-Matias & Papakonstantinou (2025)** — *Fighting Sampling Bias:
  A Framework for Training and Evaluating Credit Scoring Models.*
  European Journal of Operational Research 324 (2025) 616–628.
  Used for: reject inference (MAR/MNAR), Bayesian evaluation framework, scorecard bias.
  `https://doi.org/10.1016/j.ejor.2025.01.040`

- **Gerlin, Peng, Chen & Lessmann (2026)** — *Transfer Learning for Loan Recovery Prediction
  under Distribution Shifts with Heterogeneous Feature Spaces.*
  Humboldt-Universität zu Berlin / National University of Singapore. Unpublished preprint.
  Used for: LGD recovery curve, Beta-distributed LGD/EAD generation, covariate & label shift
  simulation framework, heterogeneous feature schema mechanics.

## Additional references

- **Kozodoi, Lessmann et al. (2022)** — *Fairness in credit scoring: Assessment, implementation
  and profit implications.* European Journal of Operational Research 297 (2022).
  `https://doi.org/10.1016/j.ejor.2021.06.023`

- **Pan & Yang (2010)** — *A Survey on Transfer Learning.*
  IEEE Transactions on Knowledge and Data Engineering 22 (10): 1345–1359.
  Taxonomy of distribution shift types (covariate, conditional, label shift).

- **Basel Committee on Banking Supervision** — *Basel III: Finalising post-crisis reforms (2017).*
  Regulatory basis for PD, LGD, EAD, CCF formulas.

---
> All feature weights and shift parameters are set via the GUI — no coefficients are hard-coded.
> The JSON catalog (`catalog.json`) persists all feature definitions, including distribution types.
