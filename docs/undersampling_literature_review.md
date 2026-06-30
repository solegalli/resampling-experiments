# Undersampling literature: evaluation-methodology catalog

A catalog of the undersampling literature for the paper's related-work / discussion,
focused on three questions per paper: (1) is there a **no-resampling control**
(comparison against a model trained on the original imbalanced data, not only
against other resamplers)? (2) which **metrics** are used? (3) is the **decision
threshold tuned**, or left at the implicit default? For the head-to-head
comparison studies, a fourth axis is added: do they report the **dispersion** of
the estimates, or only point values?

Scope is undersampling only; the oversampling side is covered by "Stop oversampling
for class imbalance," so it is not duplicated here. Entries are marked by confidence:
**verified** (claim adversarially checked against the primary source), **metadata
only** (bibliographic identity confirmed, content not), or **identified** (named and
located, full text not yet checked — do not cite the control/metric/threshold cells
without reading it).

## Catalog

| Paper | Year | Venue | Undersampler(s) | No-resampling control? | Metrics | Threshold tuned? | Link | Confidence |
|---|---|---|---|---|---|---|---|---|
| Hart | 1968 | IEEE Trans. Inf. Theory | CNN | N/A | NN error | N/A (NN rule) | [S2](https://www.semanticscholar.org/paper/7e67c9964a9defedd4f9dbe50f6e38ee58d52d62) | identified |
| Wilson | 1972 | IEEE Trans. SMC | ENN (basis of RENN, AllKNN, NCR) | N/A (not an imbalance study) | NN risk / error rate | N/A | [10.1109/TSMC.1972.4309137](https://ieeexplore.ieee.org/document/4309137/) | verified |
| Tomek | 1976 | IEEE Trans. SMC | Tomek links | N/A (not imbalance-framed) | unclear (NN error) | N/A | [10.1109/TSMC.1976.4309452](https://ieeexplore.ieee.org/document/4309452) | verified |
| Kubat & Matwin | 1997 | ICML | OSS (+ CNN, Tomek) | unclear | **G-mean** (explicitly rejects plain accuracy) | unclear | [dblp](https://dblp.org/rec/conf/icml/KubatM97.html) | verified (origin); control/threshold unclear |
| Provost & Fawcett | 2001 | Machine Learning | (accuracy critique, not an undersampler) | N/A | accuracy critique; advocates ROC | N/A | [10.1023/A:1007601015854](https://link.springer.com/article/10.1023/A:1007601015854) | verified |
| Laurikkala | 2001 | AIME (Springer LNCS) | NCR/NCL (uses ENN); vs random + OSS | **YES** (original-data baseline, explicit numbers) | accuracy, TPR, TNR, TPRC — **no AUC/F1/G-mean/MCC** | No / implicit | [10.1007/3-540-48229-6_9](https://link.springer.com/chapter/10.1007/3-540-48229-6_9) | verified (full text) |
| Mani & Zhang | 2003 | ICML workshop | NearMiss-1/2/3 | unclear | unclear | unclear | [bibbase](https://bibbase.org/network/publication/zhang-mani-knnapproachtounbalanceddatadistributionsacasestudyinvolvinginformationextraction) | identified |
| He & Garcia | 2009 | IEEE TKDE | survey (RUS, NearMiss, Tomek, CNN, OSS, NCR) | n/a (survey) | discusses skew-aware metrics | n/a | [IEEE](https://ieeexplore.ieee.org/document/5128907) | identified (named in scope; not independently verified) |
| Yen & Lee | 2009 | Expert Syst. Appl. | cluster-based undersampling | unclear | unclear | unclear | [KEEL pdf](https://sci2s.ugr.es/keel/pdf/specific/articulo/yen_cluster_2009.pdf) | identified |
| Smith, Martinez & Giraud-Carrier | 2014 | Machine Learning | Instance Hardness (basis of IHT) | unclear | unclear | unclear | [10.1007/s10994-014-5440-5](https://link.springer.com/article/10.1007/s10994-014-5440-5) | identified |
| Branco, Torgo & Ribeiro | 2016 | ACM Comput. Surv. | survey (full undersampler taxonomy) | treats **threshold adjustment as equivalent** to resampling (via Maloof 2003) | rejects accuracy; F-measure, G-mean, ROC-AUC, PR curves, IBA (no MCC) | threshold = first-class method | [10.1145/2907070](https://dl.acm.org/doi/10.1145/2907070) | verified (full text) |
| Guo et al. | 2017 | Expert Syst. Appl. | survey (527 papers) | unclear | unclear | unclear | [10.1016/j.eswa.2016.12.035](https://www.sciencedirect.com/science/article/pii/S0957417416307175) | metadata only |
| Lemaître, Nogueira & Aridas | 2017 | JMLR | imbalanced-learn (all in-scope undersamplers) | N/A (software) | ships G-mean etc. | N/A | [JMLR 18/16-365](https://jmlr.org/papers/v18/16-365.html) | verified |
| van den Goorbergh et al. | 2022 | JAMIA | RandomUnderSampler (+ ROS, SMOTE) | **YES** (no-correction baseline) | ROC-AUC + **calibration** (intercept/slope) | addresses calibration/threshold | [PMC9382395](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9382395/) | verified |
| Piccininni et al. | 2024 | J Biomed Inform | RandomUnderSampler, **NearMiss** (+ ROS, SMOTE) | **YES** (no-correction comparator) | AUROC + **calibration** | calibration focus | [10.1016/j.jbi.2024.104666](https://www.sciencedirect.com/science/article/pii/S1532046424000844) | verified |
| McDermott et al. | 2024 | NeurIPS | (metric methodology, not an undersampler) | n/a | AUROC vs AUPRC analysis | n/a | [arXiv:2401.06091](https://arxiv.org/abs/2401.06091) | verified |

## Empirical comparison studies (resamplers vs a no-resampling baseline)

These directly pit resamplers against a baseline, so all three axes plus
**dispersion reported?** apply. Added from Sole's review of PR #10; the findings
are her reading of the papers, not independently re-verified.

| Paper | Year | Venue | Methods / learners | Control? | Metrics | Threshold | Dispersion? | Finding | Link |
|---|---|---|---|---|---|---|---|---|---|
| Wainer | 2018 | arXiv | 10 US + 5 OS + ensembles + specialized; 58 datasets | **YES** (no-mitigation baseline) | accuracy, **AUC**, balanced acc, F1, G-mean, **MCC**, precision, recall | default 0.5 | **No** (one value / rank per dataset) | No single winner; depends on the metric. On **AUC, undersampling/oversampling/class-weight/baseline tie**; specialized ensembles lag. Baseline + class-weight good on AUC but poor on MCC. | [arXiv:1810.07168](https://arxiv.org/abs/1810.07168) |
| Hasanin, Khoshgoftaar et al. | 2019 | J. Big Data | RUS + ROS; 2 datasets | **YES** (baseline) | AUC | n/a (AUC) | **Yes** (dataset 1) | Dataset 1: **with uncertainty, US/OS do not improve** (matches us). Dataset 2 (**SlowlorisBig**): US/OS do improve. | [10.1186/s40537-019-0274-4](https://link.springer.com/article/10.1186/s40537-019-0274-4) |
| (class-weight medical study) | 2021 | Informatics in Medicine Unlocked | class weights vs unbalanced RF/XGBoost; 4 medical datasets (incl. Pima) | **YES** | ROC-AUC | n/a (AUC) | **No** | Class-weight **beats** the unbalanced models on ROC-AUC, notably on some datasets — but no uncertainty shown (a likely reason it disagrees with us). | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S235291482100174X) (citation to confirm) |
| Gurcan & Soylu | 2024 | Cancers | 19 resamplers (RUS, Tomek, NearMiss, ClusterCentroids, ENN, RENN, CNN, IHT, OSS, NCR + OS/hybrid) on RF, XGBoost + 8 | **YES** (baseline) | Accuracy, F1, **ROC-AUC** (mean of 3) | unstated (≈0.5) | **No** (ANOVA / Kruskal-Wallis only) | Claims resampling works (SMOTEENN + RF best); notable because it uses **strong learners** (RF/XGBoost), not only weak ones — but ROC-AUC point estimates, no dispersion. | [PMC11476323](https://pmc.ncbi.nlm.nih.gov/articles/PMC11476323/) |

## Synthesis: the methodological gap

**The undersampling literature splits into two eras, and neither closes the loop our paper needs.**

*The method papers (1968–2003) predate the modern evaluation paradigm.* Tomek (1976), Wilson (1972) and Hart (1968) are nearest-neighbour data-cleaning rules that were **not even written about class imbalance** — they were repurposed as undersamplers later. The two that are genuinely imbalance-framed, OSS (Kubat & Matwin 1997) and NCR (Laurikkala 2001), evaluate with **rate-based metrics at an implicit default threshold** (G-mean; accuracy/TPR/TNR) and never report ROC-AUC or PR-AUC or tune a probability threshold. Of all the method papers, only Laurikkala (2001) verifiably includes a **no-resampling control**, and even it reports no threshold-independent metric.

*The surveys already concede the core point.* Branco, Torgo & Ribeiro (2016) state, citing Maloof (2003), that **moving the decision threshold, applying a sampling strategy, and adjusting the cost matrix produce classifiers with the same performance** — i.e. resampling and "default-0.5-then-tune-the-threshold" are interchangeable. That is exactly our thesis (and Elkan's analytic result), already sitting in a major survey. The surveys also reject plain accuracy in favour of skew-aware metrics, but treat the undersampler taxonomy descriptively rather than testing it against a no-resampling control.

*The modern critiques supply the missing control — and the result is ours.* van den Goorbergh et al. (2022, JAMIA) and Piccininni et al. (2024, J Biomed Inform) both compare against an **uncorrected baseline** and find that random undersampling (and NearMiss) **do not improve AUROC** and **severely harm calibration**, making models that strongly overestimate the minority probability. That miscalibration is a **threshold-independent harm invisible to F1/accuracy at 0.5** — and it is the same effect we see in our own speed experiment, where naive undersampling overpredicts until the prior-restoring weights are added.

*Dispersion is the third axis, and it tracks the conclusion.* Across the empirical comparison studies, the result follows whether uncertainty is shown. The studies that report a resampling **win give point estimates with no dispersion** — Gurcan & Soylu (2024) and the class-weight medical study report ROC-AUC with no standard deviation. The studies that **account for uncertainty** reach the opposite conclusion: Hasanin et al. (2019) find undersampling and oversampling do not help once the uncertainty is shown (their first dataset), and Wainer (2018), on the threshold-independent metrics, finds resampling and the baseline tie. So a reported "win" is often an artifact of comparing point estimates, the same way it is an artifact of the default threshold — and these studies use **strong learners (RF, XGBoost)**, so it is not explained away by weak model choice.

**The gap we exploit:** most of the undersampling literature (i) omits the no-resampling AUC/PR control, (ii) reports threshold-dependent metrics at the default threshold, (iii) does not tune the threshold, and (iv) reports point estimates without dispersion — which together are precisely the confound that makes resampling look helpful. This is the undersampling analogue of the SMOTE-only-vs-SMOTE-variant control hole: methods are compared against each other, rarely against doing nothing, and rarely with the uncertainty shown.

## Honesty caveats and open items

- **~20 papers** here. The foundational/critique rows came from an adversarially-verified search; the four empirical comparison studies were added from Sole's review (her reading), and the Informatics in Medicine Unlocked row still needs its exact citation (authors/title) confirmed.
- **SlowlorisBig (Hasanin 2019) is not publicly downloadable as-is** (a derived big-data subset from the Khoshgoftaar/FAU group). The Slowloris attack is, however, labelled in the public CICIDS2017 set (UNB CIC), from which a comparable severely-imbalanced Slowloris-vs-benign dataset can be built to test reproduction.
- **Foundational cells are N/A or unclear by nature.** Tomek/Wilson/Hart have no probability threshold (NN rules) and predate AUC/PR reporting; do not upgrade those cells without reading the papers.
- **Still to verify from primary sources:** He & Garcia (2009, the survey originally named — verification anchored on Branco 2016 instead), Mani & Zhang (NearMiss 2003), Smith et al. (Instance Hardness 2014), Hart (CNN 1968), Yen & Lee (cluster-based 2009). These need their control/metric/threshold cells filled from full text.
- **Domain caveat:** the strongest no-control evidence (van den Goorbergh; Piccininni) is clinical risk-prediction with logistic/tree models and emphasises *random* undersampling + NearMiss. Generalisation to the informed undersamplers (ENN, Tomek, OSS, NCR, IHT, cluster-based) is plausible but not directly established by those two papers — a gap our own experiments help fill.

*Generated from an adversarially-verified web search (5 angles, 24 sources fetched, 25 claims voted, 20 confirmed / 5 refuted). Treat "identified" rows as leads, not citations, until the full texts are read.*
