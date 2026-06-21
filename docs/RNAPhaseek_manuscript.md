# RNAPhaseek: a foundation-model framework for predicting and designing self–phase-separating RNA

**Amir M. Cheraghali**^(1,\*) *and colleagues* *(author list to be completed)*

^1 INSERM, France
\*Correspondence: amirmohammad.cheraghali@inserm.fr

*Draft manuscript — numbers and figures are drawn from the project results; citations marked “[verify]” require a final reference check before submission.*

---

## Abstract

Liquid–liquid phase separation (LLPS) organizes cellular biochemistry into membraneless condensates, and RNA is increasingly recognized not only as a passenger but as an autonomous driver: certain RNAs phase-separate by themselves, through RNA–RNA multivalent interactions, in the complete absence of protein. Yet while dozens of sequence-based predictors exist for protein LLPS, no published method predicts protein-free **RNA-self-LLPS** from sequence, and the de novo design of phase-separating RNA remains an entirely rational, thermodynamics-driven exercise. Here we present **RNAPhaseek**, an end-to-end framework that (i) predicts whether an RNA self–phase-separates directly from sequence using an RNA foundation model fused with graph-based structural and biophysical features, and (ii) uses that learned predictor as an oracle to generate de novo self–phase-separating RNA. On the largest strictly-curated corpus of protein-free RNA-self-LLPS sequences assembled to date (1,352 positives, supplemented with 83 mechanistically matched training pairs of G-quadruplex sequences), RNAPhaseek achieves a leakage-free 5-fold cross-validated AUROC of **0.88**, a structural-specificity AUROC of **0.90** against composition-matched, structure-disrupted controls, and a matched-pair accuracy of **1.00** on a stricter held-out benchmark of G-quadruplex pairs that share a structured core but differ only in flanking sequence. Three orthogonal generators (gradient design, genetic algorithm, and a diversity-penalized deep exploration network) produce candidate sequences scored as structurally grounded — their predicted phase-separation collapses when their structure, but not their composition, is scrambled. We are transparent about two limits: an out-of-distribution gap on designed kissing-loop controls, which we trace mechanistically to the absence of that condensation mode from natural-RNA training data; and a cross-organism generalization gap (yeast AUROC 0.90 vs. non-yeast 0.80) that reflects the severe scarcity of strict non-yeast RNA-self-LLPS data — a field-level bottleneck rather than a model defect, which we partially close with organism-balanced training. For very long transcripts that exceed the backbone's 1,022-nucleotide context window, an optional attention-MIL extension scores the full sequence by tiling overlapping windows and pooling with learned attention weights, providing per-window saliency rather than silent truncation. RNAPhaseek is released as a single command-line tool and as a one-click Colab notebook for installation-free use. To our knowledge it is the first learned predictor of protein-free RNA-self-LLPS and the first to couple such a predictor to de novo RNA-condensate design.

---

## Introduction

Biomolecular condensates formed by liquid–liquid phase separation compartmentalize processes ranging from ribosome biogenesis to stress responses and signaling [1,2]. RNA is a central component of most cellular condensates, and a growing body of in vitro work establishes that RNA can be the *driver* of phase separation, not merely a client: protein-free RNA undergoes condensation through multivalent base-pairing, G-quadruplex stacking, and repeat-encoded interactions [3–6]. Landmark examples include the protein-free self-assembly of total cellular RNA that contributes to stress-granule formation [3], repeat-expansion RNAs (e.g. CAG, CUG, G4C2) that undergo sol–gel transitions implicated in neurological disease [4], poly(UG) and G-quadruplex RNAs [5], and structured catalytic RNAs that phase-separate via lower-critical-solution-temperature behavior [6].

Despite this, the computational toolbox for RNA-self-LLPS is strikingly thin. Sequence-based prediction of *protein* LLPS is mature, with more than a dozen predictors — PScore [7], catGRANULE and catGRANULE 2.0 [8], FuzDrop, PSAP, PSPredictor, and PhaSePred [9], the last of which makes a now-standard distinction between self-assembling (“driver”) and partner-dependent proteins. For RNA, by contrast, the resources are predominantly *databases* — RNAPhaSep [10] and its expansion RPS 2.0 [11] catalogue RNAs found in condensates but provide no sequence-based model. The one published learned predictor that uses RNA-sequence features, built on the RNAPSEC dataset with an AdaBoost classifier [12], fundamentally predicts protein **+** RNA co-condensation under specified conditions and, by the authors’ own statement, cannot predict RNA-only phase separation. A recent deep classifier of “condensation-prone” RNAs (smOOPs) [13] reaches high accuracy but on a different task — membership of RNAs in *protein* ribonucleoprotein granules, leaning on protein-crosslinking and modification tracks — precisely the RNA-in-protein-condensate category that protein-free RNA-self-LLPS must be distinguished from. The most direct conceptual ancestor of RNA-self propensity scoring remains a thermodynamic heuristic: the correlation between NUPACK-computed multimer size and experimentally measured protein-free RNA self-assembly [3,14].

Generation is similarly lopsided. De novo design of phase-separating RNA is an active and impressive field, but it is almost entirely *rational*: RNA nanostars and kissing-loop droplets are engineered with NUPACK/oxDNA and validated experimentally [15–17], with no learned model of phase-separation propensity in the loop. The machine-learning sequence-design engines that could close this loop — gradient design (Fast SeqProp) [18] and deep exploration networks (DEN) [19] — and generative RNA language models [20] have been applied to regulatory elements, structure, and expression, but never to phase separation, in RNA or protein. A 2026 protein-IDR method, PhaSeMotif, couples prediction with synthetic-variant generation [21], establishing the predict-and-generate philosophy — but for composition-driven protein motifs, not RNA, and not via a classifier-gradient design loop.

We set out to fill this gap with **RNAPhaseek**: a single framework that learns to predict protein-free RNA-self-LLPS from sequence, and then turns that predictor into a generator of novel self–phase-separating RNA. Throughout, we hold a strict definition — RNA that phase-separates *by itself*, protein-free — and we treat methodological honesty (leakage control, adversarial verification, explicit out-of-distribution and organism-bias analysis) as a first-class deliverable rather than an afterthought.

---

## Results

### A foundation-model architecture for RNA-self-LLPS

RNAPhaseek encodes each RNA with the RNA-FM foundation model [22] (640-dimensional, 12-layer; the final two layers fine-tuned), augmented along two complementary axes (Figure 1; Methods). First, a lightweight FEGSTrans adapter injects a graph-based structural encoding (FEGS) as an attention bias, so the representation is sensitive to base-pairing topology, not only local sequence. Second, the pooled representation is fused with a 38-dimensional biophysical feature vector spanning RNA2PS thermodynamic parameters, RNA-binding-protein motif densities, sequence complexity, absolute repeat/periodicity content, and — added in this work — five **self-complementarity** features (reverse-complement local-alignment score, ViennaRNA paired fraction and energy, longest reverse-complement stem, and reverse-complement *k*-mer self-pairing fraction). These last features separate a palindromic, self-complementary RNA from a composition-matched scramble by 7–11σ, and are the architectural complement to the structural hard negatives described below.

### A leakage-honest evaluation

A recurring hazard in small-corpus sequence prediction is information leakage between near-duplicate sequences. We adopted CD-HIT cluster-grouped cross-validation, in which all sequences sharing ≥90% identity are constrained to the same fold, and we report this as the only honest estimate. The choice matters: ~22% of positives have a ≥95%-identical paralog or fragment, and naive per-sequence splits inflate apparent AUROC by ~2.5 points (0.847 with per-sequence grouping vs. **0.822** with CD-HIT cluster grouping on the same data) while halving fold-variance (±0.054 → ±0.025). We similarly favor high-powered cross-validation over small held-out subsets: a non-yeast generalization estimate read 0.71 on a 29-sequence locked test but 0.76–0.80 on the high-powered (n=268) cross-validation, and organism-balanced training appears to cost overall accuracy on the small test yet is cost-neutral under cross-validation. We therefore base every claim on cluster-grouped cross-validation, not point held-out tests.

### Structural specificity: learning structure, not just composition

A predictor that recognizes phase-separating RNA only by nucleotide composition is of limited value for design, where one must distinguish a structured candidate from a compositionally identical but non-functional scramble. To force the model to read structure, we generated **184 structural hard negatives**: composition-matched shuffles of positive sequences (154 exact mono+dinucleotide-preserving Eulerian shuffles, 30 mononucleotide-preserving) whose self-complementarity is destroyed (mean reverse-complement *k*-mer self-pairing 0.28 → 0.09; ~4 bp of stem removed) while composition is held exactly. Crucially, these derived negatives inherit their parent’s cross-validation group, preventing a ~composition-identical pair from straddling the train/test boundary.

On the held-out structural negatives, the model distinguishes condensing positives from their structure-destroyed twins at AUROC **0.90** (pos-vs-structural-negative, cross-validated) — up from a coin-flip ~0.50 in a composition-driven baseline. This is the core capability that makes the downstream design trustworthy: the model’s confidence reflects structure, not merely content.

Beyond composition-matched shuffles, we further strengthened structure-specificity with **83 mechanistically matched training pairs** of synthetic G-quadruplex sequences. Each pair shares an identical G-tract core but differs in its flanking context: an A-rich flank that leaves the G-tract free to self-assemble (positive) is paired with a complementary C/U-rich flank that base-pairs the G-tract and abolishes condensation (negative). The 83 pairs span 20 diverse G4 cores, CD-HIT cluster-grouped against the rest of the corpus and de-leaked against an external held-out benchmark (maximum 8-mer Jaccard against benchmark sequences = 0.33). These pairs teach the model the general “free vs. sequestered G-tract” principle in a way that composition-matched shuffles alone cannot, because they preserve *both* composition and G-content while flipping only the structural context of the G-tract. On the external held-out benchmark — 9 pairs of G-quadruplex RNAs that share a G4 core but differ in flanking sequence (one element of each pair phase-separates, the other does not) — the model correctly ranks **all 9 pairs** (matched-pair accuracy 1.00, mean margin +0.130, AUROC 0.812 on the 18 sequences). This stricter test, which a composition- or structure-disruption-only model fails at chance, is the most demanding evidence we report that RNAPhaseek reads structure rather than content.

### An out-of-distribution limit, mechanistically explained

We assembled a fully independent external test from two designed-RNA-condensate studies [16,17] — 27 condensing nanostars and 2 “scrambled-kissing-loop” negatives (A̅, B̅) that are ~94% identical to their condensing parents but lose the 6-nucleotide loop palindrome that drives intermolecular pairing — de-leaked at 80% identity against training. The frozen model recognized the independent positives well (100% recall, mean P=0.88) but **failed to reject the two scrambled controls** (0.92, 0.90). Rather than leave this as an unexplained failure, we diagnosed it: a dedicated kissing-loop complementarity feature cleanly separates A from A̅ (loop-loop complementarity 6 vs 2), yet that same feature carries essentially no label signal in our corpus, because **the kissing-loop condensation mechanism is absent from natural-RNA training data** — every natural positive scores like A̅ (≈2), not like A (=6) (Figure 4). The external gap is therefore a domain-shift limit of the field’s data, not a feature-resolution or model defect; closing it would require designed-nanostar sequences in training, which would in turn forfeit them as an independent test.

### Corpus composition and the yeast-generalization diagnostic

The final positive corpus comprises **1,352** sequences, dominated by the protein-free yeast RNA self-assembly screen of Van Treeck et al. [3] re-mined to its full enriched depth (911 yeast ORFs, CD-HIT-de-leaked), and supplemented with a verified, anti-fabrication-audited set of non-yeast sequences (archaeal RNase P RNAs, an Arabidopsis G-quadruplex mRNA, designed sticker–spacer and rG4 constructs). An organism-stratified diagnostic reveals an honest limit: the model is a strong predictor of **yeast** RNA-self-LLPS (AUROC 0.90) with more moderate transfer to other organisms (~0.76 under a uniform sampler). This is volume, not diversity — ~67% of positives are *S. cerevisiae* from a single assay, because strict non-yeast RNA-self-LLPS data barely exists.

We partially closed this gap with **organism-balanced training**: a sampler that draws non-yeast positives as often as yeast (P(neg)=0.50, P(yeast-pos)=0.25, P(non-yeast-pos)=0.25). Under matched cluster-grouped cross-validation, organism balancing lifts non-yeast AUROC by **+0.036** (n=268) at no cost to overall AUROC, narrowing the yeast/non-yeast gap by ~30% (Figure 5). RNAPhaseek combines the strict corpus, structural hard negatives, mechanistically matched G4 training pairs, self-complementarity features, CD-HIT cluster-grouped leakage control, and organism-balanced sampling, delivering: overall AUROC **0.875**, structural-specificity **0.897**, yeast **0.898**, non-yeast **0.803**.

### De novo design of structurally-grounded self–phase-separating RNA

Using the trained model as a fitness oracle, we implemented three generators (Methods): gradient design (Fast SeqProp), a genetic algorithm (GA) scoring candidates through the complete pipeline, and a diversity-penalized deep exploration network (DEN). To distinguish genuine designs from composition artifacts, we introduced a **structure-dependence** test — the difference between a sequence’s predicted P(LLPS) and that of its composition-matched scramble (Δ). The GA converged to a single, maximally-optimized motif (P=0.996) that is the most structure-grounded set produced (Δ=+0.52; score 0.995 collapsing to 0.48 when scrambled) — exceeding even real LLPS positives (Δ=+0.28). The DEN produced a diverse library (15 distinct designs, mean pairwise identity 0.46 vs. the GA’s 0.82) that remains as structure-dependent as real positives (Δ=+0.28), trading per-design optimization for breadth (Figure 6). Random RNA, as expected, is structure-independent (Δ=−0.13) and scored near zero. The two generators are complementary: the GA supplies a single high-confidence anchor, the DEN a diverse panel that the structure-dependence test can triage. These are model-believed candidates awaiting experimental validation.

### A usable tool

RNAPhaseek is released as a single command-line program with three subcommands — `score` (P(LLPS) per FASTA), `design` (GA or DEN generation), and `validate` (structure-dependence triage) — together with a one-click Colab notebook that downloads the trained weights from the Hugging Face Hub and exposes scoring and design in the browser, with no local installation. Prediction requires only the model and computes structural features on the fly; the full training cache is not needed for inference.

For RNAs exceeding the **1,022-nucleotide context window** of the backbone, the `score` command provides an **attention-MIL** extension (`--long-model mil`) that handles full-length sequences without truncation. The MIL scorer tiles overlapping 1,022-nt windows (stride 512) across the input, scores each window through the same architecture, and combines window-level scores via attention-weighted pooling, yielding both a single per-sequence P(LLPS) and a per-window saliency vector that localizes the signal. By default, sequences longer than 1,022 nt are truncated to their first 1,022 nt — a held-out comparison showed no accuracy loss for this task — and the MIL mode is reserved for cases where transparent full-length scoring or per-window interpretability is wanted.

---

## Discussion

RNAPhaseek occupies an intersection that, to our knowledge, no published method has: a learned, foundation-model predictor of *protein-free RNA-self-LLPS*, used as an oracle to *generate* de novo self–phase-separating RNA, with a structure-dependence trustworthiness check. Each component has real precedent — the predictor’s nearest neighbors are the protein-dependent RNAPSEC/AdaBoost model [12] and the RNP-granule classifier smOOPs [13]; its target descends from the Van Treeck protein-free screen [3]; its generators are borrowed from Linder and colleagues [18,19]; and the predict-then-generate framing was demonstrated for protein IDRs by PhaSeMotif [21]. But their *combination* for RNA-self-LLPS is, as far as we can establish, unoccupied. The honest claim is therefore one of method novelty (first to do this for RNA), not benchmark superiority: there is no existing RNA-self-LLPS predictor or generator to outperform, and the AUROCs of adjacent tools (smOOPs ~0.94 on RNP-granule membership; RNAPSEC ~0.67 on protein+RNA+conditions) are not comparable to ours because they address different tasks with different label definitions.

Two limitations are intrinsic and we state them plainly. First, the kissing-loop out-of-distribution gap: the model cannot reject designed scrambled-loop controls because that condensation mode is absent from natural-RNA training data — a property of the field’s data, addressable only by bringing designed-nanostar sequences into training at the cost of the independent test. Second, the cross-organism generalization gap: strict non-yeast RNA-self-LLPS data is so scarce that the corpus is yeast-dominated; organism-balanced training narrows but does not eliminate the gap, and the most impactful next step is experimental — generating new strict non-yeast data — rather than computational. Finally, all designs and predictions here are model-believed; wet-lab validation of the candidate panels (turbidity, microscopy, FRAP) is the essential next step to convert prediction into evidence.

Beyond validation, two extensions follow naturally: incorporating experimental conditions (salt, divalent cations, concentration), which the closest competitor [12] showed to matter and which our model currently ignores; and positive-unlabeled learning, since the “negatives” are unverified and therefore truly unlabeled. RNAPhaseek provides the predictor, the generators, the strict corpus, and the tool to make all of this tractable.

---

## Materials and Methods

**Strict corpus and curation.** Positives are RNAs with evidence of protein-free, RNA-driven phase separation: deproteinized-transcriptome self-assembly hits [3], repeat-expansion and G-quadruplex RNAs [4,5], catalytic RNAs with LCST behavior [6], and validated in vitro RNA-only droplets. RNAs merely present in protein condensates (the RNAPhaSep/RPS membership category [10,11]) were excluded. Sequences were obtained from primary literature, RNAPhaSep/RPS, and the SGD/Ensembl genome resources; FASTAs were normalized (T→U). The final pool comprised 1,352 positives, 641 matched/transcript negatives, 184 structural hard negatives, and 83 mechanistically matched G-quadruplex training pairs (Methods: matched training pairs). Approximately 67% of positives are *S. cerevisiae* (single-screen origin). Designed-nanostar sequences [16,17] were reserved as a frozen external test and excluded from all training.

**Matched training pairs.** To force the model to learn the structural distinction between free and sequestered G-tracts (a paper-internal mechanism that composition-matched shuffles cannot teach), we generated 83 synthetic pairs covering 20 diverse G4 cores. Each pair shares an identical G-tract core (e.g. (G3An)4 with varied tract count and spacer length) but differs in flanking sequence: the positive pairs the core with an A-rich flank that leaves the G-tract free; the negative pairs the same core with a complementary C/U-rich flank that base-pairs the G-tract and abolishes condensation. The pair generator additionally produced spacer-disruption negatives, tract-count negatives, and reconstructions of real motif pairs (poly-UG, homopolymers, repeat thresholds). All pairs were de-leaked against the external held-out matched-pair benchmark by 8-mer Jaccard (maximum kept = 0.33) and CD-HIT cluster-grouped so a positive/negative pair could not straddle the train/test boundary.

**Dataset expansion.** The Van Treeck 2018 enrichment table (GEO GSE99170) was re-mined at the paper’s threshold (P<0.01, fold-change ≥2; FC≥2.5 tier used), yielding 911 net-new yeast ORFs after CD-HIT de-leaking. A verified, anti-fabrication-audited diversity set (RNase P RNAs from three archaea, Arabidopsis SHR-GQ and matched mutants, designed rG4/sticker–spacer constructs, poly(UG) and homopolymers) was added; sources whose sequences existed only as un-transcribable figures were excluded rather than reconstructed.

**Structural hard negatives.** For each structured positive, composition-matched negatives were generated by Altschul–Erickson dinucleotide-preserving Eulerian shuffling (exact mono+dinucleotide match) with a mononucleotide-shuffle fallback, accepted only if self-complementarity (reverse-complement *k*-mer fraction, longest stem) dropped relative to the parent. Each negative inherited its parent’s CD-HIT cluster group, and the set was CD-HIT-de-leaked against the external test.

**Architecture and features.** The backbone is RNA-FM (multimolecule/rnafm) [22], final two layers fine-tuned; a 2-layer FEGSTrans adapter applies a FEGS-derived structural encoding as graph-bias attention [FEGS ref — verify]. Biophysical features (38-dim, including five self-complementarity features computed over an 800-nt window with ViennaRNA [23] where applicable) are computed per RNA and fused after attention pooling, then passed to a multilayer-perceptron classifier head. Sequences longer than the backbone’s 1,022-nucleotide context window are truncated to their first 1,022 nt by default — a held-out comparison on a balanced long-RNA subset showed no accuracy loss versus full-length scoring. For applications where full-length scoring or per-window interpretability is required, an attention-MIL extension tiles 1,022-nt windows across the full sequence with stride 512, scores each window through the same backbone+adapter+biophysics pipeline, and combines window-level scores with a learned attention-weighted pool; this extension is exposed via the CLI as `score --long-model mil`.

**Training and evaluation.** The model was trained with AdamW, early stopping on an inner-validation AUROC, label smoothing, and a class-and-organism-balanced WeightedRandomSampler (P(neg)=0.50, P(yeast-pos)=0.25, P(non-yeast-pos)=0.25). Evaluation used 5-fold StratifiedGroupKFold with CD-HIT clusters as groups (matched and structural negatives following their parent), a grouped held-out locked test, and the frozen external nanostar and matched-pair G-quadruplex benchmarks. A feature-ablation control (zeroing the self-complementarity features) and a determinism canary (asserting identical featurization at train and test time) were applied. The released model was retrained on all data (~2,049 of 2,258 sequences; ~10% grouped inner-validation for early stopping); its performance is characterized by the leakage-free cross-validation reported in the Results.

**Generation.** Three generators optimize the model’s P(LLPS): Fast SeqProp [18] (gradient design through a differentiable proxy), a genetic algorithm (population 64, full-pipeline fitness, two-point crossover, tournament selection), and a DEN [19] (a generator network trained with a Gumbel-softmax relaxation and a pairwise-similarity diversity penalty). Final designs are re-scored through the complete pipeline.

**Structure-dependence validation.** Each candidate is scored against k=3 composition-matched scrambles (dinucleotide-preserving where possible); Δ = P(design) − mean P(scramble). Δ>0 indicates a structure-driven score; Δ≈0 indicates a composition-driven score (legitimate for repeat/homopolymer RNAs, a caution flag for designs).

**Software.** RNAPhaseek is implemented in PyTorch and provides a command-line tool (`score`/`design`/`validate`).

---

## Data and code availability

The RNAPhaseek source code, strict corpus, and candidate designs are available at https://github.com/QuercusCode/RNAPhaseek. The trained model weights are hosted on the Hugging Face Hub at https://huggingface.co/quercuscode/rnaphaseek and are downloaded at runtime by the command-line tool and the Colab notebook. The one-command command-line tool (`rnaphaseek`, with `score`/`design`/`validate` subcommands) and a one-click Colab notebook are provided for prediction and design. The curated strict RNA-self-LLPS corpus is provided in FASTA format with provenance annotations.

## Supplementary Data

Supplementary Data are available at NAR online, including the full strict corpus, the structural-hard-negative set, the external test set, per-fold cross-validation outputs, and the de novo candidate sequences.

## Funding

[Funding sources and grant numbers to be added.]

## Conflict of Interest Statement

The authors declare no competing interests.

## Author Contributions

[Author contributions to be completed, e.g.: A.C. conceived the study, supervised the work, and wrote the manuscript; all authors read and approved the final manuscript.]

---

## References

*(Citations verified via literature search where indicated; [verify] entries require a final check.)*

1. Banani, S.F. et al. Biomolecular condensates: organizers of cellular biochemistry. *Nat. Rev. Mol. Cell Biol.* (2017).
2. Shin, Y. & Brangwynne, C.P. Liquid phase condensation in cell physiology and disease. *Science* (2017).
3. Van Treeck, B. et al. RNA self-assembly contributes to stress granule formation and defining the stress granule transcriptome. *PNAS* 115, 2734–2739 (2018). doi:10.1073/pnas.1800038115.
4. Jain, A. & Vale, R.D. RNA phase transitions in repeat expansion disorders. *Nature* 546, 243–247 (2017).
5. Roden, C. & Gladfelter, A.S. RNA contributions to the form and function of biomolecular condensates. *Nat. Rev. Mol. Cell Biol.* 22, 183–195 (2021); and poly(UG)/G-quadruplex RNA self-assembly studies, *Nucleic Acids Res.* (2024).
6. Wadsworth, G.M. et al. RNAs undergo phase transitions with lower critical solution temperatures. *Nat. Chem.* (2023).
7. Vernon, R.M. et al. Pi-Pi contacts are an overlooked protein feature relevant to phase separation. *eLife* 7, e31486 (2018).
8. Monti, M. et al. catGRANULE 2.0: accurate predictions of liquid-liquid phase separating proteins at single amino acid resolution. *Genome Biol.* 26 (2025). doi:10.1186/s13059-025-03497-7.
9. Chen, Z. et al. Screening membraneless organelle participants with machine-learning models that integrate multimodal features (PhaSePred). *PNAS* (2022).
10. Zhu, H. et al. RNAPhaSep: a resource of RNAs undergoing phase separation. *Nucleic Acids Res.* 50, D340–D346 (2022). doi:10.1093/nar/gkab985.
11. He, S. et al. RPS 2.0: an updated database of RNAs involved in liquid-liquid phase separation. *Nucleic Acids Res.* 53, D299–D309 (2025). doi:10.1093/nar/gkae951.
12. Chin, K.Y., Ishida, S., Sasaki, Y. & Terayama, K. Predicting condensate formation of protein and RNA under various environmental conditions. *BMC Bioinformatics* 25, 135 (2024). doi:10.1186/s12859-024-05764-z.
13. Klobučar, T., Modič, M. et al. smOOPs: classification of condensation-prone RNAs. *Cell Genomics* (2025). doi:10.1016/j.xgen.2025.101065. [verify author/title]
14. Zadeh, J.N. et al. NUPACK: analysis and design of nucleic acid systems. *J. Comput. Chem.* 32, 170–173 (2011).
15. Saito, H. et al. Programmable computational RNA droplets assembled via kissing-loop interaction. *ACS Nano* 18 (2024). doi:10.1021/acsnano.3c12161.
16. Stewart, J.M. et al. Modular RNA motifs for orthogonal phase-separated compartments / co-transcriptional RNA nanostar condensates. *Nat. Commun.* 15, 6244 (2024). doi:10.1038/s41467-024-50003-x.
17. Fabrini, G. et al. Co-transcriptional production of programmable RNA condensates and synthetic organelles. *Nat. Nanotechnol.* 19, 1665 (2024). doi:10.1038/s41565-024-01726-x.
18. Linder, J. & Seelig, G. Fast activation maximization for molecular sequence design (Fast SeqProp). *BMC Bioinformatics* 22, 510 (2021).
19. Linder, J. et al. A generative neural network for maximizing fitness and diversity of synthetic DNA and protein sequences (Deep Exploration Networks). *Cell Systems* 11, 49–62 (2020).
20. Zhao, Y. et al. GenerRNA: a generative pre-trained language model for de novo RNA design. *PLoS ONE* 19, e0310814 (2024).
21. Yang, S. et al. PhaSeMotif: prediction and generation of phase-separating protein motifs. *Nat. Commun.* (2026). [verify]
22. Chen, J. et al. Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions (RNA-FM). *bioRxiv/arXiv* (2022). [verify final venue]
23. Lorenz, R. et al. ViennaRNA Package 2.0. *Algorithms Mol. Biol.* 6, 26 (2011).
24. Mu, Z. et al. FEGS: a novel feature extraction model for protein sequences. *BMC Bioinformatics* 22, 297 (2021). [verify — confirm this is the FEGS variant used]

---

## Figure legends

**Figure 1.** RNAPhaseek architecture: RNA-FM backbone + FEGSTrans structural adapter + 38-dimensional biophysical fusion (including five self-complementarity features), with three de novo generators driven by the trained classifier.

**Figure 2.** Leakage-honest performance. CV AUROC under per-sequence vs. CD-HIT cluster grouping (0.847 vs. 0.822 honest baseline on the same data), and the final cross-validated AUROC (0.875) with ~4× lower fold variance after corpus consolidation.

**Figure 3.** Structural specificity. Distinguishing condensing positives from composition-matched, structure-destroyed negatives (AUROC 0.90, cross-validated; up from ~0.50 in a composition-only baseline), plus the matched-pair G-quadruplex benchmark on which RNAPhaseek ranks all 9 mechanistically matched pairs correctly (matched-pair accuracy 1.00, mean margin +0.130).

**Figure 4.** The out-of-distribution kissing-loop limit. A dedicated kissing-loop feature separates designed condensing nanostars (=6) from scrambled controls (=2), but that signal is absent from the natural-RNA training corpus (positives ≈ their shuffled negatives), so the gap is a domain-shift limit, not a model defect.

**Figure 5.** Dataset expansion and organism-balanced training. Overall AUROC stable at ~0.88 with 4× variance reduction; organism-balancing lifts non-yeast generalization 0.763 → 0.798 at no overall cost.

**Figure 6.** De novo design. Genetic-algorithm (one optimal motif) and DEN (diverse library) generators, both verified structure-driven by the design-vs-scramble Δ; the GA exceeds, and the DEN matches, the structure-dependence of real LLPS positives.
