[Date]

To the Editors,
*Nucleic Acids Research*

Dear Editors,

We are pleased to submit our manuscript, **"RNAPhaseek: a foundation-model framework for predicting and designing self–phase-separating RNA,"** for consideration as a research article in *Nucleic Acids Research*.

RNA is now firmly established as an autonomous driver of liquid–liquid phase separation: certain RNAs form condensates by themselves, through RNA–RNA multivalent interactions, in the complete absence of protein. Yet the computational toolbox for this biology is conspicuously one-sided. While more than a dozen sequence-based predictors exist for protein phase separation, **no published method predicts protein-free RNA-self-LLPS from sequence**, and the de novo design of phase-separating RNA remains an entirely rational, thermodynamics-driven exercise. The available RNA resources are databases of condensate membership, and the one learned model that uses RNA-sequence features requires a protein partner and explicitly cannot predict RNA-only phase separation.

Our manuscript addresses this gap directly. We present RNAPhaseek, which (i) predicts protein-free RNA-self-LLPS from sequence using an RNA foundation model fused with graph-based structural and biophysical features, and (ii) uses that learned predictor as an oracle to generate de novo self–phase-separating RNA. On the largest strictly-curated corpus of protein-free RNA-self-LLPS sequences assembled to date (1,352 positives), the model achieves a leakage-free cross-validated AUROC of 0.88, and a structural-specificity AUROC of 0.90 for distinguishing condensing RNAs from composition-matched, structure-disrupted controls. Three orthogonal generators produce candidate sequences that we show to be structurally grounded rather than composition artifacts. To our knowledge, this is the first learned predictor of protein-free RNA-self-LLPS and the first to couple such a predictor to de novo RNA-condensate design.

We believe the work is well suited to *Nucleic Acids Research* for three reasons. First, it is squarely within NAR's RNA and computational-methods scope, and contributes a reusable resource: a strict RNA-self-LLPS corpus, a trained model, and a one-command tool. Second, we have prioritized methodological rigor — leakage-controlled (CD-HIT cluster-grouped) cross-validation, adversarial structural hard negatives, a frozen external test, and feature-ablation controls — and we report it transparently, including a self-corrected leakage analysis. Third, we are forthright about two limitations and analyze them mechanistically rather than concealing them: an out-of-distribution gap on designed kissing-loop controls, which we trace to the absence of that condensation mode from natural-RNA training data, and a cross-organism generalization gap reflecting the genuine scarcity of strict non-yeast data, which we partially close with organism-balanced training. We hope reviewers will view this honesty as a strength.

We confirm that this manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the submission and declare no competing interests. The model, code, curated corpus, and candidate designs will be made publicly available upon publication.

We would be grateful for your consideration and look forward to the reviewers' feedback.

Sincerely,

**Amir M. Cheraghali**
On behalf of all authors
INSERM, France
amirmohammad.cheraghali@inserm.fr

*Suggested reviewers (optional): [to be completed].*
