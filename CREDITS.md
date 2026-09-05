# Credits

This repository is a T1-T2-native implementation written for the bachelor's
thesis *Detection Transformer for Microstructure Quantification from T1-T2 Correlation MRI*
(Fatih Özkan; advisor Sebastian Endt, Technische Hochschule Ingolstadt).

The Detection Transformer architecture is adopted, not invented here. The thesis contributes
the reframing of the T1-T2 spectral inverse problem as set prediction, the experiment matrix
that isolates which design choices matter, and the finding that the loss weighting rather
than the protocol accounts for much of the difficulty with small compartments.

## What this builds on

DETR, Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020. The base
method: a fixed set of learned queries, a transformer decoder, and a bipartite-matched set
loss. Every structural idea in `t1t2/model.py` and `t1t2/loss.py` traces back to this paper.

Johannes Reinhold Schlund, diffusion-correlation DETR, predecessor thesis. The idea of
solving microstructure quantification as DETR set prediction in the first place. An earlier
version of this code also carried his peak-grouping postprocessing, moved to (T1, T2, weight),
and a figure set in his visual language; neither entered the thesis and both were removed
from the release.

Marcus Alex Wirth, diffusion-correlation DETR, parallel thesis. The evaluation framework in
`t1t2/nd_metrics.py` is his Normalised Distance criterion and the mAP construction on top of
it, adapted to two feature dimensions and to log-space normalisation. One change was made on
purpose: he searches the existence threshold on the test split, and here it is calibrated on
validation instead.

Sebastian Endt, `correlation-imaging-detr_t1t2`. The T1-T2 DETR this work started from: the
MLP signal encoder, learned queries, transformer decoder, per-query sigmoid heads with a
concatenated-query existence head, and the Hungarian loss structure. The fixed 8 x 8 TI x TE
acquisition protocol in `datagen/data/ti_te_dict.mat` comes from that repository and is used
exactly as stored.

## What is original here

- A config-driven experiment framework in which a run is fully described by its YAML file.
  This is what makes the one-change experiment matrix possible.
- Log-min-max target normalisation for the sigmoid heads, which puts T1 and T2 on the same
  footing and lets the three regression weights sit at 1.0; the existence term is weighted 0.1.
- The forward model implemented twice, in numpy for the generator and in torch for the
  signal-consistency loss; the two must stay in step.
- The synthetic data generator: fixed-count families, reproducible split streams keyed by
  SeedSequence, the paired fixed-SNR ladder, and a manifest per dataset.
- Evaluation with CSF stratified out. The longest echo time is 150 ms, so the T2 of a
  2000 ms pool is close to unconstrained, and leaving it in the aggregate misrepresents both
  halves.
- The signal-consistency loss and the two arms that measure it.
