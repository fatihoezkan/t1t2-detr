# Credits

This repository is a from-scratch, T1-T2-native implementation written for the bachelor's
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
solving microstructure quantification as DETR set prediction in the first place. The
postprocessing in `t1t2/postprocess.py` follows his peak-grouping procedure, moved from his
(MD, FA, direction, weight) attribute space to (T1, T2, weight), and the figure set in
`t1t2/viz.py` follows the visual language of his Figures 7, 9, 11, 12, 20 and 23.

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
  footing and lets every loss weight sit at 1.0.
- The forward model implemented twice, in numpy for the generator and in torch for the
  signal-consistency loss, with a parity test that keeps them in agreement.
- The synthetic data generator: fixed-count families, reproducible split streams keyed by
  SeedSequence, the paired fixed-SNR ladder, and a manifest per dataset.
- Evaluation with CSF stratified out. The longest echo time is 150 ms, so the T2 of a
  2000 ms pool is close to unconstrained, and leaving it in the aggregate misrepresents both
  halves.
- The signal-consistency loss and the two arms that measure it.
