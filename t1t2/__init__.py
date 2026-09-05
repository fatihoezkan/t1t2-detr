"""DETR-style set prediction for T1-T2 correlation MRI.

The model reads a voxel's 64-point signal (8 inversion times x 8 echo times) and returns a
fixed number of candidate compartments (T1, T2, weight), each with an existence score.
Training matches candidates to ground truth with the Hungarian algorithm, so the order of
the candidates carries no meaning.

Entry point: python -m t1t2.experiment --config <yaml>
"""

__version__ = "1.1.0"
