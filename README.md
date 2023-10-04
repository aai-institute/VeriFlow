# Veriflow: Generative Flow Based Density Estimators for Neuro-Symbolic Verification
VeriFow provides a stable and convenient library of flow based general purpose density models with flexibile base distributions,
which are specifically tailored towards the use in neuro-symbolic verification procedures. The major goal is to
provide models that can represent reference distributions which are suitable for verification,
abstract interpretation, and hypothesis testing simultaneously.
The implemented layer are carefully designed to guarntee the following properties:

- Efficient computation of exact densities as well as efficient sampling.
- A piece-wise affine log-density function for all models with (leaky-)ReLU nonlinearity and Laplacian base distribution.
- UDL preserving layers map the upper density level sets of the data distribution to the upper density level sets
of the base Distribution.
- Direct onnx export of log_prob and sampling methods.
