# Butterfly.jl

This package uses the *butterfly factorization*<sup>1,2</sup> to compute
compressed approximations of oscillatory operators which can be rapidly applied
to a vector. 

In order to be compressible in butterfly format, a matrix should have the
*complementary low-rank* property, so that all blocks of size $2^\ell \times
\frac{n}{2^\ell}$ have low numerical rank to the user-specified tolerance
$\varepsilon$. If all such blocks have ranks at most $r$, then the butterfly
factorization has $\mathcal{O}(r^2n\log n)$ storage and matrix-vector product
time complexity. Examples of transforms with complementary low-rank structure
include Fourier, Hankel, Legendre, Hermite, Laguerre, and Prolate Spheroidal
Wave Function transforms and their nonuniform analogs.

A minimal demo to factorize the nonuniform Fourier transform matrix
$\mathbf{\Phi}_{jk} := e^{i\omega_k x_j}$ from frequencies
$\{\omega_k\}_{k=1}^m$ to points $\{x_j\}_{j=1}^n$ is as follows:
```julia
using Butterfly

# Fourier kernel
kernel(xs, ws) = exp.(-im*xs'*ws)

# tolerance
tol = 1e-8

# points x_j at which to evaluate the transform
xs = reshape(sort(rand(100_000)), 1, :)

# source frequencies w_k and strengths c_k
ws = reshape(10 .^ range(-2, 2, 10_000), 1, :)
cs = randn(10_000)

# compute butterfly factorization
B = butterfly_factorize(kernel, xs, ws; tol=tol)

# compute Hankel transform by applying factorization to a vector
gs = B * cs
```
See the `scripts` directory for more detailed, heavily commented demos. 

[1] Li, Yingzhou, et al. "Butterfly factorization." Multiscale Modeling & Simulation 13.2 (2015): 714-732.

[2] O'Neil, Michael, Franco Woolfe, and Vladimir Rokhlin. "An algorithm for the rapid evaluation of special function transforms." Applied and Computational Harmonic Analysis 28.2 (2010): 203-226.

