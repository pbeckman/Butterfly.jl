using Butterfly, Plots, StaticArrays, LinearAlgebra, RegionTrees, SpecialFunctions, Printf
import Random: randperm

include("util.jl")

# size of DFT to factorize
n = 2^14
m = n
# number of levels in factorization
L = Int(log(2, min(n,m))) - 4
# whether to use bit-reversal permutation to get exact butterfly rank 1
permute = false
# tolerance for factorization
tol = 1e-8

# xs = reshape(2pi*rand(n), 1, :)
# ws = reshape(m*rand(m), 1, :)
xs = reshape(collect(range(0, 2pi, n+1)[1:end-1]), 1, :)
ws = reshape(collect(0.0:(m-1)), 1, :)

Tx = build_tree(
    xs, 
    max_levels=L, min_pts=1, sort=false
    )
Tw = build_tree(ws, max_levels=L, min_pts=1, sort=false)

if permute
    bitrev = [
        parse(Int, s, base=2)+1 for s in 
        join.(digits.(0:(n-1), base=2, pad=ceil(Int64, log2(n))))
                ]
    xs .= xs[:,bitrev]
    ws .= ws[:,bitrev]
end

# kernel(xs, ws) = Float64.((n/2pi * xs) .≈ ws')
kernel(xs, ws) = cispi.(-xs'*ws/pi)
# kernel(xs, ws) = besselj.(0, xs'*ws)

B = butterfly_factorize(
    kernel, xs, ws; L=L,
    Tx=Tx, Tw=Tw, tol=tol, verbose=true, os=1.2
    );

if n < 20_000
    A  = kernel(xs, ws)
    v  = randn(n)
    w  = A*v
    wb = B*v
    @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
    @printf(
        "Size of dense matrix  : %s\n", 
        Base.format_bytes(Base.summarysize(A))
    )
end

@printf(
    "Size of factorization : %s\n", 
    Base.format_bytes(Base.summarysize(B.Vt) + Base.summarysize(B.U))
    )