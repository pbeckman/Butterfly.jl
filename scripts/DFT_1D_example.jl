using Butterfly, Plots, StaticArrays, LinearAlgebra, RegionTrees, SpecialFunctions, Printf
import Random: randperm

include("util.jl")

# size of DFT to factorize
n = 2^12
m = n

# number of levels in factorization
L = Int(log(2, min(n,m))) - 3

# whether to use nonuniform points and frequencies
nonuniform = false

# whether to use bit-reversal permutation to get exact butterfly rank 1 for DFT
permute = true

# tolerance for factorization
tol = 1e-4

# kernel of matrix to be factorized -- replace with your own if desired
# kernel(xs, ws) = cispi.(-xs'*ws/pi)
kernel(xs, ws) = besselj.(0, xs'*ws)

if nonuniform
    xs = reshape(2pi*rand(n), 1, :)
    ws = reshape(m*rand(m), 1, :)
else
    xs = reshape(collect(range(0, 2pi, n+1)[1:end-1]), 1, :)
    ws = reshape(collect(0.0:(m-1)), 1, :)
end

trx = build_tree(xs, max_levels=L, min_pts=1, sort=false)
trw = build_tree(ws, max_levels=L, min_pts=1, sort=false)

if permute
    bitrev = [
        parse(Int, s, base=2)+1 for s in 
        join.(digits.(0:(n-1), base=2, pad=ceil(Int64, log2(n))))
                ]
    xs .= xs[:,bitrev]
    ws .= ws[:,bitrev]
end

B = butterfly_factorize(
    kernel, xs, ws; L=L,
    trx=trx, trw=trw, tol=tol, verbose=1, method=:ID#, os=3
    );

if n < 10_000
    A  = kernel(xs, ws)
    v  = randn(n)
    w  = A*v
    wb = B*v
    @printf("Relative apply error  : %.2e\n", norm(w - wb) / norm(w))
end
@printf(
        "\nSize of dense matrix : %s\n", 
        Base.format_bytes(sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))
        ))

@printf(
    "Size of factorization : %s\n", 
    Base.format_bytes(Base.summarysize(B.Vt) + Base.summarysize(B.U))
    )