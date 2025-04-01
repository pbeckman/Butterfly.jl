using Butterfly, Plots, StaticArrays, LinearAlgebra, RegionTrees, Morton, SpecialFunctions, Printf, BenchmarkTools
import Base.Iterators: product, partition

include("util.jl")

# size of DFT to factorize
n1d = 2^7
n   = n1d^2
# number of levels in factorization
L = Int(log(4, n)) - 0
# whether to use bit-reversal permutation to get exact butterfly rank 1
permute = false
# tolerance for factorization
tol = 1e-15

mort = sortperm(
        vcat(cartesian2morton.(collect.(Iterators.product(1:n1d, 1:n1d)))...)
    )

xs1d = reshape(collect(range(0, 2pi, n1d+1)[1:end-1]), 1, :)
xs   = hcat(collect.(product(xs1d, xs1d))...)

ws1d = reshape(collect(0.0:(n1d-1)), 1, :)
ws   = hcat(collect.(product(ws1d, ws1d))...)

Tx = build_tree(
    xs, 
    max_levels=L, min_pts=1, sort=false
    )
Tw = build_tree(ws, max_levels=L, min_pts=1, sort=false)

if permute
    bitrev1d = [
        parse(Int, s, base=2)+1 for s in 
        join.(digits.(0:(n1d-1), base=2, pad=ceil(Int64, log2(n1d))))
                ]
    bitrev = vcat(
        collect(v[bitrev1d] for v in partition(1:n1d^2, n1d))[bitrev1d]...
        )
    xs .= xs[:,bitrev]
    ws .= ws[:,bitrev]
end

kernel(xs, ws) = exp.(im*xs'*ws)

B = butterfly_factorize(
    kernel, xs, ws; L=L,
    Tx=Tx, Tw=Tw, tol=tol, verbose=true
    );

println("")
if n < 20_000
    A  = kernel(xs, ws)
    v  = randn(n)
    println("Dense matvec : ")
    w  = @btime $A*$v
    println("Butterfly matvec : ")
    wb = @btime $B*$v
    @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
end

fac_size = Base.summarysize(B.Vt) + Base.summarysize(B.U)
@printf(
    "Size of factorization : %s (%i Bytes)\n", 
    Base.format_bytes(fac_size), fac_size
    )

##

# M = angle.(kernel(xs[:,mort], ws[:,mort]))
M = angle.(kernel(xs, ws))

M[M .> pi - 1e-3] .= -pi

pl = heatmap(
    M, yflip=true, clims=(-pi,pi), c=:lightrainbow, 
    title="Arg(exp(iωx))", dpi=300
    )
