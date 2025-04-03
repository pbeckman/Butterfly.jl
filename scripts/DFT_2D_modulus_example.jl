using Butterfly, Plots, StaticArrays, LinearAlgebra, Morton, SpecialFunctions, Printf, BenchmarkTools
import Base.Iterators: product, partition

include("util.jl")

# size of DFT to factorize
n1d = 2^8
n   = n1d^2
# number of levels in factorization
L = Int(log(4, n)) - 2
# whether to use bit-reversal permutation to get exact butterfly rank 1
permute = false
# whether to subdivide frequencies by index rather than norm
by_index = true
# tolerance for factorization
tol = 1e-3

# compute gridded points and frequencies
xs1d = range(0, 2pi, n1d+1)[1:end-1]
xs   = hcat(collect.(product(xs1d, xs1d))...)
ws1d = -div(n1d,2):(div(n1d,2)-1)
ws   = hcat(collect.(product(ws1d, ws1d))...)

# compute eigenvalues of Laplacian
lams = norm.(eachcol(ws)).^2

Tx = build_tree(
    xs, 
    max_levels=L, min_pts=1, sort=false, k=2
    )

# number of columns to use in factorization
m  = sum(lams .<= div(n1d,2)^2)
ks = Matrix(reshape(1:m, 1, :))
sp = sortperm(lams)[1:m]
if by_index
    Tw = build_tree(
        ks, max_levels=L, min_pts=1, sort=false, k=4, 
        ll=SVector{1, Float64}(1), widths=SVector{1, Float64}(m-1)
        )
else
    Tw = build_tree(
        reshape(lams[sp], 1, :), max_levels=L, min_pts=1, sort=false, k=4
        )
end

# hack which maps column index k back to ws to evaluate kernel
kernel(xs, ks) = cispi.(xs'*ws[:,sp[ks[:]]]/pi)

if permute
    bitrev1d = [
        parse(Int, s, base=2)+1 for s in 
        join.(digits.(0:(n1d-1), base=2, pad=ceil(Int64, log2(n1d))))
                ]
    bitrev = vcat(
        collect(v[bitrev1d] for v in partition(1:n1d^2, n1d))[bitrev1d]...
        )
    xs .= xs[:,bitrev]
    # ws .= ws[:,bitrev]
end

B = butterfly_factorize(
    kernel, xs, ks; 
    L=L, Tx=Tx, Tw=Tw, tol=tol, os=Inf, verbose=true
    );

v = randn(m)

println("\nButterfly matvec : ")
wb = @btime $B*$v
fac_size = 
@printf(
    "Size of factorization : %s\n", 
    Base.format_bytes(Base.summarysize(B.Vt) + Base.summarysize(B.U))
    )

if n < 20_000
    A  = kernel(xs, ks)
    println("\nDense matvec : ")
    w  = @btime $A*$v
    @printf(
        "Size of dense matrix : %s\n", 
        Base.format_bytes(Base.summarysize(A))
        )
    @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
end

##

constant = "√b-√a"
as = range(0, div(n1d,2), 4^L+1).^2 #/ 100

# constant = "b-a"
# as = range(0, div(n1d,2)^2, 4^L+1)

ntest = 10

ls = 2:(L-2)

ranks_sizes = Array{Int64}(undef, 3, length(ls), ntest)

for (j, l) in enumerate(ls)
    println("===== LEVEL $l =====")
    xs1d_box = range(0, 2pi, n1d+1)[1:end-1][1:div(n1d,2^l)]
    xs_box   = hcat(collect.(product(xs1d_box, xs1d_box))...)
    for (ki, k) in enumerate(round.(Int64, range(1, 4^(L-l), ntest)))
        a, b = as[1:4^l:end][[k,k+1]]
        ranks_sizes[:, j, ki] .= block_rank(
            xs=xs_box, a=a, b=b, use_eigvals=true,
            tol=tol, os=10, max_points=2^12, verbose=true
            )
        println()
    end
end

##

comps = ranks_sizes[2,:,:].*ranks_sizes[3,:,:] ./ (
    ranks_sizes[1,:,:].*(ranks_sizes[2,:,:].+ranks_sizes[3,:,:])
    )

# pl = plot(
#     ls, min.(ranks_sizes[2,:,:],ranks_sizes[3,:,:]), 
#     line=(:red,2,:dash), label=["block sizes" fill("", 1, length(ls))]
#     )
pl = plot(ls, ones(length(ls)), line=(:red,2,:dash), label="")
scatter!(pl,
    ls, 
    # ranks_sizes[1,:,:], ylabel="rank",
    comps, ylabel="compression ratio",
    yscale=:log10, 
    ylims=[1e-1, 1e3], 
    c=:black, marker=5, alpha=0.5, label="",
    xlabel="level", 
    title=@sprintf("n = %i, ε = %.0e, b = %.1e", n, tol, as[end]), dpi=300
    )


display(pl)

savefig(
    pl, 
    @sprintf("/Users/beckman/Downloads/compression-n%i-b%.1e.png", n, as[end])
    )

##

mort = sortperm(
        vcat(cartesian2morton.(collect.(Iterators.product(1:n1d, 1:n1d)))...)
    )

# M = angle.(kernel(xs[:,mort], 1:m))
M = angle.(kernel(xs, 1:m))

M[M .> pi - 1e-3] .= -pi

pl = heatmap(
    M, yflip=true, clims=(-pi,pi), c=:lightrainbow, 
    title="Arg(exp(iωx))", dpi=300
    )
