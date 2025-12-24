using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, BenchmarkTools
import Base.Iterators: product, partition
import SphericalHarmonics: sphericalharmonic

include("util.jl")

# size of DFT to factorize
n1d = 2^13
n   = n1d^2
# number of eigenvectors to compress (or nothing to compress O(n) vectors)
m = round(Int64, n/4)
# number of levels in factorization
L = floor(Int64, log(4, m)) - 0
# whether to compute spherical harmonic transform
SHT = false
# whether to use uniformly random xs
nonuniform = false
# whether to subdivide frequencies by index rather than norm
by_index = false
# tolerance for factorization
tol = 1e-3

# compute points 
if SHT
    # xs are [θ; φ] where θ is radial and φ is vertical angle
    if nonuniform
        # sample points uniformly at random on the sphere
        xs   = randn(3, n)
        xs ./= norm.(eachcol(xs))'
        xs = [
            sign.(xs[2,:])' .* acos.(xs[1,:] ./ norm.(eachcol(xs[1:2,:])))';
            acos.(xs[3,:])'
        ]
    else
        # golden spiral quasi-equispaced points on the sphere
        xs = [
            mod.(pi*(1 + sqrt(5))*(0:n .+ 0.5)', 2pi);
            acos.(1 .- 2*(0:n .+ 0.5)'/n)
        ]
    end
    # build a quadtree on [θ; cos(φ)] to get an equal-area tree on the sphere
    tree_xs = [xs[1,:]'; cos.(xs[2,:])']
else
    if nonuniform
        # sample uniformly random points in [0,2π]^2
        xs = 2pi*rand(2, n)
    else
        # equispaced grid on [0,2π]^2
        xs1d = range(0, 2pi, n1d+1)[1:end-1]
        xs   = hcat(collect.(product(xs1d, xs1d))...)
    end
    tree_xs = xs
end

# compute space tree
trx, _ = build_tree(
    tree_xs, 
    max_levels=L, min_pts=1, sort=false
    )

if SHT
    # compute spherical harmonic indices and Laplacian eigenvalues
    mxl  = floor(Int64, sqrt(n)-1)
    lms  = stack([l; m] for l=0:mxl for m=-l:l)
    lams = Float64.(lms[1,:] .* (lms[1,:] .+ 1))

    # number eigenvalues
    if isnothing(m)
        m = (mxl + 1)^2
    end
    ks   = Matrix(reshape(1:m, 1, :))
    lams = lams[1:m]

    # evaluate spherical harmonics
    kernel(xs, ks) = (length(ks) == 0) ? zeros(Float64, length(xs), 0) : stack([sphericalharmonic.(
        view(xs, 2, :), 
        view(xs, 1, :),
        l=lm[1], m=lm[2]
        ) for lm in eachcol(view(lms, :, reshape(ks,:)))])
else
    # compute frequencies and Laplacian eigenvalues
    ws1d = -div(n1d,2):(div(n1d,2)-1)
    ws   = hcat(collect.(product(ws1d, ws1d))...)
    lams = norm.(eachcol(ws)).^2

    # number eigenvalues
    if isnothing(m)
        m = sum(lams .<= div(n1d,2)^2)
    end
    ks = Matrix(reshape(1:m, 1, :))

    # sort frequencies and eigenvalues
    sp   = sortperm(lams)[1:m]
    ws   = ws[:,sp]
    lams = lams[sp]

    # hack which maps column index k back to ws to evaluate kernel
    kernel(xs, ks) = cispi.(xs' * view(ws,:,reshape(ks,:)) / pi)
end

# compute frequency tree either by eigenvalue or index
if by_index
    trw, _ = build_tree(
        ks, max_levels=L, min_pts=-1, sort=false, k=4, 
        ll=SVector{1, Float64}(1), widths=SVector{1, Float64}(length(ks)-1)
        )
else
    trw, _ = build_tree(
        reshape(lams, 1, :), 
        # reshape(sqrt.(lams), 1, :), 
        # reshape(lams.^16, 1, :), 
        max_levels=L, min_pts=-1, sort=false, k=4
        )
end

##

# compute butterfly factorization of the manifold harmonic transform
B = butterfly_factorize(
    nothing, xs, ks; 
    kernel=kernel, L=L, trx=trx, trw=trw, tol=tol, method=:ID, os=2,
    verbose=true
    );

v  = randn(size(ks,2))
va = randn(size(xs,2))

@printf(
    "\nSize of factorization : %s\n", 
    Base.format_bytes(
        Base.summarysize(B.Vt) + Base.summarysize(B.U) + Base.summarysize(B.sk)
        )
    )

@printf(
        "Size of dense matrix : %s\n", 
        Base.format_bytes(
            sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))
        )
    )

##

println("\nButterfly matvec : ")
# wb = @btime $B*$v
wb  = @time B*v
println("Butterfly adjoint matvec : ")
wba = @time B'*va

if prod(size(B)) < 20_000^2
    A  = kernel(xs, ks)
    println("\nDense matvec : ")
    # w  = @btime $A*$v
    w  = @time A*v
    wa = A'*va
    @printf(
        "\nRelative apply error  : %.2e, %.2e (adjoint)\n", 
        norm(w - wb) / norm(w), norm(wa - wba) / norm(wa)
        )
end

##

sizes = [size.(Vtl) for Vtl in B.Vt]
pl = plot(
    [sum(prod.(sizes[l])) for l in eachindex(sizes)], 
    line=2, label="", marker=3, yscale=:log10, legend=:topleft,
    xlabel="ℓ", ylabel=L"nnz($\ T_\ell \, $)", title="n=$n, m=$m"
    )
# scatter!(pl, eachindex(sizes), [sum(getindex.(sizes[l], 2).^2) for l=eachindex(sizes)],label="", markerstrokewidth=0)
plot!(pl, 1:(div(L,2)+2), 4 .^ (L .+ 2*(1:(div(L,2)+2)) .- 2), label=L"4^{L + 2\ell}", line=(2, :black))
# plot!(pl, 1:(div(L,2)+1), 8 * 4 .^ (L .+ 2*(1:(div(L,2)+1))) ./ sqrt(m), label=L"4^{L + 2\ell} / \sqrt{m}", line=(2, :black, :dash))
# plot!(pl, 1:(div(L,2)+2), 16 * 20 * 4 .^ (L .+ (1:(div(L,2)+2))), label=L"4^{L+\ell}", line=(2, :red, :dash))
# plot!(pl, 1:L, fill(460 * 4 .^ (3L/2), L), label=L"4^{3L/2}", line=(2, :grey75, :dot))
# plot!(pl, div(L,2)+1:L+1, 4^4 * 16 * 20^2 * 4 .^ (2L .- (div(L,2)+1:L+1)), label=L"4^{2L-\ell}", line=(2, :grey, :dashdot))
plot!(pl, div(L,2)+1:L+1, 16 * 20^2 * 4 .^ (5L/3 .- 2/3*(div(L,2)+1:L+1)), label=L"4^{5L/3-2\ell/3}", line=(2, :grey, :dashdot))
# plot!(pl, div(L,2)+2:L+1, 16 * 20^3 * 4.0 .^ (3L .- 2*(div(L,2)+2:L+1)), label=L"4^{-2\ell}", line=(2, :grey75, :dot))
display(pl)
# savefig(pl, "/Users/beckman/Downloads/level_size_n$(n)_m$(m)_L$(L).pdf")

##

l = 7

ks = round.(Int64, range(1, 4^(L-l), 16))
trx_lvl = collect(Butterfly.LevelIterator(trx, l))
trw_lvl = collect(Butterfly.LevelIterator(trw, L-l))[ks]
bx = trx_lvl[1]

bs = sqrt.(
    getindex.(
        getfield.(getfield.(trw_lvl, :boundary), :ll) .+ 
        getfield.(getfield.(trw_lvl, :boundary), :widths)
    )
)
r = sqrt(2)*trx_lvl[1].boundary.widths[1]/2

ranks = fill(NaN, 2, length(trw_lvl))

for (j, bl) in enumerate(trw_lvl)
    @printf("Computing block rank %i of %i\n", j, length(trw_lvl))
    # x_ll, x_widths = bx.boundary.ll, bx.boundary.widths
    # l_ll, l_widths = bl.boundary.ll, bl.boundary.widths

    # xs_os = stack(Iterators.product(
    #     range(x_ll[1], x_ll[1]+x_widths[1], 32), 
    #     range(x_ll[2], x_ll[2]+x_widths[2], 32)
    #     ), dims=2)
    # rth_os = stack(Iterators.product(
    #     range(sqrt(l_ll[1]), sqrt(l_ll[1]+l_widths[1]), 8), 
    #     range(0, 2pi, 128)
    #     ), dims=2)
    # ws_os = [rth_os[1,:] .* cos.(rth_os[2,:]) rth_os[1,:] .* sin.(rth_os[2,:])]'

    M    = kernel(xs[:,bx.inds], bl.inds)
    # M_os = cispi.(xs_os' * ws_os)

    ranks[1, j] = rank(M, rtol=tol)
end

##

pl = plot(ylims=(5,500), xlims=(1,1000), scale=:log10, xlabel="k", ylabel="rank", legend=:outertopright); 
for l in eachindex(sizes)
    scatter!(pl, 
        getindex.(sizes[l][1,:], 1), 
        label="ℓ = $(l-1)", line=0, 
        markerstrokewidth=2, markerstrokecolor=:auto, marker=(3, palette(:default)[l])
        )
end
plot!(1:100, 35*(1:100).^(1/2), label=L"k^{1/2}", line=(:gray, :dashdot, 2))
plot!(1:100, 20*(1:100).^(1/3), label=L"k^{1/3}", line=(:black, :dash, 2))
scatter!(ks, ranks', labels=["DFT" "oversampled"], c=[:black :white])
# scatter!(ks, circle_ranks, label="circle", c=:red)
display(pl)