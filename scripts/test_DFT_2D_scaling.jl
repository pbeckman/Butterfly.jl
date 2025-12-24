using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, BenchmarkTools, JLD, LaTeXStrings
import Base.Iterators: product, partition
import SphericalHarmonics: sphericalharmonic

include("util.jl")

# whether to compute spherical harmonic transform
SHT = false
# whether to subdivide frequencies by index rather than norm
by_index = false
# tolerance for factorization
tol = 1e-3
# whether to scale n (scale m if false)
scale_n = true

if scale_n
    # increasing number of points in space
    ns = round.(Int64, 10 .^ range(3, 7, 12))

    # whether to compress m = O(1) or m = frac*n = O(n) eigenmodes
    m, frac = nothing, 1/4
    # m, frac = minimum(ns), nothing

    scl_txt = (isnothing(frac) ? "m$m" : "frac$frac")
else
    # fixed number of points in space
    n = 10^5

    # increasing number of eigenmodes
    ms = round.(Int64, 10 .^ range(2, log10(n/2), 10))

    scl_txt = "n$n"
end

filename = "./scaling_$(SHT ? "NUSHT" : "NUFFT2D")_$(@sprintf("tol%.0e", tol))_$scl_txt"

global bl_szs = nothing

if false # isfile(filename * ".jld")
    dict  = load(filename * ".jld")
    if scale_n
        ns = dict["ns"]
    else
        ms = dict["ms"]
    end
    sizes = dict["sizes"]
else
    sizes = zeros(Float64, 2, length(scale_n ? ns : ms))
    for (j, nm) in enumerate(scale_n ? ns : ms)
        if scale_n
            n = nm
            if !isnothing(frac)
                m = round(Int64, frac * n)
            end
        else
            m = nm
        end

        # number of levels in factorization
        L = max(1, floor(Int64, log(4, m)) - 2)
        # L = div(floor(Int64, log(4, n)), 2)

        # compute points 
        if SHT
            # sample points uniformly at random on the sphere
            xs   = randn(3, n)
            xs ./= norm.(eachcol(xs))'
            # xs are [θ; φ] where θ is radial and φ is vertical angle
            xs = [
                sign.(xs[2,:])' .* acos.(xs[1,:] ./ norm.(eachcol(xs[1:2,:])))';
                acos.(xs[3,:])'
            ]
            
            # build a quadtree on [θ; cos(φ)] to get an equal-area tree on the sphere
            tree_xs = [xs[1,:]'; cos.(xs[2,:])']
        else
            # sample uniformly random points in [0,2π]^2
            xs = 2pi*rand(2, n)
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
            ks   = Matrix(reshape(1:m, 1, :))
            lams = lams[1:m]

            # evaluate spherical harmonics
            global kernel(xs, ks) = (length(ks) == 0) ? zeros(Float64, length(xs), 0) : stack([sphericalharmonic.(
                    view(xs, 2, :), 
                    view(xs, 1, :),
                    l=lm[1], m=lm[2]
                    ) for lm in eachcol(view(lms, :, reshape(ks,:)))])
        else
            # compute frequencies and Laplacian eigenvalues
            n1d  = ceil(Int64, sqrt(n))
            ws1d = -div(n1d,2):(div(n1d,2)-1)
            ws   = hcat(collect.(product(ws1d, ws1d))...)
            lams = norm.(eachcol(ws)).^2

            # number eigenvalues
            ks = Matrix(reshape(1:m, 1, :))

            # sort frequencies and eigenvalues
            sp   = sortperm(lams)[1:m]
            ws   = ws[:,sp]
            lams = lams[sp]

            # hack which maps column index k back to ws to evaluate kernel
            global kernel(xs, ks) = cispi.(xs' * view(ws,:,reshape(ks,:)) / pi)
        end

        # compute frequency tree either by eigenvalue or index
        if by_index
            trw, _ = build_tree(
                ks, max_levels=L, min_pts=-1, sort=false, k=4, 
                ll=SVector{1, Float64}(1), widths=SVector{1, Float64}(length(ks)-1)
                )
        else
            trw, _ = build_tree(
                reshape(lams, 1, :), max_levels=L, min_pts=-1, sort=false, k=4
                )
        end

        # compute butterfly factorization of the manifold harmonic transform
        B = butterfly_factorize(
            nothing, xs, ks; 
            kernel=kernel, L=L, trx=trx, trw=trw, tol=tol, verbose=true, method=:ID, os=3
            );

        sizes[1, j] = Base.summarysize(B.Vt) + Base.summarysize(B.U)

        sizes[2, j] = sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))

        save(
            filename * ".jld",
            "sizes", sizes,
            (scale_n ? ("ns", ns) : ("ms", ms))...
            )

        global bl_szs = [size.(Vtl) for Vtl in B.Vt]
    end
end

##

nms = (scale_n ? ns : ms)

default(fontfamily="Computer Modern")
gr(size=(300, 300))

i1 = sum((!).(iszero.(sizes[1,:])))
pl = plot(
    nms[1:i1], sizes[:,1:i1]' / 2^30, 
    labels=["butterfly" "dense"],
    xlabel=(scale_n ? "n" : "m"), 
    ylabel="size (GB)",
    # title=(SHT ? "NUSHT\n" : "2D NUFFT\n") * 
    #     @sprintf("ε = %.0e, ", tol) * 
    #     (isnothing(frac) ? "m = $m" : @sprintf("m = %.4f n", frac)),
    scale=:log10,
    ylims=[1e-4, 1e3],
    line=2, marker=3, markerstrokewidth=0, dpi=300
    )

i0 = div(i1, 2)
if scale_n
    powers = [2, 3/2]
    labels = [L"\mathcal{O}(n^2)", L"\mathcal{O}(n^{3/2})"]
    inds   = [(i0:i1), (i0:i1)]
else
    powers = [1, 1/2]
    labels = [L"\mathcal{O}(m)", L"\mathcal{O}(\sqrt{m})"]
    inds   = [(i0:i1), (1:(i0-1))]
end

plot!(pl, 
    nms[inds[1]], 
    0.5 * sizes[2,inds[1][1]]/2^30 * (nms[inds[1]]/nms[inds[1][1]]).^powers[1], 
    label=labels[1], 
    line=(2,:dashdot,:gray), legend=:bottomright
    ) 
plot!(pl, 
    nms[inds[2]], 
    0.5 * sizes[1,inds[2][1]]/2^30 * (nms[inds[2]]/nms[inds[2][1]]).^powers[2], 
    label=labels[2], 
    line=(2,:dash,:black), legend=:bottomright
    )
display(pl)

savefig(pl, "/Users/beckman/Downloads/" * filename * ".pdf")