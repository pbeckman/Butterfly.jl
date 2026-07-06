using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, BenchmarkTools, JLD, LaTeXStrings
import Base.Iterators: product, partition
import SphericalHarmonics: sphericalharmonic

include("util.jl")

# whether to compute spherical harmonic transform
SHT = false
# whether to subdivide frequencies by index rather than norm
by_index = false

# increasing number of points in space
ns = 160_000 
# ns = round.(Int64, sqrt.(10 .^ range(3, 6, 10))).^2
# proportionally increasing number of eigenfunctions
# ms = round.(Int64, ns / 25)
# tolerances
tols = 10.0 .^ (-2:-1:-14)
# tols = [1e-3, 1e-6, 1e-9]

sizes = zeros(Float64, length(tols), length(ns), 2)
errs  = zeros(Float64, length(tols), length(ns))

filename = "./scaling_$(SHT ? "NUSHT" : "NUFFT2D")_tols"

if isfile(filename * ".jld")
    dict = load(filename * ".jld")
    ns = dict["ns"]
    ms = dict["ms"]
    sizes = dict["sizes"]
    errs  = dict["errs"]
else
    for (t, tol) in enumerate(tols)
    for (j, (n,m)) in enumerate(zip(ns, ms))
        # number of levels in factorization
        L = max(1, floor(Int64, log(4, m)) - 2)

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
            xs = [xs[1,:]'; cos.(xs[2,:])']
        else
            # equispaced grid on [0,2π]^2
            xs1d = range(0, 2pi, Int64(sqrt(n))+1)[1:end-1]
            xs   = hcat(collect.(product(xs1d, xs1d))...)
        end

        # compute space tree
        trx, _ = build_tree(
            xs, max_levels=L, min_pts=1, sort=false
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

        sizes[t, j, 1] = Base.summarysize(B.Vt) + Base.summarysize(B.U)
        sizes[t, j, 2] = sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))

        v  = randn(m)
        Bv = conj.(B*v)
        Av = zeros(ComplexF64, n)
        ki = zeros(ComplexF64, m)
        for i=1:n
            # compute ith row
            ki   .= kernel(xs[:,i], collect(1:m))[:]
            Av[i] = dot(ki, v)
        end
        errs[t, j] = norm(Av - Bv) / norm(Av)
        @printf("Relative error in matvec : %.2e\n", errs[t, j])

        save(
            filename * ".jld",
            "sizes", sizes,
            "errs", errs,
            "ns", ns,
            "ms", ms
            )
    end
    end
end

##

include("/Users/beckman/.julia/config/custom_colors.jl")

default(fontfamily="Computer Modern")
gr(size=(300, 300))
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

i1 = sum((!).(iszero.(sizes[1,:,1])))
pl = plot(
    ns, sizes[1,:,2] / 2^30, 
    xlabel="n", ylabel="size (GB)",
    label="dense",
    line=2, marker=3, markerstrokewidth=0, dpi=300,
    ylims=[1e-4, 1e4], xticks=([1e3, 1e4, 1e5, 1e6],[L"10^3", L"10^4", L"10^5", L"10^6"]), c=scrungle[1]
    )
for (t, (tol, label)) in enumerate(zip(tols, [L"ε=10^{-3}", L"ε=10^{-6}", L"ε=10^{-9}"]))
    plot!(pl,
        ns, sizes[t,:,1] / 2^30,
        label=label,
        scale=:log10,
        line=2, marker=3, markerstrokewidth=0, 
        c=scrungle[t+1], legend=:topleft
        )
end

i0 = 6 # div(i1, 2)
powers = [3/2, 2, 5/3]
labels = [L"\mathcal{O}(n^{3/2})", L"\mathcal{O}(n^2)", L"\mathcal{O}(n^{5/3})"]
inds   = [(i0:i1), (i0:i1), (i0:i1)]

plot!(pl, 
    ns[inds[1]], 
    0.07 * sizes[1,inds[1][1],2]/2^30 * (ns[inds[1]]/ns[inds[1][1]]).^powers[1], 
    label=labels[1], 
    line=(2,:dash,:black)
    ) 
plot!(pl, 
    ns[inds[2]],
    20 * sizes[1,inds[2][1],1]/2^30 * (ns[inds[2]]/ns[inds[2][1]]).^powers[2], 
    label=labels[2], 
    line=(2,:dashdot,:gray)
    )
# plot!(pl, 
#     ns[inds[3]], 
#     0.25 * sizes[1,inds[3][1],1]/2^30 * (ns[inds[3]]/ns[inds[3][1]]).^powers[3], 
#     label=labels[3], 
#     line=(2,:dot,:gray60), legend=:bottomright
#     )
display(pl)

savefig(pl, "/Users/beckman/Downloads/" * filename * ".pdf")

##

include("/Users/beckman/.julia/config/custom_colors.jl")

default(fontfamily="Computer Modern")
gr(size=(300, 300))
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

pl = plot(
    tols, errs,
    line=2, marker=3, markerstrokewidth=0,
    xlabel=L"$\varepsilon$", 
    ylabel=L"rel. $\ell^2$ error",
    label="BF-MHT",
    xscale=:log10, yscale=:log10, 
    ylims=(5e-16, 1e-1), xlims=(5e-16, 5e-2),
    legend=:bottomright, c=scrungle[6]
    )
plot!([1e-16, 1], [1e-16, 1], line=(1, :black, :dash), label="ε")
display(pl)

savefig(pl, "/Users/beckman/Downloads/DFT2D_accuracy_vs_tol.pdf")