using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, DelimitedFiles, BenchmarkTools, JLD, LaTeXStrings

include("util.jl")
tensor_to_vec(T) = vec(vcat([T[:,:,i] for i in axes(T,3)]...))

# whether to subdivide frequencies by index rather than norm
by_index = false
# tolerances
tols  = 10.0 .^ (-2:-1:-14) # [1e-3, 1e-6, 1e-9]

dir = "./torus/"
ns, ms = [], []
for fn in readdir(dir)
    if length(fn) >= 20 && fn[1:20] == "torus_eigenfunctions"
        push!(ns, parse(Int64, match(r"(?<=_n)\d+", fn[21:end]).match))
        push!(ms, parse(Int64, match(r"(?<=_m)\d+", fn[21:end]).match))
    end
end
sort!(ns)
sort!(ms)

ns = [ns[end]]
ms = [ms[end]]

sizes = zeros(Float64, length(tols), length(ns), 2)
errs  = zeros(Float64, length(tols), length(ns))

filename = "./scaling_torus_tols"

if isfile(filename * ".jld")
    dict  = load(filename * ".jld")
    ns    = dict["ns"]
    ms    = dict["ms"]
    sizes = dict["sizes"]
    errs  = dict["errs"]
else
    for (t, tol) in enumerate(tols)
    for (j, (n,m)) in enumerate(zip(ns, ms))
        # load points, eigenvectors, and eigenvalues
        points = readdlm(
            dir*"torus_points_n$n.csv", ',', Float64
            )
        eigvv = nothing
        try
            eigvv  = readdlm(
                dir*"torus_eigenfunctions_n$(n)_m$(m).csv", 
                ',', ComplexF64
                )
        catch err
            eigvv  = readdlm(
                dir*"torus_eigenfunctions_n$(n)_m$(m).csv", 
                ',', Float64
                )
        end

        # first column is eigenvalues
        lams = real.(eigvv[1:m,1])

        # remaining columns are eigenvectors
        Phi = eigvv[:,2:(m+1)]

        # number of levels in factorization
        L = floor(Int64, log(4, n) - 2)

        # compute space tree
        tree_sf = vec(readdlm(dir*"torus-tree-$n.csv", ',', Int64))
        sf      = tensor_to_vec(stack([fill(t, 8, 8) for t in tree_sf], dims=3))
        trx, _ = build_tree(
            Matrix(points[:,3:5]'),
            max_levels=L, min_pts=-1, sf=sf, k=2
        )

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
        
        # "kernel function" giving views of the pre-computed eigenvectors
        kernel(js, ks) = view(Phi, reshape(js, :), reshape(ks, :))
        
        # compute butterfly factorization of the manifold harmonic transform
        B = butterfly_factorize(
            nothing, reshape(1:n, 1, :), reshape(1:m, 1, :); 
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
            ki   .= kernel([i], collect(1:m))[:]
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
    ylims=[1e-3, 1e1], xticks=([5e3, 1e4, 5e4],[L"5\,×10^3", L"10^4", L"5\,×10^4"]), c=scrungle[1]
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

i0 = 4 # div(i1, 2)
powers = [3/2, 2, 5/3]
labels = [L"\mathcal{O}(n^{3/2})", L"\mathcal{O}(n^2)", L"\mathcal{O}(n^{5/3})"]
inds   = [(i0:i1), (i0:i1), (i0:i1)]

plot!(pl, 
    ns[inds[1]], 
    0.2 * sizes[1,inds[1][1],2]/2^30 * (ns[inds[1]]/ns[inds[1][1]]).^powers[1], 
    label=labels[1], 
    line=(2,:dash,:black)
    ) 
plot!(pl, 
    ns[inds[2]],
    6 * sizes[1,inds[2][1],1]/2^30 * (ns[inds[2]]/ns[inds[2][1]]).^powers[2], 
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

savefig(pl, "/Users/beckman/Downloads/torus_accuracy_vs_tol.pdf")