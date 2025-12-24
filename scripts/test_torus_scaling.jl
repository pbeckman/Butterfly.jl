using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, DelimitedFiles, BenchmarkTools, JLD, LaTeXStrings

include("util.jl")

# whether to subdivide frequencies by index rather than norm
by_index = false
# tolerance for factorization
tol = 1e-3
# whether to scale n (scale m if false)
scale_n = true

dir = "/Users/beckman/Downloads/torus/"
ns_in, ms_in = [], []
for fn in readdir(dir)
    if fn[1:20] == "torus_eigenfunctions"
        push!(ns_in, parse(Int64, match(r"(?<=_n)\d+", fn[21:end]).match))
        push!(ms_in, parse(Int64, match(r"(?<=_m)\d+", fn[21:end]).match))
    end
end
frac = ms_in[1] / ns_in[1]
sort!(ns_in)
sort!(ms_in)

if scale_n
    ns = ns_in
    ms = ms_in
    scl_txt = "frac$frac" 
else
    ms    = copy(ms_in)
    ms_in = fill(ms_in[end], length(ms_in))
    ns    = fill(ns_in[end], length(ns_in))
    ns_in = ns
    scl_txt = "n$n"
end

filename = "./scaling_torus_$(@sprintf("tol%.0e", tol))_$scl_txt"

if false # isfile(filename * ".jld")
    dict  = load(filename * ".jld")
    ns    = dict["ns"]
    sizes = dict["sizes"]
else
    sizes = zeros(Float64, 2, length(ns))
    for (j, (n, m, n_in, m_in)) in enumerate(zip(ns, ms, ns_in, ms_in))
        # load points, eigenvectors, and eigenvalues
        points = readdlm(
            dir*"torus_points_n$n_in.csv", ',', Float64
            )
        eigvv = nothing
        try
            eigvv  = readdlm(
                dir*"torus_eigenfunctions_n$(n_in)_m$(m_in).csv", 
                ',', ComplexF64
                )
        catch err
            eigvv  = readdlm(
                dir*"torus_eigenfunctions_n$(n_in)_m$(m_in).csv", 
                ',', Float64
                )
        end

        # first column is eigenvalues
        lams = real.(eigvv[1:m,1])

        # remaining columns are eigenvectors
        Phi = eigvv[:,2:(m+1)]

        # declare 2D (u,v) points
        us = Matrix(points[:, 1:2]')

        # number of levels in factorization
        L = floor(Int64, log(4, n))

        # compute space tree
        trx = build_tree(
            us, 
            max_levels=L, min_pts=1, sort=false, k=2
            )

        # compute frequency tree either by eigenvalue or index
        if by_index
            trw = build_tree(
                ks, max_levels=L, min_pts=-1, sort=false, k=4, 
                ll=SVector{1, Float64}(1), widths=SVector{1, Float64}(length(ks)-1)
                )
        else
            trw = build_tree(
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

        sizes[1, j] = Base.summarysize(B.Vt) + Base.summarysize(B.U)

        sizes[2, j] = sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))

        save(
            filename * ".jld",
            "ns", ns,
            "sizes", sizes
            )
        
        v = randn(m)
        
        println("Butterfly matvec : ")
        wb = @time B*v

        if prod(size(B)) < 20_000^2
            println("Dense matvec : ")
            # w  = @btime $Phi[:, 1:m]*$v
            w = @time Phi[:, 1:m]*v
            @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
        end
    end
end

##

default(fontfamily="Computer Modern")
gr(size=(300, 300))

nms = (scale_n ? ns : ms)

tks = nms[1] * 2 .^ (0:floor(Int64, log2(nms[end]/nms[1])))

i1 = sum((!).(iszero.(sizes[1,:])))
pl = plot(
    nms[1:i1], sizes[:,1:i1]' / 2^30, 
    labels=["butterfly" "dense"],
    xlabel=(scale_n ? "n" : "m"),  
    ylabel="size (GB)",
    # title="MHT on deformed torus\n" * 
    #     @sprintf("ε = %.0e, ", tol) * 
    #     (isnothing(frac) ? "m = $m" : "m = $frac n"),
    scale=:log10,
    # ylims=[5e-4, 1e3],
    line=2, marker=3, markerstrokewidth=0, dpi=300,
    xticks=(tks, string.(tks))
    )

i0 = div(i1, 2)
if scale_n
    powers = [2, 1.5]
    labels = [L"\mathcal{O}(n^2)", L"\mathcal{O}(n^{3/2})"]
    inds   = [(i0:i1), (i0:i1)]
else
    powers = [1, 0.5]
    labels = [L"\mathcal{O}(m)", L"\mathcal{O}(\sqrt{m})"]
    inds   = [(i0:i1), (1:(i0-1))]
end

plot!(pl, 
    nms[inds[1]], 
    0.7 * sizes[2,inds[1][1]]/2^30 * (nms[inds[1]]/nms[inds[1][1]]).^powers[1], 
    label=labels[1], 
    line=(2,:dashdot,:gray), legend=:bottomright
    ) 
plot!(pl, 
    nms[inds[2]], 
    0.8 * sizes[1,inds[2][1]]/2^30 * (nms[inds[2]]/nms[inds[2][1]]).^powers[2], 
    label=labels[2], 
    line=(2,:dash,:black), legend=:bottomright
    )
display(pl)

savefig(pl, "/Users/beckman/Downloads/" * filename * ".pdf")