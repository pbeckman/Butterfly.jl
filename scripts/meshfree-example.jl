using Butterfly, LinearAlgebra, SparseArrays, NearestNeighbors, StatsBase, KrylovKit, FileIO, Printf, Plots, LaTeXStrings, JLD

function sparse_kernel_matrix(k, xs, tol; metric=Euclidean())
    # build KD tree
    tree = KDTree(xs, metric)
    # number of neighbors at which to evaluate kernel
    nn = 10

    I = Vector{Int64}([])
    J = Vector{Int64}([])
    V = Vector{Float64}([])
    for (i, x) in enumerate(eachcol(xs))
        enough_neighbors = false
        while !enough_neighbors
            # for each point, compute kernel at neighbors
            idxs, dists = knn(tree, x, nn)
            
            ks = [k(x, xj, dj) for (xj, dj) in zip(eachcol(xs[:,idxs]), dists)]

            # check if we've found all neighbors down to kernel tolerance
            enough_neighbors = any(ks .< tol)

            if enough_neighbors
                # take only kernel evals above tolerance
                idxs = idxs[ks .> tol]
                ks   = ks[ks .> tol]

                # add these entries to sparse matrix
                append!(I, fill(i, length(idxs)))
                append!(J, idxs)
                append!(V, ks)
            else
                nn *= 2
                if nn >= size(xs, 2)
                    # We asked for more than n neighbors -- K is dense
                    @warn("Kernel matrix K appears to be dense... returning dense matrix.")
                    return k.(
                        eachcol(xs), 
                        eachcol(xs)', 
                        [norm(xi - xj) for xi in eachcol(xs), xj in eachcol(xs)]
                        )
                end
            end
        end
    end

    return sparse(I, J, V)
end

mesh = load("/Users/beckman/Downloads/hand-refined.obj")

xs_mesh = stack(mesh.vertex_attributes.position)
n_mesh  = size(xs_mesh, 2)

# compute approximate "volume" of each point by distance to neighbors
tree_mesh = KDTree(xs_mesh)
nnn = 10
vol = [sum(knn(tree_mesh, x, nnn)[2]) / nnn for x in eachcol(xs_mesh)]

##

jld_file = "/Users/beckman/Downloads/hand-sizes.jld"

if isfile(jld_file)
    dict  = load(jld_file)
    tol   = dict["tol"]
    ns    = dict["ns"]
    ms    = dict["ms"]
    s2s   = dict["s2s"]
    sizes = dict["sizes"]
else
tol   = 1e-3
ns    = round.(Int64, 10 .^ range(3, log10(200_000), 10))
mf(n) = div(n, 50)
s2s   = [0.01]

sizes = Array{Int64}(undef, length(ns), length(s2s), 2)

for (ni, n) in enumerate(ns)
    @printf("==========\nn = %i\n", n)
    for (si, s2) in enumerate(s2s)
        @printf("----------\nσ² = %.2f\n", s2)
        m   = mf(n)
        xs  = xs_mesh[:, sample(1:n_mesh, Weights(vol), n)] .+ s2 * randn(3, n)
        rho = 0.005/n^(1/4) * 100s2
        k(x1, x2, r) = exp(-r^2/rho)

        t = @elapsed K = sparse_kernel_matrix(k, xs, tol)
        if issparse(K)
            @printf(
                "Computing sparse K for %i points (%.4f%% nonzeros to tolerance %.0e) : %.2e s\n", 
                n, 100 * nnz(K) / n^2, tol, t
                )
        else
            @printf(
                "Computing dense K for %i points : %.2e s\n", 
                n, t
                )
        end
        sqw  = sqrt.(K * ones(n))
        isqw = 1 ./ sqw
        lap  = Diagonal(ones(n)) - Diagonal(isqw) * (K * Diagonal(isqw))

        t = @elapsed Lam, Phi, info = eigsolve(
            v -> lap*v, n, m+1, :SR,
            krylovdim=max(30, m+10), verbosity=1, tol=tol, issymmetric=true
            )
        Lam = Lam[2:m+1]
        Phi = stack(Phi[2:m+1]) .* sqw
        @printf("Computing %i eigenvectors of K using Krylov : %.2e s\n", m, t)

        # pl = scatter3d(
        #     eachrow(xs)..., label="", 
        #     markersize=1, markerstrokewidth=0, 
        #     camera=(0,180), 
        #     marker_z=Phi[:,1],
        #     axis=([], false), dpi=300, c=:lightrainbow
        #     )
        # display(pl)

        L = floor(Int64, log2(m) - 3)
        @printf("Computing trees of depth L = %i\n", L)

        trx, permx = build_tree(
            xs, max_levels=L, min_pts=-1, lap=lap, k=2
            )
        trw, _     = build_tree(
            reshape(Lam[1:m], 1, :), max_levels=L, min_pts=-1
            )

        # compute butterfly factorization of the manifold harmonic transform
        B = butterfly_factorize(
            ks -> Phi[:, ks], xs, collect((1:m)'); 
            L=L, trx=trx, trw=trw, tol=tol, method=:SVD,
            verbose=true
            );

        dense_bytes = sizeof(eltype(B.Vt[1][1,1])) * prod(size(B)) 
        butterfly_bytes = Base.summarysize(B.Vt) + Base.summarysize(B.U) + Base.summarysize(B.sk)

        @printf("\nSize of factorization : %s\n", Base.format_bytes(butterfly_bytes))
        @printf("Size of dense matrix : %s\n", Base.format_bytes(dense_bytes))
        @printf("Compression ratio : %.2f\n", dense_bytes / butterfly_bytes)

        sizes[ni, si, 1] = dense_bytes
        sizes[ni, si, 2] = butterfly_bytes

        save(
            jld_file, 
            "tol", tol,
            "ns", ns,
            "ms", mf.(ns),
            "s2s", s2s,
            "sizes", sizes
            )

        # v = randn(m)

        # println("\nButterfly matvec : ")
        # wb = @time B*v

        # println("Dense matvec : ")
        # A = Phi[:,1:m]
        # w = @time A*v
        # @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
    end
end
end

##

pl_inds = 1:8
pl = plot(
    ns[pl_inds], sizes[pl_inds,1,1]/1024^3, 
    line=2, label="dense", legend=:topleft,
    xscale=:log10, yscale=:log10, ylims=(1e-4, 1e1), marker=2, markerstrokewidth=0
    )
plot!(pl,
    ns[pl_inds], sizes[pl_inds,:,2]/1024^3, 
    line=2, marker=2, markerstrokewidth=0, 
    label="dense", 
    labels=[L"\sigma^2 = 0.01" L"\sigma^2 = 0.02" L"\sigma^2 = 0.04"]
    )
ref_inds = round(Int64, 2/3*length(pl_inds)):length(pl_inds)
plot!(pl,
    ns[ref_inds], 0.3ns[ref_inds].^2/1024^3, 
    line=(2, :black, :dash), label=L"\mathcal{O}(n^2)"
)
plot!(pl,
    ns[ref_inds], 11 * ns[ref_inds].^(3/2)/1024^3, 
    line=(2, :grey, :dashdot), label=L"\mathcal{O}(n^{3/2})"
)

##

pl = scatter3d(
    eachrow(xs)..., label="", 
    markersize=1, markerstrokewidth=0, 
    camera=(0,180), 
    # marker_z=Phi[:,1],
    axis=([], false), dpi=300, c=:lightrainbow
    )