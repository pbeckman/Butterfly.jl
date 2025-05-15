using Butterfly, Plots, StaticArrays, LinearAlgebra, Printf, BenchmarkTools
import Base.Iterators: product, partition
import SphericalHarmonics: sphericalharmonic

include("util.jl")

# size of DFT to factorize
n1d = 2^8
n   = n1d^2
# number of eigenvectors to compress (or nothing to compress O(n) vectors)
m = 2000
# number of levels in factorization
L = floor(Int64, log(4, n)) - 4
# whether to compute spherical harmonic transform
SHT = false
# whether to use uniformly random xs
nonuniform = true
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
trx = build_tree(
    tree_xs, 
    max_levels=L, min_pts=1, sort=false, k=2
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
    trw = build_tree(
        ks, max_levels=L, min_pts=-1, sort=false, k=4, 
        ll=SVector{1, Float64}(1), widths=SVector{1, Float64}(length(ks)-1)
        )
else
    trw = build_tree(
        reshape(lams, 1, :), max_levels=L, min_pts=-1, sort=false, k=4
        )
end

# compute butterfly factorization of the manifold harmonic transform
B = butterfly_factorize(
    kernel, xs, ks; 
    L=L, trx=trx, trw=trw, tol=tol, verbose=true, method=:ID, os=3
    );

v = randn(size(ks,2))

@printf(
    "\nSize of factorization : %s\n", 
    Base.format_bytes(
        Base.summarysize(B.Vt) + Base.summarysize(B.U) + Base.summarysize(B.sk)
        )
    )
println("Butterfly matvec : ")
# wb = @btime $B*$v
wb = @time B*v

@printf(
        "\nSize of dense matrix : %s\n", 
        Base.format_bytes(
            sizeof(eltype(B.Vt[1][1,1])) * prod(size(B))
        )
    )
if prod(size(B)) < 20_000^2
    A  = kernel(xs, ks)
    println("Dense matvec : ")
    # w  = @btime $A*$v
    w = @time A*v
    @printf("\nRelative apply error  : %.2e\n", norm(w - wb) / norm(w))
end