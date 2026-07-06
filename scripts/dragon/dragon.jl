using Butterfly, LinearAlgebra, DelimitedFiles, Printf, Plots, LaTeXStrings, JLD, KrylovKit, IterativeSolvers
import Base.Iterators: partition

mesh_file     = "dragon-reordered.csv"
tree_file     = "dragon-tree.csv"
block_file    = "dragon-blocks.csv"
file_template = "output/dragon_eigenfunctions_n460448_blsz250"
bs            = 0:24
files         = [file_template * "_b$b.bin" for b=bs]
dedup_files   = [file_template * "_b$(b)_dedup.bin" for b=bs]

xs      = Matrix(readdlm(mesh_file, ',', Float64)')
tree_sf = vec(readdlm(tree_file, ',', Int64))

n   = size(xs, 2)
npa = div(n, 16)
vec_to_tensor(v) = reshape(
    vcat(reshape.(collect(partition(v, 4*npa)), 4, :)...), 4, 4, :
    )
tensor_to_vec(T) = vec(vcat([T[:,:,i] for i in axes(T,3)]...))

function read_slice(file)
    M = Matrix{Float64}(undef, 2n+2, 250)
    read!(file, M)
    return M[[1; 3:(n+2)],:] + im*M[[2; (n+3):end],:]
end

function split_conjugates(M)
    conj_inds = findall(imag.(M[1,:]) .!= 0)
    nc        = div(length(conj_inds), 2)
    @printf("found %i conjugate pairs\n", nc)
    println(conj_inds)
    
    if conj_inds[1] == 1 && all(
        (conj_inds[3 .+ (0:2:(2nc-4))] .- 1) .== conj_inds[2 .+ (0:2:(2nc-4))]
        )
        # if first eigenvalue is second of a conjugate pair, drop it
        M         = M[:,2:end]
        conj_inds = conj_inds[2:end] .- 1
    end
    if isodd(length(conj_inds)) && all(
        (conj_inds[2 .+ (0:2:(2nc-4))] .- 1) .== conj_inds[1 .+ (0:2:(2nc-4))]
        )
        # if last eigenvalue is first of a conjugate pair, drop it
        M         = M[:,1:end-1]
        conj_inds = conj_inds[1:end-1]
    end

    # imaginary parts should only occur in conjugate pairs
    @assert iseven(length(conj_inds))
    @assert all((conj_inds[2:2:end].-1) .== conj_inds[1:2:end])

    # split duplicates into real and imaginary parts
    M[2:end,conj_inds[2:2:end]] .= imag.(M[2:end,conj_inds[1:2:end]])

    return real.(M)
end

# whether to remove redundant eigenfunctions from adjacent slices and split
# eigenfunctions which are complex conjugates due to numerical asymmetry
deduplicate = false

if deduplicate
    blszs = zeros(Int64, length(files))

    @printf("reading in slices %i and %i...\n", bs[1], bs[2])
    global M1 = read_slice(files[1])
    global M2 = read_slice(files[2])

    @printf("splitting conjugate eigenpairs in slice %i...\n", bs[1])
    global M1 = split_conjugates(M1)

    @printf("writing de-duplicated slice %i...\n", bs[1])
    write(dedup_files[1], [M1[1,:]'; M1[2:end,:]], bs[1])
    blszs[1] = size(M1, 2)

    for k=1:length(files)-1
        @printf("\nsplitting conjugate and removing duplicate eigenpairs in slice %i...\n", bs[1]+k)
        global M2 = split_conjugates(M2)
        @show extrema(M1[1,:]), extrema(M2[1,:])

        eigval_err, i = findmin(
            [norm(M1[1,end-i+1:end] - M2[1,1:i], Inf) ./ norm(M2[1,1:i], Inf) for i=1:size(M1,2)]
        )
        @printf("%.2e relative error when aligning duplicate eigenpairs between slices\n", eigval_err)
        for (l1, l2) in zip(M1[1,end-i+1:end], M2[1,1:i])
            @printf("%.8f ≈ %.8f\n", l1, l2)
        end
        # throw an error if the eigenvalues from different slices don't match
        @assert eigval_err < 1e-2

        # remove overlapping eigenpairs from lower part of M2
        global M2 = M2[:,(i+1):end]
        @printf("removed %i duplicates\n%i eigenpairs remaining in slice %i\n", i, size(M2,2), bs[1]+k)

        @printf("writing de-duplicated slice %i...\n", bs[1]+k)
        write(dedup_files[k+1], [M2[1,:]'; M2[2:end,:]])
        blszs[k+1] = size(M2, 2)

        # read in next slice
        if k < length(files)-1
            @printf("reading in slice %i...\n", bs[1]+k+1)
            global M1 = M2
            global M2 = read_slice(files[k+2])
        end
        writedlm(block_file, blszs)
    end
end

blszs   = vec(readdlm(block_file, ',', Int64))
bl_bnds = 1 .+ [sum(blszs[1:k]) for k=0:length(blszs)]

## Build full dense MHT matrix (not recommended for large m)

Phi, Lam = nothing, nothing
for (i, dedup_file) in enumerate(dedup_files)
    @show dedup_file
    M = fill(NaN, n+1, blszs[i])
    read!(dedup_file, M)
    if i == 1
        global Lam = M[1, :]
        # global Phi = M[2:end, :]
    else
        global Lam = [Lam; M[1, :]]
        # global Phi = [Phi M[2:end, :]]
    end
end

function get_columns(ks) 
    println(ks)
    return Phi[:, ks]
end

##

function get_columns(ks)
    bl  = findlast(bl_bnds .<= ks[1])
    M   = Matrix{Float64}(undef, n+1, blszs[bl])
    read!(dedup_files[bl], M)
    if ks[end] < bl_bnds[bl+1]
        @printf(
            "for k = %i:%i, taking columns %i:%i from %s\n", 
            ks[1], ks[end],
            ks[1]-bl_bnds[bl]+1, ks[end]-bl_bnds[bl]+1,
            dedup_files[bl]
            )
        Phi = M[2:end, (ks[1]:ks[end]) .- bl_bnds[bl] .+ 1]
    else
        @printf(
            "for k = %i:%i, taking columns %i:%i from %s\n", 
            ks[1], ks[end],
            ks[1]-bl_bnds[bl]+1, size(M,2), 
            dedup_files[bl]
            )
        Phi = M[2:end, (ks[1]-bl_bnds[bl]+1):end]
        M   = Matrix{Float64}(undef, n+1, blszs[bl+1])
        read!(dedup_files[bl+1], M)
        @printf(
            "for k = %i:%i, taking columns %i:%i from %s\n", 
            ks[1], ks[end],
            1, ks[end]-bl_bnds[bl+1]+1, 
            dedup_files[bl+1]
            )
        Phi = [Phi M[2:end, 1:(ks[end]-bl_bnds[bl+1]+1)]]
    end
    return Phi
end

function dense_apply(v)
    out = zeros(Float64, n)
    i0  = 1
    for bl in blszs
        out .+= get_columns(i0:(i0+bl-1)) * v[i0:(i0+bl-1)]
        i0   += bl
    end
    return out
end

##

tol = 1e-3

m = bl_bnds[bs[end]+2]-1
L = 10 # min(9, floor(Int64, log2(m) - 2))
@printf("Computing trees of depth L = %i\n", L)

sf = tensor_to_vec(stack([fill(t, 4, 4) for t in tree_sf], dims=3))
trx, permx = build_tree(
    xs, max_levels=L, min_pts=-1, sf=sf, k=2
    )
trw, _     = build_tree(
    Float64.(collect((1:m)')), max_levels=L, min_pts=-1
    )

# compute butterfly factorization of the manifold harmonic transform
B = butterfly_factorize(
    get_columns, xs, collect((1:m)'); 
    L=L, trx=trx, trw=trw, tol=tol, T=Float64, method=:ID, os=5,
    verbose=true
    );

dense_bytes = sizeof(eltype(B.Vt[1][1,1])) * prod(size(B)) 
butterfly_bytes = Base.summarysize(B.Vt) + Base.summarysize(B.U) + Base.summarysize(B.sk)

@printf("\nSize of factorization : %s\n", Base.format_bytes(butterfly_bytes))
@printf("Size of dense matrix : %s\n", Base.format_bytes(dense_bytes))
@printf("Compression ratio : %.2f\n", dense_bytes / butterfly_bytes)
v  = randn(m)
println("\nDense matvec : ")
w  = @time dense_apply(v)
println("\nButterfly matvec : ")
wb = @time B*v
@printf(
    "\nRelative apply error  : %.2e\n", 
    norm(w - wb) / norm(w)
    )

##

function apply(v, flag)
    if flag === Val(true)
        return B'*v
    else
        return B*v
    end
end

# compute coefficients by least squares
# C = Phi \ xs'
lsqr_tol = tol
damp = 0
C_x = lsqr(B, xs[1,:], damp=damp, verbose=true, atol=lsqr_tol, btol=lsqr_tol)
C_y = lsqr(B, xs[2,:], damp=damp, verbose=true, atol=lsqr_tol, btol=lsqr_tol)
C_z = lsqr(B, xs[3,:], damp=damp, verbose=true, atol=lsqr_tol, btol=lsqr_tol)
C = [C_x C_y C_z]
@printf(
    "absolute reconstruction error : %.2e (rel 2-norm), %.2e (max)\n", 
    norm(B*C - xs') / norm(xs), maximum(norm.(eachrow(B*C - xs')))
    )

##

lintoexp(t, s=100) = (exp.(s*t).-1) / (exp(s)-1)

function Ft(t, lam)
    if t < 1 # flat
        return 1
    elseif t < 4 # low pass down
        c = 20_000 - 20_000*lintoexp((t-1)/3, -10)
        w = 10
        return (lam < c) ? 1 : exp(-((lam - c)/w)^2)
    elseif t < 8 # band bump up
        w0 = 10 + 50_000lintoexp((t-4)/4, -1)
        c = 10 + 20_000*lintoexp((t-4)/4, 3)
        w = 10 + 490*lintoexp((t-4)/4, -0.1)
        a = t < 6 ? 15lintoexp((t-4)/2, 1) : 15(1-lintoexp((t-6)/2, 5))
        return exp(-(lam/w0)^2) + a*exp(-((lam - c)/w)^2)
    else
        return 1
    end
end

##

fps = 8
nf  = 8*fps
for (f, t) in enumerate(range(0, 8, nf+1)[1:end-1])
    xs_filtered = (B * (C .* Ft.(t, Lam)))'
    open("output/dragon_frame$f.csv", "w") do file
        writedlm(
            file, 
            [
                reshape(vec_to_tensor(xs_filtered[1,:]), :, 1) reshape(vec_to_tensor(xs_filtered[2,:]), :, 1) reshape(vec_to_tensor(xs_filtered[3,:]), :, 1)
                ], 
            ','
            )
    end
end

##

xs_filtered = (B * (C .* Ft.(4.05, Lam)))'
open("output/dragon_frame_t4.csv", "w") do file
    writedlm(
        file, 
        [
            reshape(vec_to_tensor(xs_filtered[1,:]), :, 1) reshape(vec_to_tensor(xs_filtered[2,:]), :, 1) reshape(vec_to_tensor(xs_filtered[3,:]), :, 1)
            ], 
        ','
        )
end

##

F(lam) = (lam < 1000) ? 1 : 0
xs_filtered = (B * (C .* F.(Lam)))'
open("output/dragon_smoothed.csv", "w") do file
    writedlm(
        file, 
        [
            reshape(vec_to_tensor(xs_filtered[1,:]), :, 1) reshape(vec_to_tensor(xs_filtered[2,:]), :, 1) reshape(vec_to_tensor(xs_filtered[3,:]), :, 1)
            ], 
        ','
        )
end