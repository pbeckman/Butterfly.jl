
function butterfly_factorize(
    get_columns, xs, ws;
    tol=1e-8, 
    L=max(0, floor(Int64, max(log(Kx, nx), log(Kw, nw))) - 3), 
    trx::Tree{Dx, Kx, Tx}=build_tree(xs, max_levels=L, min_pts=-1), 
    trw::Tree{Dw, Kw, Tw}=build_tree(ws, max_levels=L, min_pts=-1), 
    verbose=0, method=:ID, T=ComplexF64, kernel=nothing, kwargs...
    ) where {Dx, Kx, Dw, Kw, Tx, Tw}
    nx = size(xs, 2)
    nw = size(ws, 2)

    sk   = Vector{Matrix{Vector{Int64}}}(undef, L+1)
    U    = Vector{Matrix{Matrix{T}}}(undef, L+1)
    Vt   = Vector{Matrix{Matrix{T}}}(undef, L+1)
    beta = Vector{Matrix{Vector{T}}}(undef, L+1)
    for l=0:L
        sk[l+1]   = Matrix{Vector{Int64}}(undef, Kx^l, Kw^(L-l))
        U[l+1]    = Matrix{Matrix{T}}(undef,     Kx^l, Kw^(L-l))
        Vt[l+1]   = Matrix{Matrix{T}}(undef,     Kx^l, Kw^(L-l))
        beta[l+1] = Matrix{Vector{T}}(undef,     Kx^l, Kw^(L-l))
    end

    if verbose >= 1
        @printf("\nlevel  |         ranks         time\n-------|---------------------------------------\n")
    end

    min_ranks = fill(max(nx, nw), L+1)
    max_ranks = fill(-1, L+1)
    widths    = zeros(Int64, L+1)
    ts        = zeros(Float64, L+1)

    trx = root(trx)
    trw = root(trw)        
    t = @elapsed for ndk in PostOrderIterator(trw)
        l, k = L-ndk.level, ndk.pos
        cks  = child_inds(k, Kw)
        for (j, ndj) in enumerate(LevelIterator(trx, l))
            pj = parent_ind(j, Kx)
            # write factors
            ts[l+1] += @elapsed Ujk, Vt[l+1][j,k], skjk, r = get_factors(
                xs, ws, get_columns, ndj, ndk,
                (l==0) ? nothing : view(U[l],  pj, cks),
                (l==0) ? nothing : view(sk[l], pj, cks), 
                l, L,
                method=method, tol=tol, verbose=verbose, kernel=kernel; 
                kwargs...
                )
            if !isnothing(Ujk);   U[l+1][j,k] = Ujk;  end
            if !isnothing(skjk); sk[l+1][j,k] = skjk; end
            # preallocate beta based on factor size
            beta[l+1][j,k] = Vector{T}(undef, r)
            
            # update max and min ranks
            min_ranks[l+1] = min(min_ranks[l+1], r)
            max_ranks[l+1] = max(max_ranks[l+1], r)
            # update level width 
            widths[l+1] += r

            if l > 0 && mod(j, Kx) == 0 
                # free parent memory after processing all its child nodes
                for ck in cks
                    U[l][pj, ck] = Matrix{Matrix{T}}(undef, 0, 0)
                end
            end
        end
    end
    if verbose >= 1
        for l=0:L
            @printf(
                "%5i  |  %8i - %-8i  %.2e\n", 
                l, min_ranks[l+1], max_ranks[l+1], ts[l+1]
                )
        end
        @printf("\ntotal factorization time for %i by %i matrix to tolerance %.1e: %.2e s\n\n", nx, nw, tol, t)
    end

    return ButterflyMatrix(U, Vt, sk, trx, trw, L, maximum(widths), beta)
end

subsample_inds(inds, s) = (s >= length(inds)) ? inds : inds[round.(Int64, range(1, length(inds), s))]

parent_ind(j, k) = floor(Int64, (j-1)/k)+1
child_inds(j, k) = (j-1)*k .+ (1:k)

function get_factors(
    xs, ws, get_columns, ndj, ndk, Uls, sks, l, L; 
    method=:SVD, kernel=nothing, tol=1e-15, verbose=0, kwargs...
    )
    if method == :ID
        if haskey(kwargs, :os)
            os    = kwargs[:os]
            nj    = length(ndj.inds)
            kinds = (l==0) ? ndk.inds : vcat(sks...)
            nk    = length(kinds)
            njss  = min(nj, ceil(Int64, os * nk))
            ji    = randperm(nj)[1:njss]

            if !isnothing(kernel)
                Mss = kernel(
                    view(xs, :, view(ndj.inds, ji)), 
                    view(ws, :, kinds)
                    )
            else
                if l == 0
                    M = get_columns(ndk.inds)
                else
                    M = hcat([Ul[ndj.loc_inds, :] for Ul in Uls]...)
                end
                Mss = M[ji, :]
            end

            F  = idfact!(Mss, rtol=tol)
            sk = kinds[F.sk]

            if !isnothing(kernel) 
                if l != L
                    U = nothing
                else
                    U = kernel(
                        view(xs, :, ndj.inds),
                        view(ws, :, sk)
                        )
                end
            else
                U = M[:, F.sk]
            end
        else
            if l == 0
                M = kernel(
                    view(xs, :, ndj.inds), 
                    view(ws, :, ndk.inds)
                    )
            else
                M = hcat([Ul[ndj.loc_inds, :] for Ul in Uls]...)
            end

            F  = idfact(M, rtol=tol)
            U  = M[:, F.sk]
            sk = ndk.inds[F.sk]
        end
        
        r  = length(sk)
        Vt = [Matrix(I, r, r) F.T][:, invperm(F[:p])]
    elseif method == :SVD
        if l == 0
            # evaluate kernel on block column to build first factor
            M = get_columns(ndk.inds)
        else
            # compute nested factors from previous level
            M = hcat([Ul[ndj.loc_inds, :] for Ul in Uls]...)
        end

        # compute SVD
        F = svd(M)

        # unpack SVD into low-rank factors
        if length(F.S) == 0
            U  = zeros(Float64, size(F.U, 1), 0)
            Vt = zeros(Float64, 0, size(F.Vt, 2))
            return U, Vt, Vector{Int64}([]), 0
        else
            r = findfirst(F.S/F.S[1] .< tol)
            r = isnothing(r) ? length(F.S) : r-1
        end

        U  = F.U[:,1:r] .* F.S[1:r]'
        Vt = F.Vt[1:r,:]
        sk = nothing
    end

    return U, Vt, sk, r 
end