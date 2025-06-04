
function butterfly_factorize(
    kernel, xs, ws;
    tol=1e-8, 
    L=max(0, floor(Int64, max(log(Kx^Dx, nx), log(Kw^Dw, nw))) - 3), 
    trx::Tree{Dx, Kx, Tx}=build_tree(xs, max_levels=L, min_pts=-1), 
    trw::Tree{Dw, Kw, Tw}=build_tree(ws, max_levels=L, min_pts=-1), 
    verbose=0, stop_at_level=Inf, method=:ID, T=ComplexF64, kwargs...
    ) where {Dx, Kx, Dw, Kw, Tx, Tw}
    nx = size(xs, 2)
    nw = size(ws, 2)

    sk   = Vector{Matrix{Vector{Int64}}}(undef, L+1)
    U    = Vector{Matrix{Matrix{T}}}(undef, L+1)
    Vt   = Vector{Matrix{Matrix{T}}}(undef, L+1)
    beta = Vector{Matrix{Vector{T}}}(undef, L+1)

    if verbose >= 1
        @printf("\nlevel  |         ranks         time\n-------|---------------------------------------\n")
    end

    max_width = -1
    trx = root(trx)
    trw = root(trw)
    tt = @elapsed for l=0:L
        if l == stop_at_level
            return ButterflyMatrix(U, Vt, sk, trx, trw, L, max_width, beta)
        end
        max_rank, min_rank = -1, max(nx, nw)
        level_width = 0
        
        sk[l+1]   = Matrix{Vector{Int64}}(undef, Kx^(Dx*l), Kw^(Dw*(L-l)))
        U[l+1]    = Matrix{Matrix{T}}(undef,     Kx^(Dx*l), Kw^(Dw*(L-l)))
        Vt[l+1]   = Matrix{Matrix{T}}(undef,     Kx^(Dx*l), Kw^(Dw*(L-l)))
        beta[l+1] = Matrix{Vector{T}}(undef,     Kx^(Dx*l), Kw^(Dw*(L-l)))
        t = @elapsed for (k, ndk) in enumerate(LevelIterator(trw, L-l))
            cks = child_inds(Dw, k, Kw)
            for (j, ndj) in enumerate(LevelIterator(trx, l))
                pj = parent_ind(Dx, j, Kx)
                # write factors
                Ujk, Vt[l+1][j,k], skjk, r = get_factors(
                    xs, ws, kernel, ndj, ndk,
                    (l==0) ? nothing : view(U[l],  pj, cks),
                    (l==0) ? nothing : view(sk[l], pj, cks), 
                    l, L,
                    method=method, tol=tol, verbose=verbose; kwargs...
                    )
                if !isnothing(Ujk);   U[l+1][j,k] = Ujk;  end
                if !isnothing(skjk); sk[l+1][j,k] = skjk; end
                # preallocate beta based on factor size
                beta[l+1][j,k] = Vector{T}(undef, r)
                
                # update max and min ranks
                min_rank = min(min_rank, r)
                max_rank = max(max_rank, r)
                # update level width 
                level_width += r
            end
        end
        if l > 0
            # free memory from intermediate factors
            U[l] = Matrix{Matrix{T}}(undef, 0, 0)
        end
        max_width = max(max_width, level_width)

        if verbose >= 1
            @printf("%5i  |  %8i - %-8i  %.2e\n", l, min_rank, max_rank, t)
        end
    end
    if verbose >= 1
        @printf("\ntotal factorization time for %i by %i matrix to tolerance %.1e: %.2e s\n\n", nx, nw, tol, tt)
    end

    return ButterflyMatrix(U, Vt, sk, trx, trw, L, max_width, beta)
end

subsample_inds(inds, s) = (s >= length(inds)) ? inds : inds[round.(Int64, range(1, length(inds), s))]

parent_ind(d, j, k) = floor(Int64, (j-1)/k^d)+1
child_inds(d, j, k) = (j-1)*(k^d) .+ (1:(k^d))

function get_factors(
    xs, ws, kernel, ndj, ndk, Uls, sks, l, L; 
    method=:ID, tol=1e-15, verbose=0, kwargs...
    )
    if method == :ID
        if haskey(kwargs, :os)
            os    = kwargs[:os]
            nj    = length(ndj.inds)
            kinds = (l==0) ? ndk.inds : vcat(sks...)
            nk    = length(kinds)
            njss  = min(nj, ceil(Int64, os * nk))
            ji    = randperm(nj)[1:njss]
            
            Mss = kernel(
                view(xs, :, view(ndj.inds, ji)), 
                view(ws, :, kinds)
                )

            F  = idfact!(Mss, rtol=tol)
            sk = kinds[F.sk]

            if l != L
                U = nothing
            else
                U = kernel(
                    view(xs, :, ndj.inds),
                    view(ws, :, sk)
                    )
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
            M = kernel(view(xs,:,ndj.inds), view(ws,:,ndk.inds))
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
            return U, Vt, 0
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