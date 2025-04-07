
function butterfly_factorize(
    kernel, xs, ws;
    tol=1e-8, L=nothing, 
    Tx::Tree{Dx, Kx, T}=nothing, Tw::Tree{Dw, Kw, T}=nothing, 
    verbose=0, stop_at_level=Inf, method=:ID, kwargs...
    ) where {Dx, Kx, Dw, Kw, T}
    nx = size(xs, 2)
    nw = size(ws, 2)

    if isnothing(L)
        L = max(0, floor(Int64, max(log(Kx^Dx, nx), log(Kw^Dw, nw))) - 4)
    end
    if isnothing(Tx)
        Tx = build_tree(xs, max_levels=L, min_pts=-1)
    end
    if isnothing(Tw)
        Tw = build_tree(ws, max_levels=L, min_pts=-1)
    end
    Tx = root(Tx)
    Tw = root(Tw)

    M    = Matrix{ComplexF64}(undef, nx, 2 * nw / Kw^(Dw*L))
    U    = Vector{Matrix{Matrix{ComplexF64}}}(undef, L+1)
    Vt   = Vector{Matrix{Matrix{ComplexF64}}}(undef, L+1)
    beta = Vector{Matrix{Vector{ComplexF64}}}(undef, L+1)

    if verbose >= 1
        @printf("\nlevel  |         ranks         time\n-------|---------------------------------------\n")
    end

    max_width = -1
    
    tt = @elapsed for l=0:L
        if l == stop_at_level
            return ButterflyMatrix(U, Vt, Tx, Tw, L, max_width, beta)
        end
        max_rank, min_rank = -1, max(nx, nw)
        level_width = 0
        
        U[l+1]    = Matrix{Matrix{ComplexF64}}(undef, Kx^(Dx*l), Kw^(Dw*(L-l)))
        Vt[l+1]   = Matrix{Matrix{ComplexF64}}(undef, Kx^(Dx*l), Kw^(Dw*(L-l)))
        beta[l+1] = Matrix{Vector{ComplexF64}}(undef, Kx^(Dx*l), Kw^(Dw*(L-l)))
        t = @elapsed for (k, ndk) in enumerate(LevelIterator(Tw, L-l))
            cks = child_inds(Dw, k, Kw)
            for (j, ndj) in enumerate(LevelIterator(Tx, l))
                pj = parent_ind(Dx, j, Kx)
                # write factors
                U[l+1][j,k], Vt[l+1][j,k], r = get_factors(
                    xs, ws, kernel, ndj, ndk, 
                    (l==0) ? nothing : view(U[l], pj, cks), M,
                    method=method, tol=tol, verbose=verbose; kwargs...
                    )
                # preallocate beta based on factor size
                beta[l+1][j,k] = Vector{ComplexF64}(undef, r)
                
                # update max and min ranks
                min_rank = min(min_rank, r)
                max_rank = max(max_rank, r)
                # update level width 
                level_width += r
            end
        end
        if l > 0
            # free memory from intermediate factors
            U[l] = Matrix{Matrix{ComplexF64}}(undef, 0, 0)
        end
        max_width = max(max_width, level_width)

        if verbose >= 1
            @printf("%5i  |  %8i - %-8i  %.2e\n", l, min_rank, max_rank, t)
        end
    end
    if verbose >= 1
        @printf("\ntotal factorization time for %i by %i matrix : %.2e s\n\n", nx, nw, tt)
    end

    return ButterflyMatrix(U, Vt, Tx, Tw, L, max_width, beta)
end

subsample_inds(inds, s) = (s >= length(inds)) ? inds : inds[round.(Int64, range(1, length(inds), s))]

parent_ind(d, j, k) = floor(Int64, (j-1)/k^d)+1
child_inds(d, j, k) = (j-1)*(k^d) .+ (1:(k^d))

function get_factors(
    xs, ws, kernel, ndj, ndk, Uls, M; 
    method=:ID, tol=1e-15, verbose=0, kwargs...
    )
    if method == :ID
        if haskey(kwargs, :os)
            os = kwargs[:os]
            nj = length(ndj.inds)
            
            if isnothing(Uls)
                nk   = length(ndk.inds)
                njss = min(nj, ceil(Int64, os * nk))
                ji   = randperm(nj)[1:njss]
                M    = reshape(M, njss, :)
                M[:,1:nk] .= kernel(
                    view(xs,:,view(ndj.inds,ji)), 
                    view(ws,:,ndk.inds)
                    )

                if verbose >= 2
                    @printf("\tsubsampling %i by %i block to size %i by %i\n", nj, nk, size(Mss)...)
                end
            else
                nk   = sum(size.(Uls,2)) 
                njss = min(nj, ceil(Int64, os * nk))
                ji   = randperm(nj)[1:njss]
                M    = reshape(M, njss, :)
                k0   = 1
                for Ul in Uls
                    sk = size(Ul, 2)
                    M[:,k0:(k0+sk-1)] .= view(Ul, view(ndj.loc_inds,ji), :)
                    k0 += sk
                end
                    
                if verbose >= 2
                    @printf("\tsubsampling %i by %i block to size %i by %i\n", nj, sum(size.(Uls, 2)), size(Mss)...)
                end
            end

            F = idfact!(view(M,:,1:nk), rtol=tol)
        end

        M = reshape(M, length(ndj.inds), :)
        if isnothing(Uls)
            # evaluate kernel on block column to build first factor
            M[:,1:length(ndk.inds)] .= kernel(
                view(xs,:,ndj.inds), 
                view(ws,:,ndk.inds)
                )
        else
            # compute nested factors from previous level
            k0 = 1
            for Ul in Uls
                sk = size(Ul, 2)
                M[:,k0:(k0+sk-1)] .= view(Ul, ndj.loc_inds, :)
                k0 += sk
            end
        end
        
        if !haskey(kwargs, :os)
            F = idfact!(view(M,:,1:nk), rtol=tol)
        end
        U = M[:, F.sk]

        r  = length(F.sk)
        Vt = [Matrix(I, r, r) F.T][:, invperm(F[:p])]
    elseif method == :SVD
        if isnothing(Uls)
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
    end

    return U, Vt, r 
end