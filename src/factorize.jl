
function butterfly_factorize(
    kernel, xs, ws;
    tol=1e-8, L=nothing, 
    Tx::Tree{Dx, Kx, T}=nothing, Tw::Tree{Dw, Kw, T}=nothing, 
    verbose=false, stop_at_level=Inf
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

    U    = Vector{Matrix{Matrix{ComplexF64}}}(undef, L+1)
    Vt   = Vector{Matrix{Matrix{ComplexF64}}}(undef, L+1)
    beta = Vector{Matrix{Vector{ComplexF64}}}(undef, L+1)

    if verbose
        @printf("level  |         ranks         time\n-------|---------------------------------------\n")
    end

    max_width = -1
    for l=0:L
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
                if l == 0
                    # evaluate kernel on block column to build first factor
                    F = svd(
                        kernel(xs[:,ndj.inds], ws[:,ndk.inds])
                        )
                else
                    # compute nested factors from previous level
                    F = svd(
                        hcat(
                            [U[l][pj,ck][ndj.loc_inds, :] for ck in cks]...
                        )
                    )
                end
                # compute epsilon rank of block
                if length(F.S) == 0
                    r = 0
                else
                    r = findfirst(F.S/F.S[1] .< tol)
                    r = isnothing(r) ? length(F.S) : r-1
                end
                # write factors
                U[l+1][j,k]  = F.U[:,1:r] .* F.S[1:r]'
                Vt[l+1][j,k] = F.Vt[1:r,:]
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

        if verbose
            @printf("%5i  |  %8i - %-8i  %.2e\n", l, min_rank, max_rank, t)
        end
    end

    return ButterflyMatrix(U, Vt, Tx, Tw, L, max_width, beta)
end

parent_ind(d, j, k) = floor(Int64, (j-1)/k^d)+1
child_inds(d, j, k) = (j-1)*(k^d) .+ (1:(k^d))