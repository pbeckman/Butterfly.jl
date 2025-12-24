
mutable struct ButterflyMatrix{Dx, Kx, Dw, Kw, Tx<:Number, Tw<:Number, T<:Number}
    # Vt[l+1][j,k] is transfer matrix between block row j 
    # and block column k at level l, or the basis if l ∈ {0, L}
    U         :: Vector{Matrix{Matrix{T}}}
    Vt        :: Vector{Matrix{Matrix{T}}}
    sk        :: Vector{Matrix{Vector{Int64}}}
    # trees in x and w
    trx       :: Tree{Dx, Kx, Tx}
    trw       :: Tree{Dw, Kw, Tw}
    # level L of factorization
    level     :: Int64
    # maximum sum of ranks at any level 
    # used to determine size of intermediate vectors in apply
    max_width :: Int64
    # temporary vectors used to apply matrix
    beta      :: Vector{Matrix{Vector{T}}}
end

Base.size(B::ButterflyMatrix) = (length(B.trx.inds), length(B.trw.inds))
Base.size(B::ButterflyMatrix, j) = size(B)[j]

Base.show(io::IO, B::ButterflyMatrix{Dx, Kx, Dw, Kw, Tx, Tw, T}) where{Dx, Kx, Dw, Kw, Tx<:Number, Tw<:Number, T<:Number} = print(io, "$(B.level)-level $T-type butterfly matrix of size $(size(B,1)) × $(size(B,2))")

Base.adjoint(B::ButterflyMatrix{Dx, Kx, Dw, Kw, Tx, Tw, T}) where{Dx, Kx, Dw, Kw, Tx<:Number, Tw<:Number, T<:Number} = Adjoint{T, ButterflyMatrix{Dx, Kx, Dw, Kw, Tx, Tw, T}}(B)

function LinearAlgebra.:*(B::ButterflyMatrix, src::Vector)
    dest = Vector{ComplexF64}(undef, size(B, 1))
    return mul!(dest, B, src)
end

function LinearAlgebra.mul!(dest::Vector, B::ButterflyMatrix{Dx, Kx, Dw, Kw, Tx, Tw, T}, src::Vector) where{Dx, Kx, Dw, Kw, Tx<:Number, Tw<:Number, T<:Number}
    L = B.level
    k0, sk = -1, -1
    for l=0:L
        for (j, ndj) in enumerate(LevelIterator(B.trx, l))
            pj = parent_ind(j, Kx)
            for (k, ndk) in enumerate(LevelIterator(B.trw, L-l))
                cks = child_inds(k, Kw)
                if l == 0
                    mul!(
                        B.beta[1][j,k], 
                        B.Vt[1][j,k], 
                        view(src, ndk.inds)
                        )
                else
                    B.beta[l+1][j,k] .= 0
                    k0 = 1
                    for ck in cks
                        sk = length(B.beta[l][pj,ck])
                        mul!(
                            B.beta[l+1][j,k], 
                            view(B.Vt[l+1][j,k], :, k0:(k0+sk-1)), 
                            B.beta[l][pj,ck],
                            1, 1
                            )
                        k0 += sk
                    end
                end
                if l == L
                    mul!(
                        view(dest, ndj.inds), 
                        B.U[L+1][j,k], 
                        B.beta[L+1][j,k]
                        )
                end
            end
        end
    end
    return dest
end

function LinearAlgebra.mul!(dest::Vector, Bc::Adjoint{T, ButterflyMatrix{Dx, Kx, Dw, Kw, Tx, Tw, T}}, src::Vector) where{Dx, Kx, Dw, Kw, Tx<:Number, Tw<:Number, T<:Number}
    B = Bc.parent
    L = B.level
    k0, sk = -1, -1
    map(l -> map(v -> begin v .= 0; end, B.beta[l]), 1:(L+1))
    for l=L:-1:0
        for (j, ndj) in enumerate(LevelIterator(B.trx, l))
            pj = parent_ind(j, Kx)
            for (k, ndk) in enumerate(LevelIterator(B.trw, L-l))
                cks = child_inds(k, Kw)
                if l == L
                    mul!(
                        B.beta[L+1][j,k], 
                        B.U[L+1][j,k]', 
                        view(src, ndj.inds)
                        )
                end
                if l == 0
                    mul!(
                        view(dest, ndk.inds), 
                        B.Vt[1][j,k]', 
                        B.beta[1][j,k]
                        )
                else
                    k0 = 1
                    for ck in cks
                        sk = length(B.beta[l][pj,ck])
                        mul!(
                            B.beta[l][pj,ck],
                            view(B.Vt[l+1][j,k], :, k0:(k0+sk-1))',
                            B.beta[l+1][j,k],
                            1, 1
                        )
                        k0 += sk
                    end
                end
            end
        end
    end
    return dest
end