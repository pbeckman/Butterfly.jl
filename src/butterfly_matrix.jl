
mutable struct ButterflyMatrix{Dx, Kx, Dw, Kw, T}
    # Vt[l+1][j,k] is transfer matrix between block row j 
    # and block column k at level l, or the basis if l ∈ {0, L}
    U         :: Vector{Matrix{Matrix{ComplexF64}}}
    Vt        :: Vector{Matrix{Matrix{ComplexF64}}}
    # trees in x and w
    Tx        :: Tree{Dx, Kx, T}
    Tw        :: Tree{Dw, Kw, T}
    # level L of factorization
    level     :: Int64
    # maximum sum of ranks at any level 
    # used to determine size of intermediate vectors in apply
    max_width :: Int64
    # temporary vectors used to apply matrix
    beta      :: Vector{Matrix{Vector{ComplexF64}}}
end

Base.size(B::ButterflyMatrix) = (length(B.Tx.inds), length(B.Tw.inds))
Base.size(B::ButterflyMatrix, j) = size(B)[j]

function LinearAlgebra.:*(B::ButterflyMatrix, src::Vector)
    dest = Vector{ComplexF64}(undef, size(B, 1))
    return mul!(dest, B, src)
end

# function sparse(B::ButterflyMatrix{dx, dw}, fac::Symbol, l::Int64) where{dx, dw}
#     blks = getfield(B, fac)[l+1]
    
# end

function mul!(dest::Vector, B::ButterflyMatrix{Dx, Kx, Dw, Kw}, src::Vector) where{Dx, Kx, Dw, Kw}
    L  = B.level
    k0, sk = -1, -1
    for l=0:L
        for (j, ndj) in enumerate(LevelIterator(B.Tx, l))
            pj = parent_ind(Dx, j, Kx)
            for (k, ndk) in enumerate(LevelIterator(B.Tw, L-l))
                cks = child_inds(Dw, k, Kw)
                if l == 0
                    B.beta[l+1][j,k] .= B.Vt[l+1][j,k] * view(src, ndk.inds)
                else
                    k0 = 1
                    B.beta[l+1][j,k] .= 0.0
                    for ck in cks
                        sk = length(B.beta[l][pj,ck])
                        B.beta[l+1][j,k] .+= B.Vt[l+1][j,k][:, k0:(k0+sk-1)] * B.beta[l][pj,ck]
                        k0 += sk
                    end
                end
                if l == L
                    dest[ndj.inds] .= B.U[l+1][j,k] * B.beta[l+1][j,k]
                end
            end
        end
    end
    return dest
end