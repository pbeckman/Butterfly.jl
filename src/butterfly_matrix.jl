
mutable struct ButterflyMatrix{dx, dw}
    # Vt[l+1][j,k] is transfer matrix between block row j 
    # and block column k at level l, or the basis if l ∈ {0, L}
    U         :: Vector{Matrix{Matrix{ComplexF64}}}
    Vt        :: Vector{Matrix{Matrix{ComplexF64}}}
    # trees in x and w
    Tx        :: Tree{dx}
    Tw        :: Tree{dw}
    # level L of factorization
    level     :: Int64
    # maximum sum of ranks at any level 
    # used to determine size of intermediate vectors in apply
    max_width :: Int64
    # vectors used to apply matrix
    beta      :: Vector{Matrix{Vector{ComplexF64}}}
end

Base.size(B::ButterflyMatrix{dx, dw}) where{dx, dw} = (length(B.Tx.data.inds), length(B.Tw.data.inds))
Base.size(B::ButterflyMatrix{dx, dw}, j) where{dx, dw} = size(B)[j]

function LinearAlgebra.:*(B::ButterflyMatrix{dx, dw}, src::Vector) where{dx, dw}
    dest = Vector{ComplexF64}(undef, size(B, 1))
    return mul!(dest, B, src)
end

# function sparse(B::ButterflyMatrix{dx, dw}, fac::Symbol, l::Int64) where{dx, dw}
#     blks = getfield(B, fac)[l+1]
    
# end


function mul!(dest::Vector, B::ButterflyMatrix{dx, dw}, src::Vector) where{dx, dw}
    L = B.level
    for l=0:L
        for (j, ndj) in enumerate(LevelIterator(B.Tx, l))
            pj = parent_ind(dx, j)
            for (k, ndk) in enumerate(LevelIterator(B.Tw, L-l))
                cks = child_inds(dw, k)
                if l == 0
                    B.beta[l+1][j,k] .= B.Vt[l+1][j,k] * src[ndk.data.inds]
                else
                    B.beta[l+1][j,k] .= B.Vt[l+1][j,k] * vcat(
                        [B.beta[l][pj,ck] for ck in cks]...
                        )
                end
                if l == L
                    dest[ndj.data.inds] .= B.U[l+1][j,k] * B.beta[l+1][j,k]
                end
            end
        end
    end
    return dest
end

# function mul!(dest::Vector, B::ButterflyMatrix{dx, dw}, src::Vector) where{dx, dw}
#     tmp1 = Vector{ComplexF64}(undef, B.max_width)
#     tmp2 = Vector{ComplexF64}(undef, B.max_width)
#     L  = B.level
#     j0 = 1
#     @printf("\nLevel 0:\n")
#     for (j, ndj) in enumerate(LevelIterator(B.Tw, L))
#         sj = size(B.Vt[1][1,j], 1)
#         @printf("reading %i:%i, writing to %i:%i\n", ndj.data.inds[1], ndj.data.inds[end], j0, j0+sj-1)
#         tmp2[j0:(j0+sj-1)] .= B.Vt[1][1,j] * src[ndj.data.inds]
#         j0 += sj
#     end
#     @show tmp2

#     for l=1:L
#         @printf("\nLevel %i:\n", l)
#         j0, k0 = 1, 1
#         for (k, ndk) in enumerate(LevelIterator(B.Tw, L-l))
#             for (j, ndj) in enumerate(LevelIterator(B.Tx, l))
#                 sj, sk = size(B.Vt[l+1][j,k])
#                 # @show sj, sk
#                 tmp1[j0:(j0+sj-1)] .= B.Vt[l+1][j,k] * tmp2[k0:(k0+sk-1)]
#                 @printf("(j,k) = (%i,%i) : reading %i:%i, writing to %i:%i\n", j, k, k0, k0+sk-1, j0, j0+sj-1)
#                 j0 += sj
#                 if mod(j, 2^dx) == 0
#                     k0 += sk
#                 end
#             end
#             # @show child_inds(dw, parent_ind(dw, k))
#             # k0 += sum(size.(B.Vt[l+1][child_inds(dw, parent_ind(dw, k)), k], 1))
#             # @show size.(B.Vt[l+1][:, k], 1)
#             # k0 += sum(size.(B.Vt[l+1][:, k], 1))
#         end
#         tmp2 .= tmp1
#         @show tmp2
#     end

#     j0, k0 = 1, 1
#     @printf("\nLevel %i (applying U):\n", L)
#     for (j, ndj) in enumerate(LevelIterator(B.Tw, L))
#         sj, sk = size(B.U[L+1][j,1])
#         dest[j0:(j0+sj-1)] .= B.U[L+1][j,1] * tmp2[k0:(k0+sk-1)]
#         @printf("reading %i:%i, writing to %i:%i\n", k0, k0+sk-1, j0, j0+sj-1)
#         j0 += sj
#         k0 += sk
#     end
#     @show dest
#     return dest
# end