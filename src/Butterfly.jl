module Butterfly

    using LinearAlgebra, StaticArrays, LowRankApprox, KrylovKit, Printf
    import Random: seed!, randperm

    export build_tree, root, butterfly_factorize

    include("trees.jl")
    include("butterfly_matrix.jl")
    include("factorize.jl")
    
end