module Butterfly

    using LinearAlgebra, StaticArrays, Printf

    export build_tree, root, butterfly_factorize

    include("trees.jl")
    include("butterfly_matrix.jl")
    include("factorize.jl")
    
end