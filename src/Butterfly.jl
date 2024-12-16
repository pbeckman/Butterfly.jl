module Butterfly

    using LinearAlgebra, StaticArrays, RegionTrees, Printf

    export build_tree, get_root, get_all_level, butterfly_factorize

    include("trees.jl")
    include("butterfly_matrix.jl")
    include("factorize.jl")
    
end