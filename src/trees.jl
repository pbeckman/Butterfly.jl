struct HyperRectangle{D, T}
    ll     :: SVector{D, T}
    widths :: SVector{D, T}
end

divisions(H::HyperRectangle{D, T}, k) where {D, T} = H.ll .+ H.widths .* (0.5^log2(k) * (1:(k-1))')

mutable struct Tree{D, K, T} 
    level     :: Int64
    boundary  :: HyperRectangle{D, T}
    inds      :: AbstractVector{Int64}
    loc_inds  :: AbstractVector{Int64}
    children  :: Union{AbstractArray{Tree{D, K, T}}, Nothing}
    parent    :: Union{Tree{D, K, T}, Nothing}
    next      :: Union{Tree{D, K, T}, Nothing}
end

Tree(level, boundary::HyperRectangle{D, T}, inds, loc_inds, children, parent, next, k=2) where {D, T} = 
    Tree{D, k, T}(level, boundary, inds, loc_inds, children, parent, next)

Tree(ll::SVector{D, T}, width::SVector{D, T}, n::Int64, k::Int64) where {D, T} =
    Tree{D, k, T}(0, HyperRectangle(ll, width), collect(1:n), collect(1:n), nothing, nothing, nothing)

@inline isleaf(tree::Tree) = tree.children === nothing
@inline isroot(tree::Tree) = tree.parent === nothing

show(io::IO, tree::Tree{D, K}) where {D, K} = print(io, "$D-dimensional $K-ary tree on $(tree.boundary)")

@inline getindex(tree::Tree, I) = getindex(tree.children, I)
@inline getindex(tree::Tree, I...) = getindex(tree.children, I...)

function child_boundary(tree::Tree{D, K, T}, indices) where {D, K, T}
    ch_widths = tree.boundary.widths ./ K
    HyperRectangle(
        tree.boundary.ll .+ (SVector{D, T}(indices) .- 1) .* ch_widths,
        ch_widths
        )
end

function split!(node::Tree{D, K, T}, ch_inds) where {D, K, T}
    node.children = [
        Tree(
            node.level+1,
            child_boundary(node, I.I),
            node.inds[ch_inds .== i],
            eachindex(node.inds)[ch_inds .== i],
            nothing, node, nothing
            ) for (i, I) in enumerate(CartesianIndices(ntuple(_ -> K, D)))
        ]
end

function build_tree(pts::Matrix{T}; 
    max_levels=nothing, min_pts=1,
    ll=nothing, widths=nothing, sort=false, k=2
    ) where {T}
    d, n = size(pts)
    if isnothing(max_levels)
        max_levels = floor(Int64, log(k, size(pts,2)))
    end

    if xor(isnothing(ll), isnothing(widths))
        error("Must provide both or neither of ll and width.")
    elseif isnothing(ll)
        ll     = SVector{d, T}(minimum.(eachrow(pts)))
        widths = SVector{d, T}(maximum.(eachrow(pts)) .- ll)
    end

    # build tree and add pointers across each level
    tree = Tree(ll, widths, n, k)
    refine_tree!(tree, pts; max_levels=max_levels, min_pts=min_pts)
    add_next_pointers!(tree)
    
    if sort
        perm = tree.inds[
            vcat([nd.inds for nd in LevelIterator(tree, depth(tree))]...)
            ]
        tree.inds .= tree.inds[perm]
        pts       .= pts[:, perm]
    end

    return tree
end

function refine_tree!(tree::Tree{D, K, T}, pts::Matrix{T}; 
    max_levels=Inf, min_pts=1
    ) where{D, K, T}
    n = size(pts, 2)

    ch_inds   = Vector{Int64}(undef, n)
    to_refine = Vector{Tree{D, K, T}}([tree])
    while !isempty(to_refine)
        node = pop!(to_refine)
        nn   = length(node.inds)
        divs = divisions(node.boundary, K)

        for j=1:nn
            # determine which of the K^D child boxes each point belongs in
            ch_inds[j] = parse(Int64, 
            reverse(join(
                sum(pts[:,node.inds[j]] .> divs, dims=2)
                )), 
                base=K) .+ 1
        end
        split!(node, view(ch_inds, 1:nn))

        if node.level < max_levels-1
            for child in node.children
                if length(child.inds) > min_pts
                    append!(to_refine, [child])
                end
            end
        end
    end
end

function add_next_pointers!(tree::Tree{D, K, T}) where{D, K, T}
    for l=1:depth(tree)
        # go down tree to left-most node at desired level
        tree = root(tree)
        for _=1:l
            tree = tree.children[1]
        end
        j = 1
        for _=1:K^(l*D)
            if j == K^D
                # if no right siblings, point to right cousin
                if !isnothing(tree.parent.next)
                    tree.next = tree.parent.next.children[1]
                    tree      = tree.next
                end
                j = 1
            else
                # point to right sibling
                tree.next = tree.parent.children[j+1]
                tree      = tree.next
                j += 1
            end
            
        end
    end
end

function root(tree)
    while !isnothing(tree.parent)
        tree = tree.parent
    end
    return tree
end

function depth(tree)
    tree = root(tree)
    dpt  = 0
    while !isleaf(tree)
        dpt += 1
        tree = tree.children[1]
    end
    return dpt
end

mutable struct LevelIterator{D, K}
    tree  :: Tree{D, K}
    level :: Int64
end

Base.length(iter::LevelIterator{D, K}) where {D, K} = K^(D * iter.level)

function Base.iterate(iter::LevelIterator{D, K}, state=1) where {D, K}
    if state > length(iter)
        return nothing
    elseif state == 1
        # go down tree to left-most node at desired level
        iter.tree = root(iter.tree)
        for _=1:iter.level
            iter.tree = iter.tree.children[1]
        end
        return (iter.tree, state+1)
    else
        iter.tree = iter.tree.next
        return (iter.tree, state+1)
    end
end