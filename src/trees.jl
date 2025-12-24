# integer nth root
inrt(K::Int, D::Int) = Int(K^(1/D))

# adapted from the RegionTrees.jl package
struct HyperRectangle{D, T}
    ll     :: SVector{D, T}
    widths :: SVector{D, T}
end

function divisions(H::HyperRectangle{D, T}, s) where {D, T}
    H.ll .+ H.widths .* (0.5^log2(s) * (1:(s-1))')
end

mutable struct Tree{D, K, T} 
    level     :: Int64
    boundary  :: HyperRectangle{D, T}
    inds      :: AbstractVector{Int64}
    loc_inds  :: AbstractVector{Int64}
    children  :: Union{AbstractArray{Tree{D, K, T}}, Nothing}
    parent    :: Union{Tree{D, K, T}, Nothing}
    next      :: Union{Tree{D, K, T}, Nothing} 
    pos       :: Int64
end

Tree(level, boundary::HyperRectangle{D, T}, inds, loc_inds, children, parent, next, pos) where {D, T} = 
    Tree{D, 2^D, T}(level, boundary, inds, loc_inds, children, parent, next, pos)

Tree(ll::SVector{D, T}, width::SVector{D, T}, n::Int64, k::Int64) where {D, T} =
    Tree{D, k, T}(0, HyperRectangle(ll, width), collect(1:n), collect(1:n), nothing, nothing, nothing, 1)

@inline isleaf(tree::Tree) = tree.children === nothing
@inline isroot(tree::Tree) = tree.parent === nothing

Base.show(io::IO, tree::Tree{D, K}) where {D, K} = print(io, "$(depth(tree)+1)-level $D-dimensional $K-ary tree on $(
    join(collect.(zip(tree.boundary.ll, tree.boundary.ll+tree.boundary.widths)), " × ")
)")

@inline getindex(tree::Tree, I) = getindex(tree.children, I)
@inline getindex(tree::Tree, I...) = getindex(tree.children, I...)

function child_boundary(tree::Tree{D, K, T}, indices) where {D, K, T}
    # divide into s = K^(1/D) subdomains
    s = inrt(K, D) 
    ch_widths = tree.boundary.widths ./ s
    return HyperRectangle(
        tree.boundary.ll .+ (SVector{D, T}(indices) .- 1) .* ch_widths,
        ch_widths
        )
end

function bounding_box(pts::Matrix{T}) where T
    d = size(pts, 1)
    return HyperRectangle(
        SVector{d, T}(minimum(pts, dims=2)),
        SVector{d, T}(maximum(pts, dims=2) .- minimum(pts, dims=2)),
        )
end

function split_regular!(node::Tree{D, K, T}, ch_inds) where {D, K, T}
    s = inrt(K, D) 
    node.children = [
        Tree(
            node.level+1,
            child_boundary(node, I.I),
            node.inds[ch_inds .== i],
            eachindex(node.inds)[ch_inds .== i],
            nothing, node, nothing, K*(node.pos-1) + i
            ) for (i, I) in enumerate(CartesianIndices(ntuple(_ -> s, D)))
        ]
end

function split_meshfree!(node::Tree{D, K, T}, pts, ch_inds) where {D, K, T}
    node.children = [
        Tree(
            node.level+1,
            bounding_box(pts[:, node.inds[ch_inds .== i]]),
            node.inds[ch_inds .== i],
            eachindex(node.inds)[ch_inds .== i],
            nothing, node, nothing, K*(node.pos-1) + i
            ) for i=1:K
        ]
end

function build_tree(pts::Matrix{T}; 
    max_levels=nothing, min_pts=1,
    ll=nothing, widths=nothing, sort=false, k=2^size(pts,1), 
    lap=nothing, sf=nothing
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
    refine_tree!(
        tree, pts; 
        max_levels=max_levels, min_pts=min_pts, lap=lap, sf=sf
        )
    add_next_pointers!(tree) # TODO : only works if tree is uniform
    
    # compute sorting permutation
    iter = LevelIterator(tree, depth(tree))
    perm = tree.inds[vcat([nd.inds for nd in iter]...)]
    if sort
        tree.inds .= tree.inds[perm]
        pts       .= pts[:, perm]
    end

    return tree, perm
end

function refine_tree!(tree::Tree{D, K, T}, pts::Matrix; 
    max_levels=Inf, min_pts=1, lap=nothing, sf=nothing
    ) where{D, K, T}
    n = size(pts, 2)

    ch_inds   = Vector{Int64}(undef, n)
    to_refine = Vector{Tree{D, K, T}}([tree])
    while !isempty(to_refine)
        node = pop!(to_refine)
        nn   = length(node.inds)

        if !isnothing(lap)
            # determine child boxes by sign of second eigenfunction of Laplacian
            lap_submatrix = lap[node.inds, node.inds]
            phi2 = eigsolve(
                    v -> lap_submatrix * v, 
                    nn, 2, :SR, verbosity=0, issymmetric=true
                )[2][2]
            ch_inds = 1 .+ (phi2 .> 0)
            if sum(ch_inds .== 1) < nn/8 || sum(ch_inds .== 2) < nn/4
                # tree is becoming unbalanced, prioritize balance (stupidly)
                @warn("tree is becoming unbalanced... switching from intrinsic to extrinsic metric.")
                xs   = @view(pts[:,node.inds])
                _, I = findmax(
                    [norm(xi - xj) for xi in eachcol(xs), xj in eachcol(xs)]
                    )
                i1, i2  = Tuple(I)
                ch_inds = 1 .+ [norm(xs[:, i1] - xj) < norm(xs[:, i2] - xj) for xj in eachcol(xs)]
            end
            split_meshfree!(node, pts, view(ch_inds, 1:nn))
        elseif !isnothing(sf)
            # determine child boxes by SurfaceFun label 
            ch_inds = 1 .+ Base.getindex.(
                reverse.(digits.(sf[node.inds], base=2)), 
                node.level + 2
                )
            split_meshfree!(node, pts, view(ch_inds, 1:nn))
        else
            s    = inrt(K, D) 
            divs = divisions(node.boundary, s)
            for j=1:nn
                # determine child boxes each point belongs in by divisions
                ch_inds[j] = parse(Int64, 
                reverse(join(
                    sum(pts[:,node.inds[j]] .> divs, dims=2)
                    )), 
                    base=s) .+ 1
            end
            split_regular!(node, view(ch_inds, 1:nn))
        end
        
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
        for _=1:K^l
            if j == K
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

Base.length(iter::LevelIterator{D, K}) where {D, K} = K^iter.level

# Base.firstindex(iter::LevelIterator{D, K}) where {D, K} = 1
# Base.firstindex(iter::Base.Iterators.Enumerate{LevelIterator{D, K}}) where {D, K} = 1
# function Base.getindex(iter::Base.Iterators.Enumerate{LevelIterator{D, K}}) where {D, K}

# end

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

mutable struct PostOrderIterator{D, K}
    tree :: Tree{D, K}
end

Base.length(iter::PostOrderIterator{D, K}) where {D, K} = Int(
    (K^(depth(root(iter.tree))+1) - 1) / (K - 1)
    )

function Base.iterate(iter::PostOrderIterator{D, K}, state=1) where {D, K}
    if state > length(iter)
        return nothing
    elseif state == 1
        iter.tree = root(iter.tree)
        # go down tree to left-most node at leaf level
        while !isnothing(iter.tree.children)
            iter.tree = iter.tree.children[1]
        end
    elseif mod(iter.tree.pos-1, K)+1 < K
        # go to right sibling
        iter.tree = iter.tree.parent.children[mod(iter.tree.pos-1,K)+1 + 1]
        # go to left-most node at leaf level to process all children
        while !isnothing(iter.tree.children)
            iter.tree = iter.tree.children[1]
        end
    else
        # all children have been processed; move up to parent
        iter.tree = iter.tree.parent
    end
    return (iter.tree, state+1)
end