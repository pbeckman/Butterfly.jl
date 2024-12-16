mutable struct Data
    level      :: Int64
    pos        :: Int64
    inds       :: AbstractVector{Int64}
    local_inds :: AbstractVector{Int64}
end

# Data(level, pos, inds) = Data(level, pos, inds)

Tree{d} = Cell{Data, d}

function build_tree(pts; 
    max_levels=floor(Int64, log(size(pts,1), size(pts,2))), min_pts=1,
    ll=nothing, width=nothing, sort=false
    )
    d, n = size(pts)

    if xor(isnothing(ll), isnothing(width))
        error("Must provide both or neither of ll and width")
    elseif isnothing(ll)
        ll    = SVector{d}(minimum.(eachrow(pts)))
        width = SVector{d}(maximum.(eachrow(pts)) .- ll)
    end

    tree = Cell(ll, width, Data(0, 0, collect(1:n), collect(1:n)))
    refine_tree!(tree, pts; max_levels=max_levels, min_pts=min_pts)
    
    if sort 
        perm = tree.data.inds[
            vcat([nd.data.inds for nd in LevelIterator(tree, depth(tree))]...)
            ]
        tree.data.inds .= tree.data.inds[perm]
        pts            .= pts[:, perm]
    end

    return tree
end

function refine_tree!(tree, pts; 
    max_levels=Inf, min_pts=1
    )
    d, n = size(pts)

    to_refine = Vector{Tree{d}}([tree])
    while !isempty(to_refine)
        node = pop!(to_refine)

        ch_inds = parse.(
            Int64, reverse.(join.(eachcol(Int64.(pts[:,node.data.inds] .> node.divisions)))), base=2
            ) .+ 1
        split!(
            node, 
            [Data(
                node.data.level+1, i, 
                node.data.inds[ch_inds .== i],
                eachindex(node.data.inds)[ch_inds .== i]
                ) for i=1:2^d]
            )

        if node.data.level < max_levels-1
            for child in node.children
                if length(child.data.inds) > min_pts
                    append!(to_refine, [child])
                end
            end
        end
    end
end

function get_root(tree)
    while !isnothing(tree.parent)
        tree = tree.parent
    end
    return tree
end

function depth(tree)
    tree = get_root(tree)
    dpt  = 0
    while !isleaf(tree)
        dpt += 1
        tree = tree.children[1]
    end
    return dpt
end

parent_ind(d, j) = floor(Int64, (j-1)/2^d)+1
child_inds(d, j) = (j-1)*(2^d) .+ (1:(2^d))

mutable struct LevelIterator{d}
    tree  :: Tree{d}
    level :: Int64
end

Base.length(iter::LevelIterator{d}) where{d} = 2^(d * iter.level)

function Base.iterate(iter::LevelIterator{d}, state=1) where{d}
    if state > length(iter)
        return nothing
    elseif state == 1
        # go down tree to left-most node at desired level
        iter.tree = get_root(iter.tree)
        for _=1:iter.level
            iter.tree = iter.tree.children[1]
        end
    elseif iter.tree.data.pos < 2^d
        # move to right sibling
        pos = iter.tree.data.pos
        iter.tree = iter.tree.parent.children[pos+1]
    else
        # if no right siblings, go up until there are
        l = 0
        while iter.tree.data.pos == 2^d 
            iter.tree = iter.tree.parent
            l += 1
        end
        pos = iter.tree.data.pos
        iter.tree = iter.tree.parent.children[pos+1]
        # go back down to desired level
        for _=1:l
            iter.tree = iter.tree.children[1]
        end
    end
    return (iter.tree, state+1)
end