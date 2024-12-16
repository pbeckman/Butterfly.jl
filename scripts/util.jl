function plot_quadtree!(pl, tree; kwargs...)
    ll, wd = tree.boundary.origin, tree.boundary.widths

    plot!(pl, [ll[1], ll[1]], [ll[2],ll[2]+wd[2]]; kwargs...)
    plot!(pl, [ll[1], ll[1]+wd[1]], [ll[2]+wd[2],ll[2]+wd[2]]; kwargs...)
    plot!(pl, [ll[1]+wd[1], ll[1]+wd[1]], [ll[2]+wd[2],ll[2]]; kwargs...)
    plot!(pl, [ll[1]+wd[1], ll[1]], [ll[2],ll[2]]; kwargs...)

    if !isnothing(tree.children)
        for child in tree.children
            plot_quadtree!(pl, child; kwargs...)
        end
    end

    return pl
end
 
plot_quadtree(tree; kwargs...) = plot_quadtree!(plot(), tree; kwargs...)

function plot_binarytree!(pl, tree; kwargs...)
    ll, wd = tree.boundary.origin[1], tree.boundary.widths[1]

    plot!(pl, [ll, ll+wd],  zeros(2); kwargs...)
    plot!(pl, [ll, ll],       [-1,1]; kwargs...)
    plot!(pl, [ll+wd, ll+wd], [-1,1]; kwargs...)

    if !isnothing(tree.children)
        for child in tree.children
            plot_binarytree!(pl, child; kwargs...)
        end
    end

    return pl
end

plot_binarytree(tree; kwargs...) = plot_binarytree!(plot(), tree; kwargs...)
