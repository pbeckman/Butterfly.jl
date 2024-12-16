using Butterfly, Plots, StaticArrays, LinearAlgebra, RegionTrees, Morton, Printf

include("util.jl")

n1d  = 16
mort = sortperm(
        vcat(cartesian2morton.(collect.(Iterators.product(1:n1d, 1:n1d)))...)
    )

xs1d = range(0, 2pi, n1d+1)[1:end-1]
xs   = hcat(collect.(Iterators.product(xs1d, xs1d))...)
# xs  .= xs[:. mort]

ks1d  = 0:(n1d-1)
ks    = hcat(collect.(Iterators.product(ks1d, ks1d))...)
lmds  = norm.(eachcol(ks))
ks    = ks[:, lmds .< n1d]
lmds  = lmds[lmds .< n1d]

p     = sortperm(lmds)
lmds  = lmds[p]
ks    = ks[:,p]
ws    = reshape(lmds, 1, :)

max_levels = Int(log(2, n1d)) - 1

sp_tree = build_tree(
    xs, 
    ll=SVector{2}(-pi/n1d, -pi/n1d), width=SVector{2}(2pi, 2pi),
    max_levels=max_levels, min_pts=-1
    )
fr_tree = build_tree(ws, max_levels=max_levels, min_pts=-1)

##

gr(size=(500, 500))

sp_tree = get_root(sp_tree)
fr_tree = get_root(fr_tree)

pl = plot_quadtree(sp_tree, c=:black, label="")
scatter!(pl, xs[1,:], xs[2,:], marker=(2, :red), markerstrokewidth=0.1, label="")
display(pl)

pl = plot_binarytree(fr_tree, c=:black, label="")
scatter!(pl, ws[1,:], zeros(length(ws)), marker=(2, :red), markerstrokewidth=0.1, label="")
display(pl)

##

tol = 1e-3

sp_tree = get_root(sp_tree)
fr_tree = get_root(fr_tree)
while !isleaf(fr_tree)
    fr_tree = fr_tree.children[2]
end

for _=1:max_levels
    B = exp.(im*ks[:, fr_tree.data.inds]'*sp_tree.data.pts)
    # if prod(size(B)) < 1e6
    #     pl = heatmap(real.(B), yflip=true, c=:lightrainbow)
    #     display(pl)
    # end
    r = rank(B, tol)
    @printf(
        "\nx width %.2e, ω width %.2e\nblock of size %i x %i\nrank %i\n",
        sp_tree.boundary.widths[1], 
        fr_tree.boundary.widths[1],
        size(sp_tree.data.pts, 2), size(fr_tree.data.pts, 2),
        r
        )

    sp_tree = sp_tree.children[1]
    fr_tree = fr_tree.parent
end