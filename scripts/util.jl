using LinearAlgebra, LowRankApprox, FINUFFT, Printf

R(t) = [cos(t) -sin(t); sin(t) cos(t)]

function block_rank(;a=nothing, b=nothing,
    s=nothing, t=nothing, u=nothing, v=nothing, 
    ws=nothing, xs=nothing, use_eigvals=false,
    max_points=2^12, tol=1e-15, verbose=false, os=2
    )
    if all(isnothing.([a, b, ws]))
        error("You must specify ws _or_ both a and b.") 
    end
    if all(isnothing.([s, t, u, v, xs]))
        error("You must specify xs _or_ all of s, t, u, and v.") 
    end

    # set regions from ws and xs
    if all(isnothing.([a, b]))
        a, b = extrema(norm.(eachcol(ws))).^2
    end
    if all(isnothing.([s, t, u, v]))
        s, t = extrema(xs[1,:])
        u, v = extrema(xs[2,:])
    end

    # compute number of points by Nyquist with oversampling
    nx = 2*(t-s)*sqrt(b)/(2pi) + 1
    ny = 2*(v-u)*sqrt(b)/(2pi) + 1
    nr = 2*max(t-s, v-u)*(sqrt(b) - sqrt(a))/(2pi) + 1
    nt = 2*max(t-s, v-u)*2pi*sqrt(b)/(2pi) + 1
    nx, ny, nr, nt = max.(2, ceil.(Int64, os * [nx, ny, nr, nt]))

    # set ws and xs if not given
    if use_eigvals
        ws = hcat(
                [[k1,k2] for k1=0:floor(Int64, sqrt(b)) 
                        for k2=(a > k1^2 ? ceil(Int64, sqrt(a-k1^2)) : 1):floor(Int64, sqrt(b-k1^2))
                ]...
            )
        if length(ws) > 0
            ws = Float64.([ws [0 -1; 1 0]*ws [-1 0; 0 -1]*ws [0 1; -1 0]*ws])
        else
            @warn(@sprintf(
                "no eigenvalues in annulus a : %.2e, b : %.2e", a, b
                ))
            return 0
        end
    elseif isnothing(ws)
        ts = range(0, 2pi, nt+1)[1:end-1]
        ws = ([
            cos.(repeat(ts, nr)) sin.(repeat(ts, nr))
            ] .* repeat(range(sqrt(a), sqrt(b), nr), inner=nt))'
    end
    if isnothing(xs)
        xs = [repeat(range(s, t, nx), ny) repeat(range(u, v, ny), inner=nx)]'
    end

    # compute size of block before subsampling
    sM = (size(xs,2), size(ws,2))

    # subsample if too many points or frequencies are given
    if size(xs, 2) > max_points
        @warn(@sprintf(
                "there are %i > %i = max_points xs in box -- subsampling to reduce computation.", 
                size(xs, 2), max_points
                ))
        xs = xs[:, round.(Int64, range(1, size(xs, 2), max_points))]
    end
    if size(ws, 2) > max_points
        @warn(@sprintf(
                "there are %i > %i = max_points frequencies in annulus -- subsampling to reduce computation.", 
                size(ws, 2), max_points
                ))
        ws = ws[:, round.(Int64, range(1, size(ws, 2), max_points))]
    end

    # compute rank densely (if matrix is small) or implicitly
    M = Matrix{ComplexF64}(undef, 0, 0)
    if nx*ny*nr*nt < 1000^2
        M   = exp.(im*(xs' * ws))
        rnk = rank(M, rtol=tol)
    else
        fwd_plan = finufft_makeplan(
            3, 2, +1, 1, max(1e-15, 1e-2*tol)
            )
        finufft_setpts!(
            fwd_plan,
            ws[1,:], ws[2,:], Vector{Float64}([]), 
            xs[1,:], xs[2,:], Vector{Float64}([])
            )

        adj_plan = finufft_makeplan(
            3, 2, -1, 1, max(1e-15, 1e-2*tol)
            )
        finufft_setpts!(
            adj_plan,
            xs[1,:], xs[2,:], Vector{Float64}([]), 
            ws[1,:], ws[2,:], Vector{Float64}([])
            )
        
        function apply!(fs, cs, plan) 
            tmpc = Vector{ComplexF64}(undef, size(cs,1))
            tmpf = Vector{ComplexF64}(undef, size(fs,1))
            for j=1:size(cs,2)
                tmpc .= cs[:,j]
                finufft_exec!(plan, tmpc, tmpf)
                fs[:,j] .= tmpf
            end
            return fs
        end
        ml!(fs,  _, cs) = apply!(fs, cs, fwd_plan) 
        mlc!(fs, _, cs) = apply!(fs, cs, adj_plan) 
        
        M = LowRankApprox.LinOp{ComplexF64}(
            size(xs,2), size(ws,2), ml!, mlc!, nothing
            )

        rnk = length(psvdvals(M, rtol=tol))
    end

    verbose && @printf(
        "a : %.2e, b : %.2e, s : %.2e, t : %.2e, u : %.2e, v : %.2e\nnx : %i, ny : %i, nr : %i, nt : %i, size(M) : (%i,%i), rank %i, compression : %.2f\n", a, b, s, t, u, v, nx, ny, nr, nt, sM..., rnk, 
        prod(sM) / (rnk*sum(sM))
        )

    return rnk, sM...
end

function plot_quadtree!(pl, tree; kwargs...)
    ll, wd = tree.boundary.ll, tree.boundary.widths

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
    ll, wd = tree.boundary.ll[1], tree.boundary.widths[1]

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
