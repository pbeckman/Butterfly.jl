using SpecialFunctions, ApproxFun, Plots, Printf, LinearAlgebra

e = exp(1)

function ckl(k, l, a, b, r; terms=100)
    return 2 * (l==0 ? 0.5 : 1.0) * sum(
        p -> big(
            (p==0 ? 0.5 : 1.0) *
            besselj((p+l)/2, r*(b-a)/4) * 
            besselj((p-l)/2, r*(b-a)/4) *
            (besselj(k-p, r*(b+a)/2) + (-1)^p * besselj(k+p, r*(b+a)/2))
            ),
        (0:2:(2*terms)) .+ isodd(l)
        )
end

function ckl_bound(k, l, a, b, r; terms=100, c=e)
    # return (2c + log(1024) + 2log(c) - 2log(8c + a*r - b*r) - 2log(tol)) / (log(8) - log(-a + b) + log(c) - log(r))

    return 2 * sum(
        p -> begin big(
            # abs(besselj((p+l)/2, r*(b-a)/4)) # * 
            # abs(besselj((p-l)/2, r*(b-a)/4)) *
            # (
            #     abs(besselj(k-p, r*(b+a)/2)) +
            #     abs(besselj(k+p, r*(b+a)/2))
            #     ) # *
            exp(c + (p+l)/2 * log(r*(b-a)/(8c))) * 2
            # ((p-l)/2 < (b-a)/4 ? 1.0 : exp(c*(b-a)/8 - (p-l)/2*log(c))) * 
            # (
            #     # (abs(k-p) < c*r*(b+a)/4 ? 1.0 : exp(c*(b+a)/4 - abs(k-p)*log(c))) +
            #     1 + 
            #     # (k+p < (b+a)/2 ? 1.0 : exp(c*(b+a)/4 - (k+p)*log(c)))
            #     abs(besselj(k+p, (b+a)/2))
            # )
            # exp(e*(b-a)*r/8 - (p+l)/2)
            # (
            #     1 +
            #     # (abs(k-p) < c*(b+a)*r/4 ? 1.0 : exp(c*(b+a)*r/4 - abs(k-p)*log(c))) +
            #     (k+p < c*(b+a)*r/4 ? 1.0 : exp(c*(b+a)*r/4 - (k+p)*log(c)))
            # )

            # abs(besselj((p+l)/2, r*(b-a)/4)) 
            # * 2abs(besselj(k+p, r*(b+a)/2))
            ) end,
        (0:2:(2*terms)) .+ isodd(l)
        )
    # r = big(r)

    # return 7exp(-l/2)*(exp(e*(b-a)*r/8) + 1)

    # return 4exp(-l/2 - 3e*(b+a)*r/4) * (
    #     exp(e*a*r/8 + 3) / (e^3 - 1)
    #     + exp(e*(5b+3a)*r/8) * (exp(e*(b+a)*r/4 + 1) - 1) / (e - 1)
    #     )

    # Q = e*(b+a)*r/4
    
    # (1 / (exp(3) - 1)) * 2 * exp(0.125 * (-20k - 12Q + (-a + b) * ℯ * r)) * (
    # exp(3 + 0.25 * (a + b) * ℯ * r) * (1 + exp(2k)) + 
    # exp(Q + 0.25 * (a + b) * ℯ * r) * (
    #     exp(1 + 2k) + exp(2 + 2k) + exp(3 + 2k) - 
    #     exp(2Q) - exp(0.5 * (3k + Q)) - exp(0.5 * (2 + 3k + Q)) - 
    #     exp(0.5 * (4 + 3k + Q)) + exp(0.5 * (6 + 3k + Q))
    # ) + 
    # (exp(1 + Q) - 1) * (
    #     exp(2k + Q) + exp(1 + 2k + Q) + exp(2 + 2k + Q) + 
    #     exp(0.25 * (a + b) * ℯ * r) + exp(1 + Q + 0.25 * (a + b) * ℯ * r) + 
    #     exp(2 + 2Q + 0.25 * (a + b) * ℯ * r))
    # )

    # (1 / (exp(3) - 1)) * 2 * exp(0.125 * (-20k - 12Q + (-a + b) * ℯ * r)) * (
    # exp(3 + 0.25 * (a + b) * ℯ * r) * 2exp(2k) + 
    # exp(Q + 0.25 * (a + b) * ℯ * r) * (
    #     31exp(2k) + exp(0.5 * (6 + 3k + Q))
    # ) + 
    # (exp(1 + Q)) * (
    #     12exp(2k + Q) + 3exp(2 + 2Q + 0.25 * (a + b) * ℯ * r))
    # )

    # 4/(exp(3)-1) * exp(e*(3b+a)*r/8 - l/2 + 3)
    
end

function ckl_cheb(ks, max_l, a, b, r)
    C = zeros(ComplexF64, max_l+1, length(ks))
    for (ki, k) in enumerate(ks)
        Jk = Fun(
            x -> besselj(k, x), 
            Chebyshev(Interval(a*r, b*r))
            )
        nk = min(max_l+1, length(Jk.coefficients))
        C[1:nk, ki] .= Jk.coefficients[1:nk]
    end

    return C
end

##

tol = 1e-3

r     = sqrt(2)*pi / 1000
terms = 100

bma = 1000
a = 100000
b = a + bma

# lb = 2e*(b-a)*sqrt(r)/4 # + 6
# lb = e*(b-a)*r/4 # 6 + log(tol^(-2))
lb = 2 - 2log(8 - (b-a)*r) + log(1024) + log(tol^(-2)) / log(8/((b-a)*r))
kb = e*b*r/2 + log(tol^(-1))

skk, skl = ceil(Int64, (kb + 5)/200), ceil(Int64, (2lb + 5)/200)
ks = 0:skk:(kb + 5)
ls = 0:skl:(2lb + 5)

C_exp = ckl.(ks', ls, a, b, r, terms=terms)
C_bnd = ckl_bound.(ks', ls, a, b, r, terms=terms, c=10)

pl = contourf(
    ks, ls, 
    log10.(abs.(C_exp)), 
    # clims=(-16, 0), levels=15, 
    clims=(log10(tol)-1, 0), levels=Int(-2log10(tol)+1), 
    linewidth=0,
    # real.(C_exp),
    c=:lightrainbow, xlabel="k", ylabel="l", title="expansion", yflip=true
    )
contour!(pl,
    ks, ls, 
    log10.(abs.(C_bnd)),
    levels=[log10(tol)], c=:black, line=2
)
plot!(
    [0, kb], [lb, lb], line=(2,:grey,:dashdot), label="", 
    # xlims=[0, 1.2kb], ylims=[0, 1.2lb]
    )
plot!([kb, kb], [0, lb], line=(2,:grey,:dashdot), label="")
display(pl)

##

# C = 1
# a = 800
# b = sqrt(C + a^2)

# a, b = 60, 64

##

L   = 9
m   = 4^L
lvl = 3 # ≤ L
tol = 1e-3

r = sqrt(2)*pi / 2.0^(L - lvl) * 4

rhos = range(0, m/pi, 4^lvl)
nt   = 100
inds = round.(Int64, exp.(range(3, log(length(rhos)), nt)))
bs   = sqrt.(rhos[inds])
as   = sqrt.(rhos[inds .- 1])

cs = ones(nt)
# cs = 100sqrt.((bs - as)*r)
# cs = 100((bs - as)*r).^(3/4)

lbs = 2 * (cs .+ log(4) .+ log(tol^(-1))) ./ (log.(8*cs ./ ((bs - as)*r)))
kbs = e * bs * r/2 .+ log(tol^(-1))

pl = plot(inds, kbs .* lbs, line=2, label="bound", scale=:log10, marker=1)
plot!(pl, inds, 300*inds.^(1/4), line=2, label="O(k^{1/4})")
display(pl)

##

L  = 7.5
m  = Int64(4^L)

ws1d = -1.5div(sqrt(m),2):1.5(div(sqrt(m),2)-1)
lams = sort(vec(norm.(collect.(Iterators.product(ws1d, ws1d))).^2))[1:m]

lvl = 7
tol = 1e-3

r     = sqrt(2)*pi / 2.0^(L - lvl) * 4
terms = 20

kis  = round.(Int64, 4 .^ range(0, lvl, min(30, 4^lvl)))
rnks = zeros(Int64, length(kis))
bnds = zeros(Int64, length(kis))
for (kii, ki) in enumerate(kis)
    a = sqrt((ki-1) * lams[end] / 4^lvl)
    b = sqrt(ki     * lams[end] / 4^lvl)

    c = 10 ^ (1 / ki)
    # c = sqrt((b-a)*r)
    # c = ((b-a)*r)^(1/2)
    # c = b*r
    # c = ((b-a)*r).^(1/4)

    # c1 = log(1024) - 2log(tol)
    # c2 = log(8) - log((b-a)*r)
    # c  = (1 + sqrt(1 + c1/2*(1 + c2))) / (1 + c2)

    # lb = e*(b-a)*r/4 + 6 + log(tol^(-2))
    kb = e*b*r/2 + log(tol^(-1))
    if c > (b-a)*r/8
        # lb = (2c + 2log(c) - 2log(8c - (b-a)*r) + log(1024) + log(tol^(-2))) / (log(c) - log(b-a) - log(r) + log(8))

        # lb = 2 * (c - log(1 - (b-a)*r/(8c)) + log(4) + log(tol^(-1))) / (log(8c/((b-a)*r)))

        lb = 2 * (c + log(4) + log(tol^(-1))) / (log(8c/((b-a)*r)))

        # lb = (2c + 2log(c) + log(1024) + log(tol^(-2))) / (log(c) - log(b-a) - log(r) + log(8))

        # lb = (2c + 2log(c) + log(1024) + log(tol^(-2))) / (1 - 1/c - log(b-a) - log(r) + log(8))

        # lb = (9 + log(tol^(-2))) / (log(8) - log((b-a)*r))

        # lb = (9 + log(tol^(-2))) * sqrt((b-a)*r)

        # lb = 2*(sqrt((b-a)*r) + log(4) + log(tol)) / log(sqrt((b-a)*r)/8)
    else
        println("using small c bound!")
        lb = e*(b-a)*r/4 + 6 + log(tol^(-2))
    end

    skk, skl = ceil(Int64, 1.2kb/200), ceil(Int64, 1.2lb/200)
    ks = 0:skk:1.2kb
    ls = 0:skl:1.2lb

    C_exp  = ckl.(ks', ls, a, b, r, terms=terms)
    C_bnd  = ckl_bound.(ks', ls, a, b, r, terms=terms, c=c)

    rnks[kii] = 2 * sum(abs.(C_exp) .> tol) * skk * skl
    bnds[kii] = round(Int64, 2 * lb * kb)
    # bnds[kii] = ceil(Int64, lb * kb)
    neig      = sum((sqrt.(lams) .>= a) .&& (sqrt.(lams) .<= b))

    @printf(
        "(%i/%i) bR = %.1e, (b-a)R = %.1e, eigenvalues = %i, ε-rank for ε = %.0e is %i (≤ %i)\n", 
        kii, length(kis), b*r, (b-a)*r, neig, tol, 
        rnks[kii], bnds[kii]
        )

    # # pl = contourf(
    # #     ks, ls, 
    # #     log10.(abs.(C_exp)), 
    # #     # clims=(-16, 0), levels=15, 
    # #     clims=(-4, 0), levels=7, 
    # #     linewidth=0,
    # #     # real.(C_exp),
    # #     c=:lightrainbow, xlabel="k", ylabel="l", title="expansion", yflip=true
    # #     )
    # C_exp[C_exp .< tol] .= NaN
    # pl = heatmap(
    #     ks .+ 0.5, ls .+ 0.5, log10.(abs.(C_exp)),
    #     c=:lightrainbow, clims=(log10(tol), 0),
    #     xlabel="k", ylabel="l", title="expansion"
    # )
    # # contour!(pl,
    # #     ks, ls, 
    # #     log10.(abs.(C_bnd)),
    # #     levels=[log10(tol)], c=:black, line=(2, :dash)
    # # )
    # plot!([0, kb], lb*ones(2), line=(2,:black,:dash), label="")
    # plot!(kb*ones(2), [0, lb], line=(2,:black,:dash), label="")
    # # plot!(b*r*ones(2), [0, ls[end]], line=(2,:grey,:dashdot), label="")
    
    # # g(c) = -2*(log(32c) + c - log(8c - (b-a)*r) - log(tol))/(log(b-a) + log(r) - log(8c))

    # # g(c) = (2c + log(1024) + 2log(c) - 2log(8c + a*r - b*r) - 2log(tol)) / (log(8) - log(-a + b) + log(c) - log(r))

    # # g2(c) = (2c + log(1024) - 2log(tol)) / (log(c) + log(8) - log((b-a)*r))

    # # g3(c) = (2c + log(1024) - 2log(tol)) / (1 - 1/c + log(8) - log((b-a)*r))

    # # cs = range((b-a)*r/8, 100, 1000)[2:end]
    # # gs = g.(cs)
    # # pl = plot(
    # #     cs, gs, label="true", line=(3, :black),
    # #     yscale=:log10, xscale=:log10, 
    # #     ylims=collect(extrema(gs))
    # #     )
    # # plot!(pl,
    # #     cs, g2.(cs), c=:red,
    # #     line=(3, :dot), label="remove log"
    # # )
    # # plot!(pl,
    # #     cs, g3.(cs), c=:blue,
    # #     line=(3, :dot), label="logx ≥ 1 - 1/x"
    # # )
    # # plot!(pl,
    # #     cs[findmin(gs)[2]]*ones(2), 
    # #     collect(extrema(gs)), 
    # #     line=(:black, 2, :dash), label=""
    # # )
    # # plot!(pl,
    # #     c*ones(2), 
    # #     collect(extrema(gs)), 
    # #     line=(:blue, 2, :dash), label=""
    # # )
    # # plot!(pl,
    # #     1*ones(2), 
    # #     collect(extrema(gs)), 
    # #     line=(:green, 2, :dash), label="[(b-a)r]^{1/4}"
    # # )

    # display(pl)
end

##

pl = plot(
    kis, rnks, label="continuous rank", marker=3, line=2,
    scale=:log10, legend=:bottomright
    )
# plot!(
#     1:64, size.(B.Vt[3][1,:], 1), c=palette(:default)[3], marker=2, label="ℓ=3", xlabel="k", ylabel="block rank"
#     )

plot!(kis, bnds, label="bound", marker=3, line=2)
# plot!(kis, 20kis.^(1/2), label="O(√n)")
display(pl)

##

pl = contourf(
    ks, ls, 
    log10.(abs.(C_exp)), 
    # clims=(-16, 0), levels=15, 
    clims=(-4, 0), levels=7, 
    linewidth=0,
    # real.(C_exp),
    c=:lightrainbow, xlabel="k", ylabel="l", title="expansion", yflip=true
    )
# contour!(pl,
#     ks, ls, 
#     log10.(ckl_bound.(abs.(ks)', abs.(ls), a, b, r, terms=terms, c=e)),
#     levels=[log10(tol)], c=:black, line=2
# )
plot!([0, kb], [lb, lb], line=(2,:grey,:dashdot), label="", xlims=[0, 1.2kb], ylims=[0, 1.2lb])
plot!([kb, kb], [0, lb], line=(2,:grey,:dashdot), label="")
display(pl)

# pl = contourf(
#     ks, ls, 
#     log10.(abs.(C_cheb)), 
#     clims=(-15, 0), levels=8, linewidth=0,
#     # real.(C_cheb),
#     c=:lightrainbow, xlabel="k", ylabel="l", title="chebyshev", yflip=true
#     )
# display(pl)


##

nu = 10000
xs = range(nu^0.99, nu^1.01, 10000)
plot(xs, besselj.(nu, xs), label="", line=2)
# plot(xs, log10.(abs.(besselj.(nu, xs))), label="", line=2)