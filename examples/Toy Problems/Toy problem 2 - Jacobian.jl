using LinearAlgebra, Zygote, Symbolics, SymbolicUtils, BenchmarkTools
using ComplexElliptic, ForwardDiff, Elliptic, SparseArrays, SparseDiffTools
using Elliptic.Jacobi: sn, cn, dn
import ComplexElliptic.ellipjc
import ComplexElliptic.ellipkkp
using Plots, LaTeXStrings

# Define a toy energy function, see the example in Section 4.3 of the companion paper
function energy(c, u, ε=1e-3)
    N = length(u)
    E = sum( c[i, i+1] * (u[i+1] - u[i])^2 for i = 1:N-1 )
    E += sum( (u[i+1] - u[i])^4 for i = 1:N-1 )
    E += 0.5* sum( (u[i+1] - u[i])^3 for i = 1:N-1 )
    E += c[N, 1] * (u[1] - u[N])^2 + (u[1] - u[N])^4
    E += ε * sum(u[i]^2 for i = 1:N)
    E += 0.5* (u[1]-u[N])^3
    return E
end

# Perturb matrix to introduce impurity
function createC(n)
    c_0 = Matrix{Float64}(I, n, n)
    for i in 1:(n-1)
        c_0[i+1, i] = 1
        c_0[i, i+1] = 1
    end
    c_0[1, n] = 1
    c_0[n, 1] = 1
    c1 = copy(c_0)
    c1[1, 2] += 1.0
    c1[2, 1] += 1.0
    return c1
end

# AD-based Hessians
H_zyg(c, u) = Zygote.hessian(u -> energy(c, u), u)
H_fwd(c, u) = ForwardDiff.hessian(u -> energy(c, u), u)

function dHdU(C,u) # with coloring
    n = length(u)
    global hcalls = 0
    function hess!(out, y)
        global hcalls += 1
        H = ForwardDiff.hessian(v -> energy(C, v), y)
        out .= vec(H)
        nothing
    end
    # Symbolics to detect the sparsity pattern and compute Jacobian using forwarddiff_color
    @variables x[1:n]
    x_vec = [x[i] for i in 1:n]
    energy_expr = energy(C, x_vec)
    H = Symbolics.hessian(energy_expr, x_vec)

    sparsity_pattern = Symbolics.jacobian_sparsity(vec(H), x_vec) ## Sparsity Pattern
    jac = Float64.(sparse(sparsity_pattern))
    colors = matrix_colors(jac)
    Jac_fin = forwarddiff_color_jacobian!(jac, hess!, u, colorvec = colors)
    return reshape(Jac_fin,n,n,n)
end

function dHdU_wo(C,u) # w/o coloring
    n = length(u)
    H(y) = ForwardDiff.hessian(v -> energy(C, v), y)
    Jac_fin = ForwardDiff.jacobian(y -> H(y), u)
    return reshape(Jac_fin,n,n,n)
end

function contour_diff(A, dhdu, N) # dhdu can be with coloring or without, i.e. dHdU_wo or dHdU 
    e = eigvals(A)
    m = minimum(e)
    M = maximum(e)
    k = ((M/m)^(1/4) - 1) / ((M/m)^(1/4) + 1)
    L = -log(k) / π
    K, Kp = ellipkkp(L)

    sn, cn, dn = ellipjc(0.5im * Kp .- K .+ (0.5:N) .* (2 * K / N), L)
    w =  (m * M)^(1/4) .* ((1/k .+ sn) ./ (1/k .- sn))
    dzdt = cn .* dn ./ (1/k .- sn).^2
    dS = zeros(ComplexF64, size(dhdu))

    for ℓ in 1:size(dS)[1]
        for j in 1:N
            dS[:,:,ℓ] += (sqrt(w[j]^2) * w[j]) * inv(w[j]^2 * I - A)* dhdu[:,:,ℓ] * inv(w[j]^2 * I - A) * dzdt[j]
        end
    end
    dS = -8 * K * (m * M)^(1/4) * imag(dS) / (k * π * N)
    return dS
end

# ------------------------------------------------------------------
# Forward mode
# ------------------------------------------------------------------

# Forward differentiation of S
function dHdu(c, u)
    n = length(u)
    ∂H∂u = ForwardDiff.jacobian(x -> vec(H_fwd(c, x)), u)
    ∂H∂u = reshape(∂H∂u, n, n, n)
    return ∂H∂u
end

function fwd_deriv_S(A, u, m, M, k, K, z, w_z)
    dAdu = dHdu(A, u)
    dS = zeros(length(u))
    N = 20
    for ℓ in 1:length(dAdu[1,1,:])
        S = 0.0
        for (zi, wi) in zip(z, w_z)
            Rz = pinv(zi^2 * I - A)
            S += wi * tr(Rz * dAdu[:, :, ℓ] * Rz)
        end
        dS[ℓ] = -8 * K * (m * M)^(1/4) * imag(S) / (k * π * N)
    end
    return dS
end

# ------------------------------------------------------------------
# Finite Difference comparisons
# ------------------------------------------------------------------

function fd_hessian_sqrt(C, u, h)
    n = length(u)   
    H_diff = zeros(n,n,n)  
    for i in 1:n
        y_plus = copy(u)
        y_minus = copy(u)
        y_plus[i] += h
        y_minus[i] -= h
        H_plus = sqrt(H_fwd(C, y_plus))
        H_minus = sqrt(H_fwd(C, y_minus))
        H_diff[:,:,i] = (H_plus - H_minus) / (2 * h)
    end
    return H_diff
end

# Compute errors compared to finite difference variants with different stepsizes
u = rand(10)
cc = 0.5*createC(10)
HH = H_fwd(cc, u)
err10,err15,err20,err25 = [], [], [], []
fdticks = [1, 0.1 , 0.01, 0.001, 0.0001, 1e-5]
for hh in fdticks
    X = fd_hessian_sqrt(cc, u, hh)
    dHdU_test = dHdU(cc,u)
    push!(err10,norm(contour_diff(HH,dHdU_test,10)-X))
    push!(err15,norm(contour_diff(HH,dHdU_test,15)-X))
    push!(err20,norm(contour_diff(HH,dHdU_test,20)-X))
    push!(err25,norm(contour_diff(HH,dHdU_test,25)-X))
end

# Plot errors compared to finite difference approach
ytick = [10.0^i for i in -10:2:1]
scatter(fdticks, err10[1:end], xscale = :log10, yscale=:log10, color=1, ylabel="Errors", xlabel="Step Size", label=L"\ell=10", legend=:bottomright, marker=:o, lw=3, markersize=3,)
plot!(fdticks, err10[1:end], xscale = :log10, yscale=:log10, label=false, color=1)
yticks!(ytick)
scatter!(fdticks, err15[1:end], label=L"\ell=15",color=2, markersize=3)
plot!(fdticks, err15[1:end],label=false, color=2)
scatter!(fdticks, err20[1:end], label=L"\ell=20",color=3, markersize=3)
plot!(fdticks, err20[1:end],label=false, color=3)
scatter!(fdticks, err25[1:end], label=L"\ell=25",color=4, markersize=3)
plot!(fdticks, err25[1:end],label=false, color=4)
plot!(fdticks[3:end-1], 0.2*fdticks[3:end-1].^2, label=L"\propto x^2",linestyle=:dash,color=5)


# ------------------------------------------------------------------
# CPU timings
# ------------------------------------------------------------------
# Different matrix sizes which will be tested
sizes = [200,250,300,350]
times_color = []
times_nocolor = []

# Loop to compute CPU timings.
for size in sizes
    u = rand(size)
    c = createC(size)
    HH = H_fwd(c, u)

    push!(times_color, @belapsed contour_diff($HH, dHdU($c,$u),15))
    push!(times_nocolor, @belapsed contour_diff($HH, dHdU_wo($c,$u),15))
end

# Plot elapsed CPU time on a log-log plot to compare asymptotic rates (see paper for expectation)
using Plots
scatter(sizes, times_color, label="with coloring", ylabel="CPU time [s]", xlabel="Matrix size", yscale=:log10, color=1, legend=:topleft)
plot!(sizes, times_color, label=false, linestyle=:dash, color=1)
scatter!(sizes, times_nocolor, label="without coloring", color=2)
plot!(sizes, times_nocolor, label=false, linestyle=:dash, color=2)