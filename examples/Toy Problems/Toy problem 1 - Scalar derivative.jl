using LinearAlgebra, Zygote, BenchmarkTools
using ComplexElliptic, ForwardDiff, Elliptic
using Elliptic.Jacobi: sn, cn, dn
import ComplexElliptic.ellipjc
import ComplexElliptic.ellipkkp
using Plots, LaTeXStrings

# This generates an SPD example matrix from an input vector u
function example_A(u::AbstractVector{T}) where T
    n = length(u)
    B = [ (i >= max(1, j-3) && i <= min(n, j+3)) ? 
          (sin(u[i] + u[j]) + cos(u[i] * u[j]) + exp(u[i] - u[j])) : 
          zero(T) for i in 1:n, j in 1:n]
    A = B' * B  # This ensures the generated example A is SPD
    return A
end

# The below conformal map quadrature is hard-coded for g(f(A)) where f(A)=A^(1/3) and g(X)=sum(X.^3)
function conformal_map_data(A, N)
    e = eigvals(A)
    e[abs.(e) .< 1e-10] .= 1 
    m, M = minimum(e), maximum(e)
    k = ((M/m)^(1/4) - 1) / ((M/m)^(1/4) + 1)
    L = -log(k) / π
    K, Kp = ellipkkp(L)
    t = 0.5im * Kp .- K .+ (0.5:N) .* (2 * K / N)
    u, cn, dn = ellipjc(t, L)
    z = Complex(m * M)^(1/4) .* ((1/k .+ u) ./ (1/k .- u))
    dzdt = cn .* dn ./ (1/k .- u).^2
    w_z = (z.^2).^(1/3) .* dzdt .* z
    return m, M, k, K, z, w_z
end
function conformal_map_fun(u, m, M, k, K, z, w_z, N)
    A = example_A(u)
    S = zeros(length(u),length(u))
    for (zi, wi) in zip(z, w_z)
        Rz = zi^2 * I - A
        LU = lu(Rz)
        S += wi * inv(LU)
    end
    S_fin = -8 * K * (m * M)^(1/4) * imag(S) / (k * π * N)
    return sum(S_fin.^(3))
end

# ------------------------------------------------------------------
# A simple finite difference implemenation for sanity checks
# ------------------------------------------------------------------
function fd_jacobian(f::Function, u::AbstractVector, deltax::Float64)
    A0 = f(u)  
    p = length(u)
    jacobian_tensor = zeros(p)
    
    for j in 1:p
        # Perturb u in the j-th direction
        u_plus = copy(u)
        u_minus = copy(u)
        u_plus[j] += deltax
        u_minus[j] -= deltax
        
        A_plus = sum((f(u_plus)^(1/3)).^3)
        A_minus = sum((f(u_minus)^(1/3)).^3)
        
        jacobian_tensor[j] = (A_plus - A_minus) / (2 * deltax)
    end
    
    return jacobian_tensor
end


# ------------------------------------------------------------------
# Reverse mode
# ------------------------------------------------------------------

function grad_S_fin(u, m, M, k, K, z, w_z,N)
    return Zygote.gradient(u -> conformal_map_fun(u, m, M, k, K, z, w_z, N), u)
end

# ------------------------------------------------------------------
# Forward mode
# ------------------------------------------------------------------

function grad_S_fw(u, m, M, k, K, z, w_z,N)
    return ForwardDiff.gradient(u -> conformal_map_fun(u, m, M, k, K, z, w_z,N), u)
end

# ------------------------------------------------------------------
# Test Accuracy
# ------------------------------------------------------------------

# Test setup
sz = 50
u = rand(sz)
A = example_A(u)
err10, err15, err20, err25 = [], [], [], []

for steps in [1.0,1e-1,1e-2,1e-3,1e-4]
    JFD = fd_jacobian(example_A, u, steps)
    m, M, k, L, K, Kp = conformal_map_data(A, 10)
    push!(err10, norm(JFD - grad_S_fin(u, m, M, k, L, K, Kp,10)[1])/norm(JFD))
    m, M, k, L, K, Kp = conformal_map_data(A,15)
    push!(err15, norm(JFD - grad_S_fin(u, m, M, k, L, K, Kp,15)[1])/norm(JFD))
    m, M, k, L, K, Kp = conformal_map_data(A,20)
    push!(err20, norm(JFD - grad_S_fin(u, m, M, k, L, K, Kp,20)[1])/norm(JFD))
    m, M, k, L, K, Kp = conformal_map_data(A,25)   
    push!(err25, norm(JFD - grad_S_fin(u, m, M, k, L, K, Kp,25)[1])/norm(JFD))
end


xticks = [1,1e-1,1e-2,1e-3,1e-4]
yticks = [10.0^i for i in -10:2:-2]

scatter(xticks, err10[1:end], xscale = :log10, yscale=:log10, color=1, ylabel="Errors", xlabel="Step Size", label=L"\ell=10", legend=:bottomright, marker=:o)
plot!(xticks, err10[1:end], xscale = :log10, yscale=:log10, label=false, color=1)
yticks!(yticks)
scatter!(xticks, err15[1:end], label=L"\ell=15",color=2, markersize=3)
plot!(xticks, err15[1:end], label=false, color=2)
scatter!(xticks, err20[1:end], label=L"\ell=20",color=3, markersize=3)
plot!(xticks, err20[1:end], label=false, color=3)
scatter!(xticks, err25[1:end], label=L"\ell=25", color=4, markersize=3)
plot!(xticks, err25[1:end], label=false, color=4)
plot!(xticks[3:end-1], 0.2*xticks[3:end-1].^2, label=L"\propto x^2", inestyle=:dash, color=5)

# ------------------------------------------------------------------
# Compare Modes & Plot
# ------------------------------------------------------------------

# Set the matrix sizes for which CPU timings will be checked
sizes = [10,20,50,70,100,120,150]

# Loop over matrix sizes to generate the CPU timings
times_rv = Vector{Float64}(undef, length(sizes))
times_fw = Vector{Float64}(undef, length(sizes))
for i in 1:length(sizes)
    u = rand(sizes[i])
    A = example_A(u)
    m, M, k, K, z, w_z = conformal_map_data(A,10)
    times_rv[i] = @belapsed grad_S_fin($u, $m, $M, $k, $K, $z, $w_z, 10)
    print("Reverse mode with size $(sizes[i]) took $(times_rv[i]) seconds\n") # Un/comment for verbose progress reports
    times_fw[i] = @belapsed grad_S_fw($u, $m, $M, $k, $K, $z, $w_z, 10)
    print("Forward mode with size $(sizes[i]) took $(times_fw[i]) seconds\n") # Un/comment for verbose progress reports
end

# Plot elapsed CPU time on a log-log plot to compare asymptotic rates (see paper for expectation)
scatter(sizes, times_rv, label="Reverse Mode", ylabel="CPU time [s]", xlabel="Matrix size", xscale=:log10, yscale=:log10, color=1, legend=:topleft)
plot!(sizes, times_rv, label=false, linestyle=:dash, color=1)
scatter!(sizes, times_fw, label="Forward Mode", color=2)
plot!(sizes, times_fw, label=false, linestyle=:dash, color=2)