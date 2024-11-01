module EntropyGrad begin

    using AtomsBase, Unitful, ExtXYZ, AtomsCalculators, AtomsBuilder,
    EmpiricalPotentials, StaticArrays, Test, JSON, ForwardDiff
    using AtomsCalculators.AtomsCalculatorsTesting
    using LinearAlgebra: dot, norm, I, diagm, svd
    using EmpiricalPotentials: cutoff_radius, StillingerWeber
    using AtomsCalculators: potential_energy, forces
    using NeighbourLists, Zygote, ChainRulesCore
    using ChainRulesTestUtils, Test
    using LinearAlgebra, Random, Distributions, Zygote
    using ComplexElliptic, ForwardDiff, BenchmarkTools, Plots
    using LaTeXStrings, SparseArrays, SparseDiffTools, Elliptic
    using Elliptic.Jacobi: sn, cn, dn
    import ComplexElliptic.ellipkkp

    EP = EmpiricalPotentials
    ACT = AtomsCalculators.AtomsCalculatorsTesting

    sw = StillingerWeber()

    export entropywithgradient_hess, entropywithgradient, entropyonlygradient, entropyonlyentropy

    # ellipjc is not currently reverse-AD friendly, this is a fix for that
    function Zygcompat_ellipjc(u, L; flag=false)
        if !flag
            _, Kp = ellipkkp(L)
            high = [(imag(x) > real(Kp) / 2) for x in u]
            if any(high)
                u = [(imag(x) > real(Kp) / 2) ? im * Kp .- x : x for x in u]
            end
            m = exp(-2 * pi * L)
        else
            high = falses(size(u))  
            m = L
        end
        if abs(m) < 6eps(Float64)
            sinu = [sin(x) for x in u]
            cosu = [cos(x) for x in u]
            sn = sinu + m / 4 * (sinu .* cosu - u) .* cosu
            cn = cosu - m / 4 * (sinu .* cosu - u) .* sinu
            dn = 1 .+ m / 4 .* (cosu .^ 2 - sinu .^ 2 .- 1)
        else
            if abs(m) > 1e-3
                kappa = (1 - sqrt(Complex(1 - m))) / (1 + sqrt(Complex(1 - m)))
            else
                kappa = ComplexElliptic.polyval(Complex{Float64}.([132.0, 42.0, 14.0, 5.0, 2.0, 1.0, 0.0]), Complex{Float64}(m / 4.0))
            end
            sn1, cn1, dn1 = Zygcompat_ellipjc(u / (1 + kappa), kappa ^ 2, flag=true)

            denom = 1 .+ kappa .* sn1 .^ 2
            sn = (1 .+ kappa) .* sn1 ./ denom
            cn = cn1 .* dn1 ./ denom
            dn = (1 .- kappa .* sn1 .^ 2) ./ denom
        end

        if any(high)
            snh = Zygote.Buffer(sn)
            cnh = Zygote.Buffer(cn)
            dnh = Zygote.Buffer(dn)
            snh[:] = sn[:]
            cnh[:] = cn[:]
            dnh[:] = dn[:]
            snh[high] = -1 ./ (sqrt(m) * sn[high])
            cnh[high] = im * dn[high] ./ (sqrt(m) * sn[high])
            dnh[high] = im * cn[high] ./ sn[high]
        end

        return copy(sn), copy(cn), copy(dn)
    end
    @inline Rs2x = Rs -> reinterpret(eltype(eltype(Rs)), Rs)
    @inline function x2Rs(x)
        return reinterpret(SVector{3, eltype(x)}, x)
    end
    @inline function ∇Ei(sw::StillingerWeber, x::AbstractVector{T}, Zs, z0) where T
        V = reduce(vcat,(EmpiricalPotentials.eval_grad_site(sw, x2Rs(x), Zs, z0)[2]))
        return reshape(V,length(V))::Vector{T}
    end
    @inline function eyelike(j,N,T)
        v = zeros(T,N)
        v[j] = one(T)
        return v
    end
    function ChainRulesCore.rrule(::typeof(∇Ei), sw, x, Zs, z0)
        y = ∇Ei(sw, x, Zs, z0)
        function eval_grad_site_pullback(Δy)
            if Δy isa Tangent
                # use via ChainRules supplies Tangent types which contain a Tuple, so convert that to Vector
                return (NoTangent(), NoTangent(), ForwardDiff.jacobian(x -> ∇Ei(sw, x, Zs, z0), Rs2x(x))'*collect(Δy.backing), NoTangent(), NoTangent())
            else
                return return (NoTangent(), NoTangent(), ForwardDiff.jacobian(x -> ∇Ei(sw, x, Zs, z0), Rs2x(x))'*Δy, NoTangent(), NoTangent()) # Zygote supplies Arrays
            end
        end
        return y, eval_grad_site_pullback
    end

    adHzygote(sw, Rs, Zs, z0) = Zygote.jacobian(∇Ei, sw, Rs, Zs, z0)[2]  ## site hessian itself

    function adHzygotepushforwardhelper(ℓ, sw, x, Zs, z0)
        φ(t1,t2,U,V) = ∇Ei(sw, x + t1 * U + t2 * V, Zs, z0)
        ∂₁φ(t1,t2,U,V) = ForwardDiff.derivative(t -> φ(t,t2,U,V), t1)
        ∂₂∂₁φ(t1,t2,U,V) = ForwardDiff.derivative(t -> ∂₁φ(t1,t,U,V), t2)
        J = reduce(hcat,([∂₂∂₁φ(0.0,0.0,eyelike(i,ℓ,eltype(x)),eyelike(j,ℓ,eltype(x))) for j in 1:ℓ for i in 1:ℓ]))
        return reshape(J,size(J,2),size(J,1))
    end
    function ChainRulesCore.rrule(::typeof(adHzygote), sw, x, Zs, z0)
        y = adHzygote(sw, Rs2x(x), Zs, z0)
        function adHzygote_pullback(Δy)
            return (NoTangent(), NoTangent(), adHzygotepushforwardhelper(size(Δy,1), sw, x, Zs, z0)'*reshape(Δy,length(Δy)), NoTangent(), NoTangent())
        end
        return y, adHzygote_pullback
    end

    function relativepos_reverse_ad_2(XX, nlist, idx)
        ii = nlist.i
        jj = nlist.j
        S = nlist.S 
        C = nlist.C'
        X = [XX[3i-2:3i] for i in 1:(length(XX) ÷ 3)]
        M2 = [X[jj[i]] - X[ii[i]] + (C * S[i]) for i in idx]
        VV = reduce(vcat,M2)
        return reshape(VV,length(VV))
    end

    function comp_hess(Js, info, X, Zs, z0, nlist, indexcheckpoints)
        Rs = [relativepos_reverse_ad_2(X, nlist, indexcheckpoints[kk]+1:indexcheckpoints[kk+1]) for kk in 1:length(indexcheckpoints)-1]
        Nat::Integer = length(info)
        Hess = Zygote.Buffer(X,3*Nat,3*Nat)
        Hess[1:end,1:end] = zeros(3*Nat, 3*Nat)
        D = 3
        DD = 1:D
        @inbounds for i = 1:Nat 
            js = Js[i]
            rs = Rs[i]
            zs = Zs[i]
            z_0 = z0[i]
            # Hardcoded for StillingerWeber
            Hi = adHzygote(StillingerWeber(), rs, zs, z_0)
            Ji = (i-1)*D .+ DD
            @inbounds for α1 in eachindex(js)
                @inbounds for α2 in eachindex(js)
                    A1 = (α1-1) * D .+ DD
                    A2 = (α2-1) * D .+ DD
                    J1 = (js[α1]-1) * D .+ DD
                    J2 = (js[α2]-1) * D .+ DD
                    Hess[J1, J2] += view(Hi,A1, A2)
                    Hess[J1, Ji] -= view(Hi,A1, A2)
                    Hess[Ji, J2] -= view(Hi,A1, A2)
                    Hess[Ji, Ji] += view(Hi,A1, A2)
                end
            end
        end
        return copy(Hess)
    end


    function conformal_map(A,ℓ)
        e = eigvals(A)
        m = max(minimum(e),1.0) 
        M = maximum(e)
        k = ((M/m)^(1/4) - 1) / ((M/m)^(1/4) + 1)
        L = -log(k) / π
        K, Kp = ellipkkp(L)
        N = ℓ
        u, cn, dn = Zygcompat_ellipjc(0.5im * Kp .- K .+ (0.5:N) .* (2 * K / N), L)
        dzdt = cn .* dn ./ (1/k .- u).^2
        z = (m * M)^(1/4) .* ((1/k .+ u) ./ (1/k .- u))
        return m, M, k, K, z, log.(z.^2) .* dzdt ./ z
    end

    function compute_S(Js, info, X, Zs, z0, nlist, indexcheckpoints, ℓ)
        H_u = comp_hess(Js, info, X, Zs, z0, nlist, indexcheckpoints)
        m, M, k, K, z, w_z = conformal_map(Symmetric(H_u), ℓ)
        S = 0.0
        @inbounds for (zi, wi) in zip(z, w_z)
            Rz = lu(zi^2 * I - H_u)
            S += wi * tr(H_u*inv(Rz))
        end
        return -8 * K *  (m * M)^(1/4) * imag(S) / (k * π * ℓ)
    end

    function entropy_onlyentropy(sys, x, ℓ)
        nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber())) 
        info = [EmpiricalPotentials.get_neighbours(sys, sw, nlist, i) for i in 1:length(sys)]
        Js_ = [element[1] for element in info]
        Rs_ = [element[2] for element in info]
        Zs_ = [element[3] for element in info]
        z0_ = [element[4] for element in info]
    
        indexcheckpoints = vcat(0,cumsum(length.(Rs_)))
        return compute_S(Js_, info, x, Zs_, z0_, nlist, indexcheckpoints, ℓ)
    end

    function entropy_onlygradient(sys, x, ℓ)
        nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber())) 
        info = [EmpiricalPotentials.get_neighbours(sys, sw, nlist, i) for i in 1:length(sys)]
        Js_ = [element[1] for element in info]
        Rs_ = [element[2] for element in info]
        Zs_ = [element[3] for element in info]
        z0_ = [element[4] for element in info]
    
        indexcheckpoints = vcat(0,cumsum(length.(Rs_)))
        S(x) = compute_S(Js_, info, x, Zs_, z0_, nlist, indexcheckpoints,ℓ)
        return Zygote.gradient(S,x)
    end
    
    function entropy_withgradient(sys, x, ℓ)
        nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber())) 
        info = [EmpiricalPotentials.get_neighbours(sys, sw, nlist, i) for i in 1:length(sys)]
        Js_ = [element[1] for element in info]
        Rs_ = [element[2] for element in info]
        Zs_ = [element[3] for element in info]
        z0_ = [element[4] for element in info]
    
        indexcheckpoints = vcat(0,cumsum(length.(Rs_)))
        S(x) = compute_S(Js_, info, x, Zs_, z0_, nlist, indexcheckpoints, ℓ)
        return Zygote.withgradient(S,x)
    end

    function entropy_withgradientandhess(sys, x, ℓ)
        nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber())) 
        info = [EmpiricalPotentials.get_neighbours(sys, sw, nlist, i) for i in 1:length(sys)]
        Js_ = [element[1] for element in info]
        Rs_ = [element[2] for element in info]
        Zs_ = [element[3] for element in info]
        z0_ = [element[4] for element in info]
        indexcheckpoints = vcat(0,cumsum(length.(Rs_)))
        S(x) = compute_S(Js_, info, x, Zs_, z0_, nlist, indexcheckpoints, ℓ)
        H = comp_hess(Js_, info, x, Zs_, z0_, nlist, indexcheckpoints)
        return Zygote.withgradient(S,x),  H
    end

end