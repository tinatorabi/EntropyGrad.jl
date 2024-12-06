using AtomsBase, DecoratedParticles, AtomsBuilder, 
      GeomOpt, StaticArrays, Unitful, LinearAlgebra , 
      EmpiricalPotentials, SaddleSearch, JuLIP, Interpolations
using GeomOpt:DofManager, set_dofs!
using AtomsCalculators: virial, forces, potential_energy
import Random

function ortho_vec(u, scale)
    v1 = cross(u, [1, 0, 0])
    if norm(v1) ≈ 0
        v1 = cross(u, [0, 1, 0])
    end
    return normalize(v1) * scale
end

function adjust_positions(A, B)
    bond_vector = B - A
    midpoint = 0.5 * (A + B)
    bond_length = norm(bond_vector)
    orthogonal_vector = ortho_vec(bond_vector, bond_length)
    A_prime = midpoint - 0.5 * orthogonal_vector
    B_prime = midpoint + 0.5 * orthogonal_vector
    return A_prime, B_prime
end

# Setup
GO = GeomOpt      
DP = DecoratedParticles
Random.seed!(100)
AtomsBase.atomic_number(sys::SoaSystem, i::Integer) = sys.arrays.𝑍[i]
sw = EmpiricalPotentials.StillingerWeber() 
kb2ev = 8.617333262 * 1e-5 # conversion constant
ℓ = 35 # Set number of quadrature points to use
Nimg = 18

r0 = AtomsBuilder.Chemistry.rnn(:Si)
sys_0 = AosSystem( AtomsBuilder.bulk(:Si, cubic=true)*2)
nlist0 = PairList(sys_0, cutoff_radius(StillingerWeber()))
XX_0 = reduce(vcat, nlist0.X)
S_0 = kb2ev * entropy_onlyentropy(sys_0,XX_0,ℓ)

Htst = []
Vtst = []
ratio = []

T = 10 .* [10,30,50,70]
for TT in T
    at0 = JuLIP.bulk(:Si, cubic=true) * 2
    at1 = deepcopy(at0)
    at2 = deepcopy(at0)
    
    A = at0.X[22]
    B = at0.X[57]
    
    A_prime, B_prime = adjust_positions(A, B)
    
    at1.X[22] = A_prime
    at1.X[57] = B_prime
    
    at2.X[22] = B
    at1.X[57] = A
    
    sys0 = JuLIP2AtomsBuilder(at0)
    sys1 = JuLIP2AtomsBuilder(at1)
    sys2=  JuLIP2AtomsBuilder(at2)
    r11 = position(sys0, 22)
    r12 = position(sys1, 22)
    r21 = position(sys0, 57)
    r22 = position(sys1, 57)
    
    zU = AtomsBuilder.atomic_number(sys0, 1)
    sw = StillingerWeber() 
    calc = sw
    
    # generate an initial guess for the MEP 
    path_init = [ (sys = deepcopy(sys0); set_position!(sys, 22, (1-t)*r11 + t * r12);set_position!(sys, 57, (1-t)*r21 + t * r22) ; sys)
                for t in range(0.0, 1.0, length = Nimg) ]
    
    nlist = PairList.(path_init, cutoff_radius(sw))
    XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
    S = entropy_onlyentropy.(path_init,XX_flat,ℓ)
    δS = -1*(kb2ev* S .- S_0)
    dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
    obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
    xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_init ]
    
    preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
    neb = ODENEB(reltol=1e-4, k = 0.002, interp=3, tol = 3e-1, maxnit = 2000,
    precon_scheme = preconI, verbose = 2, maxtol = 1e4)
    xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))
    
    reaction_coords = range(0, stop=1, length=Nimg)
    F_string = [ obj_f(x) for x in xx_string ]
    
    using Plots
    display(scatter(reaction_coords,F_string))
    
    using Interpolations
    xx_matrix = hcat(xx_string...) 
    interpolants = [interpolate(xx_matrix[i, :], BSpline(Cubic()), OnCell()) for i in 1:size(xx_matrix,1)]
    
    function u(ξ)
        return [interpolants[i](ξ * (length(reaction_coords) - 1) + 1) for i in 1:size(xx_matrix,1)]
    end
    
    function du_dxi(ξ)
        ξ_scaled = ξ * (length(reaction_coords) - 1) + 1  
        return [Interpolations.gradient(interpolants[i], ξ_scaled)[1] * (length(reaction_coords) - 1) for i in 1:size(xx_matrix,1)]
    end
    
    s_x = xx_string[argmax(F_string)]
    at0.X += x2Rs(s_x)
    saddle = AosSystem(JuLIP2AtomsBuilder(at0))
    
    F_saddle = maximum(F_string)
    F_min =  F_string[1]
    F_HTST = F_saddle - F_min #ev
    k_HTST= exp(-(F_HTST)/(kb2ev*TT)) * 10^13
    
    s_xx = Rs2x(at0.X)
    saddleimgpos = argmax(F_string)/Nimg
    ∇S = dot(kb2ev * entropy_withgradient(saddle, s_xx, ℓ).grad[1], du_dxi(saddleimgpos))
    H = du_dxi(saddleimgpos)' * entropy_withgradientandhess(saddle, s_xx, ℓ)[2] * du_dxi(saddleimgpos)
    
    correction  = exp(0.5 * (1/kb2ev) * TT * ∇S^2/H )
    
    k_vtst = k_HTST * correction
    push!(ratio, correction)
    push!(Htst, k_HTST)
    push!(Vtst, k_vtst)
end

using Plots
plot(T, Htst, yscale=:log10, legend = :bottomright)
plot!(T, Vtst, yscale=:log10, legend = :bottomright)
plot(T, Htst.-Vtst, yscale=:log10, legend = :bottomright)