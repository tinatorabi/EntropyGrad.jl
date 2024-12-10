using AtomsBase, DecoratedParticles, AtomsBuilder, 
      GeomOpt, StaticArrays, Unitful, LinearAlgebra , 
      EmpiricalPotentials, SaddleSearch, JuLIP, Interpolations
using GeomOpt:DofManager, set_dofs!
using AtomsCalculators: virial, forces, potential_energy
import Random

GO = GeomOpt      
DP = DecoratedParticles

using EntropyGrad

Random.seed!(100)
AtomsBase.atomic_number(sys::SoaSystem, i::Integer) = sys.arrays.𝑍[i]
sw = EmpiricalPotentials.StillingerWeber()  # empirical potential
kb2ev = 8.617333262 * 1e-5  # conversion factor
ℓ = 25  # number of quadrature points in the contour integral

# Pick an arbitrary FCC system (needs to be closed-packed)
r0 = AtomsBuilder.Chemistry.rnn(:Si)
sys0 = AosSystem( AtomsBuilder.bulk(:Si, cubic=true, pbc=true) * 2)
nlist0 = PairList(sys0, cutoff_radius(EmpiricalPotentials.StillingerWeber()))
XX_0 = reduce(vcat, nlist0.X)
S_0 = kb2ev * entropy_onlyentropy(sys0,XX_0,ℓ)[1]

# Sanity check that x2 is a nearest-neighbour of x1
@assert norm(position(sys0, 1)) == 0u"Å"
@assert norm(position(sys0, 2)) <= 1.0001 * r0
𝐫2 = deepcopy(position(sys0, 2))

# Construct system with defect
deleteat!(sys0.particles, 2)
sys0 = AosSystem(sys0)
zU = AtomsBase.atomic_number(sys0, 1)
sw = StillingerWeber() 
calc = sw

# Generate an initial guess for the MEP 
Nimg = 30
old = deepcopy(position(sys0,1))
path_init = [ (sys = deepcopy(sys0); set_position!(sys, 1, (1-t)*old+t*𝐫2); sys)
               for t in range(0.0, 1.0, length = Nimg) ]
nlist = PairList.(path_init, cutoff_radius(StillingerWeber()))
XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
S = entropy_onlyentropy.(path_init,XX_flat,ℓ)
δS = -1*(kb2ev* S .- S_0)

# We need to re-initialise into the minima for this problem and to get closer to them we will
# first run a T=O run and find the new minima from there.
T = [0]
sysoriginal = deepcopy(sys0) # Store initial configuration
global Estring = []
global xx_string
for TT in T
        dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
        obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
        xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_init ]
        preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
        # This uses NEB, String searches are also supported
        neb = ODENEB(reltol = 1e-2, k =0.003, tol = 1e-1, maxnit = 2000,
        precon_scheme = preconI, verbose = 2)
        xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))
        reaction_coords = range(0, stop=1, length=Nimg)
        F_string = [ obj_f(x) for x in xx_string ]
        push!(Estring, F_string)
end

using Plots
reaction_coords = range(0, stop=1, length=Nimg)
scatter(reaction_coords, Estring, label="0K", marker=:circle, line=false,
     xlabel="Reaction Coordinate",
     ylabel="Free Energy (eV)",
     legend=:topright)

# The following uses the above computation to compute the vacancy migration starting from
# minimum to minimum instead of original lattice positions
sys0 = deepcopy(sysoriginal) # Ensure that the order in which we check temperatures is irrelevant
XX_start, XX_end = xx_string[argsmallest(Estring[1],2)][sortperm(argsmallest(Estring[1],2))]
nlist0 = PairList(sys0, cutoff_radius(EmpiricalPotentials.StillingerWeber()))
XX_prev = nlist0.X
XX_start = (XX_prev.+x2Rs(XX_start)).*Unitful.angstrom
XX_end = (XX_prev.+x2Rs(XX_end)).*Unitful.angstrom
DecoratedParticles.set_positions!(sys0, XX_start)

# Generate an initial guess for the MEP 
Nimg = 50
path_new = [ (sys = deepcopy(sys0); DecoratedParticles.set_positions!(sys, (1-t).*XX_start+t*XX_end); sys)
               for t in range(0.0, 1.0, length = Nimg) ]
nlist = PairList.(path_new, cutoff_radius(StillingerWeber()))
XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
S = entropy_onlyentropy.(path_new,XX_flat,ℓ)
δS = -1*(kb2ev * S .- S_0)

TT = 0 # VTST calculations perturb away from T = 0K
sysoriginal = deepcopy(sys0) # Store initial configuration to allow resetting
global Estring = []
global xx_string
sys0 = deepcopy(sysoriginal) # Ensure that the order in which we check temperatures is irrelevant
dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_new ]
preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
# This uses NEB, String searches are also supported
neb = ODENEB(reltol = 1e-3, k =0.003, tol = 1e-1, maxnit = 200,
precon_scheme = preconI, verbose = 2)
xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))
reaction_coords = range(0, stop=1, length=Nimg)
F_string = [ obj_f(x) for x in xx_string ]
push!(Estring, F_string)

# Compute continuous path interpolation
spline1 = cubic_spline_interpolation(reaction_coords, Estring[1])
fine_coords = range(0, stop=1, length=300)
fine_energies1 = [spline1(x) for x in fine_coords]
plot(fine_coords,  fine_energies1.-fine_energies1[1], label="0K", color=1, xlabel="Reaction Coordinate", ylabel="Free Energy (eV)",legend=:topright)


# VTST calculations based on vacancy migration at 0K

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

at0 = JuLIP.bulk(:Si, cubic=true, pbc=true) * 2
deleteat!(at0, 2)

s_x = xx_string[argmax(F_string)]
at0.X += x2Rs(s_x)
saddle = AosSystem(JuLIP2AtomsBuilder(at0))

F_saddle = maximum(F_string)
F_min =  minimum(F_string)
F_HTST = F_saddle - F_min #ev

s_xx = Rs2x(at0.X)
saddleimgpos = (argmax(F_string))/Nimg
∇S = dot(kb2ev * entropy_withgradient(saddle, s_xx, ℓ).grad[1], du_dxi(saddleimgpos))
H = du_dxi(saddleimgpos)' * entropy_withgradientandhess(saddle, s_xx, ℓ)[2] * du_dxi(saddleimgpos)
correction  = 0.5 * (1/kb2ev) * ∇S^2/H 

# Define and plot VTST and HTST for temperature ranges
htst(T) = 1/exp(T * correction)  * exp(-(F_HTST)/(kb2ev*T)) * 10^13
vtst(T) = exp(-(F_HTST)/(kb2ev*T)) * 10^13

T = 50:1200
plot(T, htst, yscale=:log10, legend = :bottomright, label="HTST") 
plot!(T, vtst, yscale=:log10, legend = :bottomright, label="VTST")