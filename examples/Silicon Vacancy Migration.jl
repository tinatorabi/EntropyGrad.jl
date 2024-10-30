using AtomsBase, DecoratedParticles, AtomsBuilder, 
      GeomOpt, StaticArrays, Unitful, LinearAlgebra , 
      EmpiricalPotentials, SaddleSearch, JuLIP, Interpolations
using GeomOpt: DofManager, set_dofs!
using AtomsCalculators: virial, forces, potential_energy
import Random

GO = GeomOpt      
DP = DecoratedParticles

using EntropyGrad

function free_energy_dofs(sys, calc, dofmgr, Temp, s0, x)
    set_dofs!(sys, dofmgr, x)
    nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber()))
    XX_flat = reduce(vcat, nlist.X)
    EE = ustrip(potential_energy(sys, calc))
    if Temp > 0
        return EE - Temp * (s0-kb2ev*(entropyonlyentropy(sys,XX_flat,ℓ)[1]))
    else
        return EE
    end
end

function FE_gradient_dofs(sys, calc, dofmgr,Temp, x::AbstractVector{T}) where {T} 
    set_dofs!(sys, dofmgr, x)
    nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber()))
    XX_flat = reduce(vcat, nlist.X)
    ∇E = GO.gradient_dofs(sys, calc, dofmgr, x)
    if Temp > 0 
        return ∇E - Temp*kb2ev*entropyonlygradient(sys,XX_flat,ℓ)[1]
    else
        return ∇E
    end
end

function get_obj_fg_FE(sys::AbstractSystem, calc, s0, dofmgr::DofManager, Temp) 
    f = x -> free_energy_dofs(sys, calc, dofmgr, Temp, s0, x)
    g = x -> FE_gradient_dofs(sys, calc, dofmgr,Temp, x)
    return f, g 
end 

# Setup
Random.seed!(100)
AtomsBase.atomic_number(sys::SoaSystem, i::Integer) = sys.arrays.𝑍[i]
sw = StillingerWeber()  # empirical potential
kb2ev = 8.617333262 * 1e-5  # conversion factor
ℓ = 30  # number of quadrature points in the contour integral

# Pick an arbitrary FCC system (needs to be closed-packed
r0 = AtomsBuilder.Chemistry.rnn(:Si)
sys0 = AosSystem( AtomsBuilder.bulk(:Si, cubic=true, pbc=true))
nlist0 = EmpiricalPotentials.PairList(sys0, cutoff_radius(StillingerWeber()))
XX_0 = reduce(vcat, nlist0.X)
S_0 = kb2ev * entropyonlyentropy(sys0,XX_0,ℓ)[1]
# Sanity check that x2 is a nearest-neighbour of x1
@assert norm(position(sys0, 1)) == 0u"Å"
@assert norm(position(sys0, 2)) <= 1.0001 * r0
𝐫2 = position(sys0, 2)

# Construct system with defect
deleteat!(sys0.particles, 2)
sys0 = AosSystem(sys0)
zU = AtomsBase.atomic_number(sys0, 1)
sw = StillingerWeber() 
calc = sw

# generate an initial guess for the MEP 
Nimg = 50
old = deepcopy(position(sys0,1))
path_init = [ (sys = deepcopy(sys0); set_position!(sys, 1, (1-t)*old+t*𝐫2); sys)
               for t in range(0.0, 1.0, length = Nimg) ]
nlist = EmpiricalPotentials.PairList.(path_init, cutoff_radius(StillingerWeber()))
XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
S = [s for s in entropyonlyentropy.(path_init,XX_flat,ℓ)]
δS = -1*(kb2ev* S .- S_0)
Temp = 0
F_init = ustrip(potential_energy.(path_init, Ref(calc))) - Temp*δS

T = [0, 200] # Temperatures that will be tested. 
# Careful, depending on your system this can be quick or slow. Try with only a few temperatures first.
global Estring = []
global xx_string
sysoriginal = deepcopy(sys0) # Store initial configuration to allow resetting
for TT in T
        sys0 = deepcopy(sysoriginal) # Ensure that the order in which we check temperatures is irrelevant
        dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
        obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
        xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_init ]
        preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
        # This uses NEB, String searches are also supported
        neb = ODENEB(reltol=1e-3, interp=3, k=0.0002, tol = 1e-1, maxnit = 1000,
        precon_scheme = preconI, verbose = 2, fixed_ends=false, path_traverse=serial())
        xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))
        reaction_coords = range(0, stop=1, length=Nimg)
        F_string = [ obj_f(x) for x in xx_string ]
        push!(Estring, F_string)
end

# Plot with respect to reaction coordinates
using Plots
scatter(reaction_coords, Estring[1], label="0K", marker=:circle, line=false,
     xlabel="Reaction Coordinate",
     ylabel="Free Energy (eV)",
     legend=:topright)
scatter(reaction_coords, Estring[2], label="200K", marker=:circle, line=false,
     xlabel="Reaction Coordinate",
     ylabel="Free Energy (eV)",
     legend=:topright)

# Export .xyz files, e.g. for animations in Avogadro
using AtomsIO
dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
for j = 1:Nimg
        set_dofs!(sys0, dofmgr, xx_string[j])
        save_system(AtomsIO.ExtxyzParser(), "$j"*"silicontest.xyz", sys0|>FastSystem)
end
