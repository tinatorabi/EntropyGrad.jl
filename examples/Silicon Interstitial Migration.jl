using AtomsBase, DecoratedParticles, AtomsBuilder, 
      GeomOpt, StaticArrays, Unitful, LinearAlgebra , 
      EmpiricalPotentials, SaddleSearch, Interpolations, JuLIP
using GeomOpt:DofManager, set_dofs!
using AtomsCalculators: virial, forces, potential_energy
import Random

using EntropyGrad

# Setup
GO = GeomOpt      
DP = DecoratedParticles
Random.seed!(100)
AtomsBase.atomic_number(sys::SoaSystem, i::Integer) = sys.arrays.𝑍[i]
sw = EmpiricalPotentials.StillingerWeber() 
kb2ev = 8.617333262 * 1e-5 # conversion constant
ℓ = 25 # Set number of quadrature points to use

# Define `at0` with an interstitial atom at a tetrahedral site
at0 = JuLIP.bulk(:Si, cubic=true) * 2
at1 = deepcopy(at0)
lattice_constant = 5.43

# Two tetrahedral interstitial sites
tetrahedral_site_1 = lattice_constant * SVector(1/2, 1/2, 1/2) 
tetrahedral_site_2 = lattice_constant * SVector(3/4, 3/4, 3/4)

# Insert an interstitial atom at the first tetrahedral site in `at0`
insert!(at0.X, length(at0.X)+1, tetrahedral_site_1)
insert!(at1.X, length(at1.X)+1, tetrahedral_site_2)
sys0 = AosSystem(JuLIP2AtomsBuilder(at0))
sys1 = AosSystem(JuLIP2AtomsBuilder(at1))
r0 = AtomsBuilder.Chemistry.rnn(:Si)
def_free= AosSystem(JuLIP2AtomsBuilder(deepcopy(at0)))
nlist0 = PairList(def_free, cutoff_radius(sw))
XX_0 = reduce(vcat, nlist0.X)
S_0 = kb2ev * entropy_onlyentropy(def_free,XX_0,ℓ)[1]
S_0_grad = kb2ev * entropy_onlygradient(def_free,XX_0,ℓ)[1]
zU = AtomsBase.atomic_number(sys0, 1)
calc = sw

# Generate an initial guess for the MEP 
Nimg = 7 # Number of images along the path
old = deepcopy(position(sys0,65))
𝐫2 = deepcopy(position(sys1,65))
path_init = [ (sys = deepcopy(sys0); set_position!(sys, 65, (1-t)*old+t*𝐫2); sys)
               for t in range(0.0, 1.0, length = Nimg) ]
nlist = PairList.(path_init, cutoff_radius(sw))
XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
S = entropy_onlyentropy.(path_init,XX_flat,ℓ)
δS = -1*(kb2ev* S .- S_0)

@info("Initial guess, energy difference")
Temp = 0
F_init = ustrip(potential_energy.(path_init, Ref(calc))) - Temp*δS
δF_init = round.( F_init .- F_init[1], digits=4)

# Loop over temperatures and compute paths
T = [0, 200, 400, 600, 800, 1000]
global Estring = []
global xx_string
sysoriginal = deepcopy(sys0) # Keep original configuration accessible
for TT in T
        dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
        obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
        xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_init ]

        preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
        neb = ODENEB(reltol=1e-3, interp=3, k=0.002, tol = 1e-1, maxnit = 1000,
        precon_scheme = preconI, verbose = 2, fixed_ends=false, path_traverse=serial())
        xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))

        reaction_coords = range(0, stop=1, length=Nimg)
        F_string = [ obj_f(x) for x in xx_string ]
        push!(Estring,F_string)
end

plot(reaction_coords, Estring)

using Plots
reaction_coords = range(0, stop=1, length=Nimg)

# Compute continuous path interpolation.
spline1 = cubic_spline_interpolation(reaction_coords, Estring[1])
spline2 = cubic_spline_interpolation(reaction_coords, Estring[2])
spline3 = cubic_spline_interpolation(reaction_coords, Estring[3])
fine_coords = range(0, stop=1, length=300)
fine_energies1 = [spline1(x) for x in fine_coords]
fine_energies2 = [spline2(x) for x in fine_coords]
fine_energies3 = [spline3(x) for x in fine_coords]
plot(fine_coords,  fine_energies1.-fine_energies1[1], label="0K", color=1, xlabel="Reaction Coordinate", ylabel="Free Energy (eV)",legend=:topright)
plot!(fine_coords, fine_energies2.-fine_energies2[1], label="200K", color=2)
plot!(fine_coords, fine_energies3.-fine_energies3[1], label="400K", color=3)