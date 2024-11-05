using AtomsBase, DecoratedParticles, AtomsBuilder, 
      GeomOpt, Test, StaticArrays, Unitful, LinearAlgebra , 
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
ℓ = 30 # Set number of quadrature points to use

function free_energy_dofs(sys, calc, dofmgr, Temp, s0, x)
    set_dofs!(sys, dofmgr, x)
    nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(sw))
    XX_flat = reduce(vcat, nlist.X)
    EE = ustrip(potential_energy(sys, calc))
    if Temp > 0
        return EE - Temp * (s0-kb2ev*(entropy_onlyentropy(sys,XX_flat,ℓ)[1]))
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
        return ∇E - Temp*kb2ev*entropy_onlygradient(sys,XX_flat,ℓ)[1]
    else
        return ∇E
    end
end

function get_obj_fg_FE(sys::AbstractSystem, calc, s0, dofmgr::DofManager, Temp) 
    f = x -> free_energy_dofs(sys, calc, dofmgr, Temp, s0, x)
    g = x -> FE_gradient_dofs(sys, calc, dofmgr,Temp, x)
    return f, g 
end 

function JuLIP2AtomsBuilder(at)
    # This function converts between incompatible Atoms structs from JuLIP to AtomsBuilder
    # For now this is hard-coded for all atoms being the same element.
    @assert all(at.Z .== at.Z[1])
    newCell = Matrix(at.cell) * Unitful.angstrom
    newX = at.X * Unitful.angstrom
    Nat = length(newX)
    v = (true, true, true)
    syms = Chemistry.chemical_symbol.(fill(at.Z[1],Nat))
    atoms = [ Atom(syms[i], newX[i]) for i in 1:Nat ]
    bc =  [ (v[i] ? AtomsBase.Periodic() : nothing) for i = 1:3 ]
    bb = [newCell[i, :] for i = 1:3]
    return AosSystem(AtomsBuilder.FlexibleSystem(atoms, bb, bc))
end

# Define `at0` with an interstitial atom at a tetrahedral site
at0 = JuLIP.bulk(:Si, cubic=true) * 2
at1 = deepcopy(at0)
def_free= AosSystem(JuLIP2AtomsBuilder(deepcopy(at0)))
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
nlist0 = EmpiricalPotentials.PairList(def_free, cutoff_radius(sw))
XX_0 = reduce(vcat, nlist0.X)
S_0 = kb2ev * entropy_onlyentropy(def_free,XX_0,ℓ)[1]
zU = AtomsBase.atomic_number(sys0, 1)
calc = sw

# Generate an initial guess for the MEP 
Nimg = 10 # Number of images along the path
old = deepcopy(position(sys0,65))
𝐫2 = deepcopy(position(sys1,65))
path_init = [ (sys = deepcopy(sys0); set_position!(sys, 65, (1-t)*old+t*𝐫2); sys)
               for t in range(0.0, 1.0, length = Nimg) ]
nlist = EmpiricalPotentials.PairList.(path_init, cutoff_radius(sw))
XX_flat = [reduce(vcat, ttn.X) for ttn in nlist]
S = [s for s in entropy_onlyentropy.(path_init,XX_flat,ℓ)]
δS = -1*(kb2ev* S .- S_0)

@info("Initial guess, energy difference")
Temp = 0
F_init = ustrip(potential_energy.(path_init, Ref(calc))) - Temp*δS
δF_init = round.( F_init .- F_init[1], digits=4)

# Loop over temperatures and compute paths
T = [0, 100 , 200]
global Estring = []
global xx_string
sysoriginal = deepcopy(sys0) # Keep original configuration accessible
sys0 = deepcopy(sysoriginal)
for TT in T
        dofmgr = GeomOpt.DofManager(sys0; variablecell=false)
        obj_f, obj_g = get_obj_fg_FE(sys0, calc, S_0, dofmgr, TT)
        xx_init = [ GeomOpt.get_dofs(sys, dofmgr) for sys in path_init ]

        FE_check = [ obj_f(x) for x in xx_init ]
        all(FE_check .≈ ustrip.(F_init))

        preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)
        neb = ODENEB(reltol=1e-3, interp=3, k=0.0002, tol = 1e-1, maxnit = 1000,
        #neb = StaticNEB(alpha=1e-3, interp=3, k=0.0002, tol = 6e-2, maxnit = 1000,
        precon_scheme = preconI, verbose = 2, fixed_ends=false, path_traverse=serial())
        xx_string, _, _ = SaddleSearch.run!(neb, obj_f, obj_g, Path(xx_init))

        reaction_coords = range(0, stop=1, length=Nimg)
        @info("energy along initial / final path")
        F_string = [ obj_f(x) for x in xx_string ]
        push!(Estring,F_string)
end

using Plots
reaction_coords = range(0, stop=1, length=Nimg)
scatter(reaction_coords, Estring, label="0K", marker=:circle, line=false,
     xlabel="Reaction Coordinate",
     ylabel="Free Energy (eV)",
     legend=:topright)

# Compute continuous path interpolation
spline1 = cubic_spline_interpolation(reaction_coords, Estring[1])
spline2 = cubic_spline_interpolation(reaction_coords, Estring[2])
spline3 = cubic_spline_interpolation(reaction_coords, Estring[3])
spline4 = cubic_spline_interpolation(reaction_coords, Estring[4])
fine_coords = range(0, stop=1, length=300)
fine_energies1 = [spline1(x) for x in fine_coords]
fine_energies2 = [spline2(x) for x in fine_coords]
fine_energies3 = [spline3(x) for x in fine_coords]
fine_energies4 = [spline4(x) for x in fine_coords]

plot(fine_coords,  fine_energies1, label=L"\mathrm{0K}", color=1, xlabel=L"\mathrm{Reaction\ Coordinate}", ylabel=L"\mathrm{Free\ Energy\ (eV)}",xtickfont=font(9, "Times New Roman"),ytickfont=font(9, "Times New Roman"),xguidefont=font(10, "Times New Roman"),yguidefont=font(10, "Times New Roman"),legend=:topright)
plot!(fine_coords, fine_energies2, label=L"\mathrm{50K}", color=2)
plot!(fine_coords, fine_energies3, label=L"\mathrm{100K}", color=3)
plot!(fine_coords, fine_energies4, label=L"\mathrm{200K}", color=4)