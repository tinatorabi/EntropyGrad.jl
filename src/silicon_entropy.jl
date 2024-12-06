function free_energy_dofs(sys, calc, dofmgr, Temp, s0, x)
      set_dofs!(sys, dofmgr, x)
      nlist = NeighbourLists.PairList(sys, cutoff_radius(StillingerWeber()))
      XX_flat = reduce(vcat, nlist.X)
      EE = ustrip(potential_energy(sys, calc))
      if Temp > 0
          return EE - Temp * (s0-kb2ev*(entropy_onlyentropy(sys,XX_flat,ℓ)[1]))
      else
          return EE
      end
  end
  
  function FE_gradient_dofs(sys, calc, dofmgr, Temp, x::AbstractVector{T}) where {T} 
      set_dofs!(sys, dofmgr, x)
      nlist = NeighbourLists.PairList(sys, cutoff_radius(StillingerWeber()))
      XX_flat = reduce(vcat, nlist.X)
      ∇E = GeomOpt.gradient_dofs(sys, calc, dofmgr, x)
      if Temp > 0 
          return ∇E + Temp*kb2ev*entropy_onlygradient(sys,XX_flat,ℓ)[1]
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
      # This helper function converts between incompatible Atoms structs from JuLIP to AtomsBuilder
      # For now this is hard-coded for all atoms being the same element.
      @assert all(at.Z .== at.Z[1])
      newCell = Matrix(at.cell) * Unitful.angstrom
      newX = at.X * Unitful.angstrom
      Nat = length(newX)
      v = (true, true, true) # only for periodic systems
      syms = Chemistry.chemical_symbol.(fill(at.Z[1],Nat))
      atoms = [ Atom(syms[i], newX[i]) for i in 1:Nat ]
      #bc =  [ (v[i] ? AtomsBase.Periodic() : nothing) for i = 1:3 ]
      bb = [newCell[i, :] for i = 1:3]
      return AosSystem(AtomsBuilder.FlexibleSystem(atoms, bb, v))
  end