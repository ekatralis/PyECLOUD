# Beam parameters

**Basic definitions**

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **m0_part**
  - (optional – default=proton mass)

    \[Kg\] Mass of beam particles.
* - **q_part**
  - (optional – default=proton charge)

    \[C\] Charge of beam particles.
* - **energy_eV**
  - \[eV\] Total energy of the beam particles.
* - **Dp_p**
  - (optional – default=0)

    Momentum spread (r.m.s.) of the beam. This parameter is not used if the transverse beam size is
    directly provided (vars. sigmax, sigmay).
* - **nemittx, nemitty**
  - \[m\] Normalized transverse emittance of the beam. These parameters are not used if the
    transverse beam size is directly provided (vars. sigmax, sigmay).
* - **x_beam_pos, y_beam_pos**
  - (optional – default=0)

    \[m\] Transverse position of the beam within the vacuum chamber.
* - **sigmax, sigmay**
  - (optional – default=-1)

    \[m\] Transverse geometrical beam size. If sigmax, sigmay=-1 the beam size is calculated
    through the energy, the normalized emittance, the dispersion and the momentum spread of the
    beam.
```

**Transverse electric field of the beam**

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **beam_field_file**
  - Name of a .mat file containing the map of the transverse beam electric field.

    If beam_field_file = -1 or beam_field_file = ‘computeFD’ the beam field map is computed using
    the embedded Finite Difference (FD) Poisson solver.

    If beam_field_file=‘computeBE’ the electric field is computed using the Bassetti-Erskine
    formula and the image terms for the elliptical boundary conditions (this option is available
    only when the chamber profile is elliptical).

    If beam_field_file=‘compute_FDSW_multigrid’ the beam field map is computed using the embedded
    Finite Difference (FD) Poisson solver in the multigrid case.
* - **save_beam_field_file_as**
  - (optional – default: the beam field map is not saved)

    Name of the file where the employed field map is saved.
```

When beam_field_file = -1 or beam_field_file = ’computeFD’ the following parameter must be
provided:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **Dh_beam_field**
  - Grid size for the FD solver and for the saved field map.
```

When beam_field_file = ‘computeBE’ the following parameters must be provided:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **Nx, Ny**
  - Number of points of the employed field map.
* - **nimag**
  - Number of image terms.
```

When beam_field_file = ’compute_FDSW_multigrid’ the following parameters must be provided:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **Dh_beam_field**
  - Grid size for the FD solver and for the saved field map.
* - **f_telescope_beam**
  - Magnification factor between grids (it must be always $0<f<1$)
* - **target_grid_beam**
  - Target grid parameters. It is a dictionary containing: ’x_min_target’, ’x_max_target’,
    ’y_min_target’, ’y_max_target’ and ’Dh_target’.
* - **N_nodes_discard_beam**
  - Number of nodes at the edge of the internal grids which are discarded in the field
    interpolation (going to coarser grid).
* - **N_min_Dh_main_beam**
  - Minimum size of the first internal grid in the $\Delta h$ of the main grid.
```

**Beam longitudinal profile**

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **b_spac**
  - Bunch spacing. It is used to generate the beam profile when it is defined in the form of a
    bunched beam. It is also the time interval for cleaning and regeneration of the MP set, as well
    as for saving.
* - **fact_beam**
  - Rescaling factor applied to the beam profile.
* - **coast_dens**
  - \[p/m\] Coasting beam density added to the beam profile.
* - **flag_bunched_beam**
  - Two possible settings:

    1.  flag_bunched_beam = 0 the beam profile is loaded from file.
    2.  flag_bunched_beam = 1 the beam profile is generated from a filling pattern.
```

When flag_bunched_beam = 1 the following parameters must be provided:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **sigmaz**
  - \[m\] Bunch length (1$\sigma$).
* - **t_offs**
  - \[s\] Delay in the longitudinal profile (n.b. if t_offs=0 only half of the first bunch is
    simulated).
* - **filling_pattern_file**
  - Name of a file providing the intensities of the different bunches (zeros for empty slots). The
    data can be provided also in form of a python list with no need to provide an input file, e.g.
    filling_pattern_file=4\*(72\*\[1.\]+8\*\[0.\]).
```

When flag_bunched_beam = 0 the following parameter must be provided:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **beam_long_prof_file**
  - Name of a .mat file providing the longitudinal beam profile. This file will also define the
    time step used for the simulation.
```
