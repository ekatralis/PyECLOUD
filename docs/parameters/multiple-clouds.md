# Simulations with multiple clouds

The code provides the option of including more than one cloud in the simulation. Each cloud is
influenced by the other clouds through their space charge. Cross-ionization processes between
different clouds can be defined. These options can be enabled by providing the following
additional information.

In the **simulation_parameters.input** file:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **additional_clouds_file_list**
  - (optional – default=\[\]) List of names of additional cloud files.
* - **cross_ion_definitions**
  - (optional – default={ }) Dictionary that describes the desired cross-ionization processes (more
    details can be found in <https://indico.cern.ch/event/835473/contributions/3525107/>).
```

Each cloud file should contain a subset of the input parameters specified in the main input
files, as listed below. The following parameters are required in the cloud input file.

**Mandatory cloud parameters**

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **cloud_mass**
  - See Simulation parameters section.
* - **cloud_charge**
  - Cloud particle charge. See the Simulation parameters section.
* - **gas_ion_flag**
  - See Machine parameters section.
* - **photoem_flag**
  - —
* - **init_unif_flag**
  - —
* - **init_unif_edens_flag**
  - —
* - **switch_model**
  - See Secondary emission parameters section.
```

In addition, a cloud file can contain the following optional parameters. Any optional parameter
that is not specified in the cloud input file takes its value from the main input, and gets the
default value if the parameter is not given in the main input files.

**Optional cloud parameters**

```{list-table}
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - **MP management settings**
  - See the Simulation parameters section for the detailed list of parameters and their usage.
* - **Saving settings**
  - —
* - **Log and progress files**
  - —
* - **N_sub_steps**
  - See under “Tracking and magnetic field” in the Machine parameters section.
* - **Residual gas ionization parameters**
  - See the Machine parameters section for the detailed list of parameters and their usage.
* - **Photoemission parameters**
  - —
* - **Uniform initial distribution**
  - —
* - **Uniform initial density**
  - —
* - **Secondary emission parameters**
  - As listed in the Secondary emission parameters section.
```

Each cloud produces its own .mat output file, as described in the original reference manual.
The file is named according to the main output file name and the cloud file name.
