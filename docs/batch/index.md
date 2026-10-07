# HTCondor at CERN

These examples describe the submission structure for buildup scans and
single-bunch stability simulations. Replace the example EOS locations,
environment activation script, resources, and simulation inputs before use.

```{toctree}
:maxdepth: 1

buildup-scans
stability
```

## Where files live

For the standard CERN batch setup:

| Location | Use in these examples |
| --- | --- |
| AFS, mounted on submit, scheduler, and worker nodes | Submit descriptions, executable scripts, DAG files, scheduler logs |
| EOS, mounted on submit and worker nodes, but not the standard scheduler nodes | Simulation inputs, results, and optionally the Python environment |
| Worker scratch (`_CONDOR_SCRATCH_DIR`) | Running the simulation and producing local output |

Submit from an AFS directory. Keep scheduler-facing paths there. An EOS mount
path used **inside the worker script** is different from an EOS path that the
scheduler would need to open. Use the CERN XRootD transfer plugin with `root://`
URLs for scheduler-managed EOS data transfers, or run `xrdcp` inside the worker.
See the [CERN batch documentation](https://batchdocs.web.cern.ch/) for site setup
and authentication requirements. Workers need credentials and permission to
read the inputs and write the destination for the duration of the job.

## Python environment

For the default examples, install the **complete Python/Conda environment on
EOS**, including its interpreter, packages, and activation scripts. Workers
activate the environment by sourcing:

```bash
source /eos/project/e/PROJECT/environments/activate-ecloud.sh
python -c 'import PyECLOUD, PyPARIS'
```

Create this script to activate your EOS-hosted virtualenv or Conda environment.
For Conda, source the EOS-hosted installation's `conda.sh` before activating
the environment by its EOS path. Use an environment compatible with the
worker OS. Follow the
[installation instructions](../installation.md) when preparing it.

### Alternative: a CVMFS container

We recommend `ghcr.io/ekatralis/ecloud-containers:latest`, maintained for
electron cloud simulations and available through CERN CVMFS. This avoids
maintaining your own Python environment on EOS.
Replace environment activation and the Python invocation with a wrapper such as:

```bash
run_python() {
    apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \
        --home "$_CONDOR_SCRATCH_DIR" --pwd "$PWD" \
        /cvmfs/unpacked.cern.ch/ghcr.io/ekatralis/ecloud-containers:latest/ \
        python "$@"
}
run_python run_sim.py
```

The `latest` image evolves; record `ECLOUD_CONTAINER_VERSION` from inside the
container with your run and keep the software environment consistent across
continuation jobs. Check
the scratch-directory bindings for your image/site configuration. Run EOS
transfers from the host worker shell, where its credentials and XRootD tools
are available. Container selection and output-transfer strategy are independent.

## Output transfers and held jobs

Configure output transfers in the **HTCondor submit file** (`htcondor.sub`).
For the `root://` destinations used here, explicitly list individual files in
the submit file's `transfer_output_files`. Do not use wildcards or rely on directory transfer via
this plugin. Archive a directory into one named file, or transfer its files
yourself from the worker.

**Every file listed in `transfer_output_files` must exist when output transfer
runs.** A missing file causes a transfer error and can put the job on hold,
making it appear stuck after the calculation finishes. This is why the buildup
example lists one archive and creates a diagnostic version before starting.
An archive's existence does not establish simulation success.

With `transfer_output_files = ""`, the worker is responsible for saving the
simulation data. HTCondor still transfers standard output and error to the
locations specified by `output` and `error`.
`when_to_transfer_output = ON_EXIT` does not recover outputs on eviction or
removal. See the [HTCondor file-transfer manual](https://htcondor.readthedocs.io/en/latest/users-manual/file-transfer.html).

Inspect held jobs with `condor_q -hold` and the job's `HoldReason` before
releasing them. Check both the simulation exit status and transfer success.
Start with one short job before submitting a scan or a long continuation chain.
