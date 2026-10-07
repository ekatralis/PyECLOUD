# Buildup simulation scans

Use one independent HTCondor job per scan point. Each job receives common files
and its own input archive, runs in scratch, and returns one explicitly named
archive to EOS. Read the [storage and environment setup](index.md) first.

## Prepare the scan

Keep this submission layout on AFS:

```text
scan/
  htcondor.sub
  job.sh
  sims.txt
  logs/
  point_000/inputs.tgz
  point_001/inputs.tgz
```

`sims.txt` contains one relative directory name per line (`point_000`,
`point_001`, ...), without spaces. Put each point's `.input` and `.beam` files
at the top level of its `inputs.tgz`. Put `run_sim.py`, common geometry, and
other shared files at the top level of `jobfiles.tgz` on EOS. Paths in the
simulation inputs must resolve within the extracted working directory.

## Submit description

Save as `htcondor.sub`. Resource requests below are illustrative; allow scratch
space for inputs, unpacked results, and the compressed archive simultaneously.

```text
universe = vanilla
executable = job.sh
output = logs/$(ClusterId).$(ProcId).out
error = logs/$(ClusterId).$(ProcId).err
log = logs/$(ClusterId).log

should_transfer_files = YES
transfer_input_files = $(folder)/inputs.tgz, root://eosproject-e.cern.ch//eos/project/e/PROJECT/scan/jobfiles.tgz
when_to_transfer_output = ON_EXIT
transfer_output_files = output.tgz
output_destination = root://eosproject-e.cern.ch//eos/project/e/PROJECT/scan/results/$(folder)/$(ClusterId).$(ProcId)/
MY.XRDCP_CREATE_DIR = True

request_cpus = 1
request_memory = 4GB
request_disk = 10GB
+JobFlavour = "tomorrow"
queue folder from sims.txt
```

Each attempt has a separate destination to avoid overwriting earlier results.
The event log stays on AFS; the named archive goes to EOS.

## Worker script

Save as `job.sh`. The archive includes inputs, outputs, and a wrapper status
file, including when extraction, environment activation, or Python fails.

```bash
#!/bin/bash
set -euo pipefail
cd "${_CONDOR_SCRATCH_DIR:?}"
scratch=$PWD
mkdir work
printf 'Worker started; simulation has not completed.\n' > work/job-status.txt
# A fallback exists even if later archive creation fails.
tar -czf output.tgz -C work .

finish() {
    rc=$?
    trap - EXIT
    set +e
    printf 'worker_exit_code=%s\n' "$rc" > "$scratch/work/job-status.txt"
    tar -czf "$scratch/output.tmp.tgz" -C "$scratch/work" .
    archive_rc=$?
    if (( archive_rc == 0 )); then
        mv "$scratch/output.tmp.tgz" "$scratch/output.tgz"
        archive_rc=$?
    fi
    if (( archive_rc != 0 )); then
        echo "Output archiving failed; retaining fallback archive" >&2
        if (( rc == 0 )); then rc=$archive_rc; fi
    fi
    exit "$rc"
}
trap finish EXIT

tar -xzf jobfiles.tgz -C work
tar -xzf inputs.tgz -C work
cd work
source /eos/project/e/PROJECT/environments/activate-ecloud.sh
python run_sim.py > simulation.stdout 2> simulation.stderr
```

The EXIT trap preserves a nonzero simulation exit code. If the simulation
succeeds but archiving fails, the worker returns the archive failure instead.
Archives are created outside `work` to avoid including themselves. A failure
before the fallback is created, disk failure, or forced termination can still
prevent delivery; this is not eviction recovery.

After retrieval, inspect `job-status.txt`, the logs, and expected simulation
completion. A fallback archive or partially written `output.mat` is not a
successful result. Keep the archive until validation is complete.

## Submit

```bash
mkdir -p logs
chmod +x job.sh
condor_submit -dry-run submit.ads htcondor.sub
condor_submit htcondor.sub
```

The dry run parses the description without submitting jobs; it does not check
worker environment availability or EOS permissions.

## Alternative: transfer individual files from the worker

To avoid archiving, set `transfer_output_files = ""`, remove
`output_destination`, and explicitly upload selected outputs with `xrdcp`
after Python exits. Capture Python's exit code before transferring. Enumerate
existing local files in the shell and check **every** transfer's exit code;
do not pass wildcards to `transfer_output_files`. Fail the job if a required
result is missing or its upload fails. Use a distinct EOS directory per
attempt, and retain logs for failed simulations too.
