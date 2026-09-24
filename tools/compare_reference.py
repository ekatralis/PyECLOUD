"""Run upstream fixtures and report differences; no unverified physics tolerance.

Run with the installed package, not the source checkout on PYTHONPATH.
Omit --duration to run the full reference duration. Reports are diagnostic:
reference files do not record RNG seeds or define numerical acceptance limits.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat
from PyECLOUD.buildup_simulation import BuildupSimulation

CASES = {
    'drift': 'LHC_Drift_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns',
    'dipole': 'LHC_ArcDipReal_450GeV_sey1.70_2.5e11ppb_bl_1.00ns',
    'boris_fortran': 'LHC_Solenoid_sey1.10_100.00mT',
    'boris_cython': 'LHC_Solenoid_sey1.10_100.00mT_BorisCython',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float)
    args = parser.parse_args()
    source = args.fixtures.resolve() / CASES[args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = output / 'input'
    shutil.copytree(source, folder, dirs_exist_ok=True)
    reference = loadmat(folder / 'Pyecltest_angle3D_ref.mat', squeeze_me=True)
    previous = Path.cwd()
    try:
        os.chdir(output)
        np.random.seed(1729)
        with (output / 'simulation.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            sim = BuildupSimulation(pyecl_input_folder=str(folder), extract_sey=False,
                secondary_angle_distribution='cosine_3D', photoelectron_angle_distribution='cosine_3D',
                filen_main_outp=str(output / 'simulation.mat'))
            sim.run(t_end_sim=args.duration)
        saver = sim.cloud_list[0].pyeclsaver
        times = saver.t[:saver.i_last_save + 1]
        metrics = {}
        for key in ['Nel_timep', 'cen_density', 'En_kin_eV_time', 'Nel_imp_time', 'Nel_emit_time']:
            values = np.asarray(getattr(saver, key))[:len(times)]
            expected = np.interp(times, reference['t'], reference[key])
            if not np.isfinite(values).all():
                raise RuntimeError(f'{key} contains non-finite values')
            scale = float(np.linalg.norm(expected))
            metrics[key] = {'relative_l2_difference': float(np.linalg.norm(values - expected) / scale) if scale else None,
                            'maximum_absolute_difference': float(np.max(np.abs(values - expected)))}
        report = {'case': args.case, 'simulated_until': float(sim.beamtim.tt_curr),
                  'samples': len(times), 'seed': 1729, 'metrics': metrics,
                  'acceptance': 'Requires scientific review; upstream provides plots, no tolerances or reference RNG seed.'}
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    main()
