#!/usr/bin/env python
"""
Week 14: QuakeMigrate Locate stage - refined sub-grid event location, phase
re-picking, and local magnitude estimation, for events identified in the
Trigger stage (threshold 1.75).

Built directly against QuakeMigrate's own confirmed real template
(examples/template_scripts/locate.py).

Magnitude estimation uses a CUSTOM attenuation function, not QuakeMigrate's
built-in "Hutton-Boore" option - deliberately, to keep magnitudes on the
same real scale used for the EQTransformer and PhaseNet catalogues (the
Taupo-specific formula from Illsley-Kemp et al. 2021), enabling a genuine
apples-to-apples Mc/b-value comparison in Phase 4. Confirmed directly from
QuakeMigrate's source (signal/local_mag/magnitude.py) that a custom A0
function is called as att = A0(dist), with dist in km, and that
mags = log10(amps) + att + station_corrections - the same structural form
we already used ourselves.

One real, documented adjustment: QuakeMigrate's own docstring specifies
`amps` as *half* peak-to-trough amplitude, whereas our EQTransformer/
PhaseNet pipeline used the *full* peak-to-trough amplitude (matching the
literal wording of Illsley-Kemp et al. 2021). Left uncorrected this would
introduce a systematic ~log10(2) = 0.301 magnitude offset relative to the
other two catalogues. Corrected by adding log10(2) to the attenuation
function's return value - exact under the standard assumption that full
peak-to-trough ~= 2x half peak-to-trough for a roughly symmetric waveform,
not a mathematical guarantee for every real seismogram, but a reasonable,
documented correction rather than an unflagged discrepancy.

Vertical-component amplitude (amp_feature/trace_filter) and hypocentral
distance (use_hyp_dist=True) are both set explicitly to match the
established EQTransformer/PhaseNet magnitude convention.

No per-station correction term (station_corrections={}), matching the same
documented simplification used in Weeks 7 and 11 (no calibration table was
available from the published source).
"""
import os
import argparse
import numpy as np

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

from obspy.core import AttribDict

from quakemigrate import QuakeScan
from quakemigrate.io import Archive, read_lut, read_response_inv, read_stations
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker
from quakemigrate.signal.local_mag import LocalMag

PROJECT_DIR = os.path.expanduser("~/tvz_project")
QM_DIR = os.path.join(PROJECT_DIR, "quakemigrate")
QM_ARCHIVE_DIR = os.path.join(PROJECT_DIR, "data", "waveforms_qm")


def taupo_logA0(dist):
    """Illsley-Kemp et al. (2021) Taupo-region attenuation relationship,
    the same real formula used for the EQTransformer and PhaseNet
    catalogues (Weeks 7, 11). +log10(2) compensates for QuakeMigrate's
    half- vs our full-peak-to-trough amplitude convention - see module
    docstring."""
    return 1.5337 * np.log10(dist / 17.0) + 0.002343 * (dist - 17.0) + 2.0 + np.log10(2)


def main(starttime, endtime, run_name, threads):
    station_file = os.path.join(QM_DIR, "lut", "stations.csv")
    lut_file = os.path.join(QM_DIR, "lut", "tvz_lut")
    response_file = os.path.join(PROJECT_DIR, "data", "inventory",
                                  "NZ_2019-08-01_2019-09-30.xml")
    run_path = os.path.join(QM_DIR, "runs")

    stations = read_stations(station_file, delimiter=",")
    response_inv = read_response_inv(response_file)

    response_params = AttribDict()
    response_params.pre_filt = (0.05, 0.06, 30, 35)
    response_params.water_level = 600
    response_params.remove_full_response = False

    archive = Archive(
        archive_path=QM_ARCHIVE_DIR, stations=stations,
        archive_format="STATION.YEAR.JULIANDAY",
        response_inv=response_inv,
        response_removal_params=response_params,
    )

    lut = read_lut(lut_file=lut_file)  # full resolution - unlike Detect, not decimated

    onset = STALTAOnset(position="centred", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 16, 2]}
    onset.sta_lta_windows = {"P": [0.2, 1.0], "S": [0.2, 1.0]}

    picker = GaussianPicker(onset=onset)
    picker.plot_picks = True

    amp_params = AttribDict()
    amp_params.signal_window = 5.0
    amp_params.bandpass_filter = True
    amp_params.bandpass_lowcut = 2.0
    amp_params.bandpass_highcut = 20.0

    mag_params = AttribDict()
    mag_params.A0 = taupo_logA0            # custom function, not "Hutton-Boore"
    mag_params.use_hyp_dist = True         # hypocentral, matching EQT/PhaseNet
    mag_params.amp_feature = "S_amp"
    mag_params.station_corrections = {}    # documented simplification, matching Weeks 7/11
    mag_params.trace_filter = ".[BH]HZ$"   # vertical component only, matching Illsley-Kemp et al. (2021)
    mag_params.noise_filter = 3.0
    mag_params.dist_filter = False

    mags = LocalMag(amp_params=amp_params, mag_params=mag_params)
    mags.plot_amplitudes = True

    scan = QuakeScan(
        archive, lut, onset=onset, picker=picker, mags=mags,
        run_path=run_path, run_name=run_name, log=True, loglevel="info",
    )

    scan.marginal_window = 1.0
    scan.threads = threads
    scan.plot_event_summary = True
    scan.write_cut_waveforms = True
    scan.pre_cut = 20.0
    scan.post_cut = 60.0

    print(f"Running locate: {starttime} to {endtime}, run_name={run_name}, "
          f"threads={threads}", flush=True)
    scan.locate(starttime=starttime, endtime=endtime)
    print("Locate complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--starttime", default="2019-08-01T00:00:00.0")
    parser.add_argument("--endtime", default="2019-09-30T00:00:00.0")
    parser.add_argument("--run-name", default="tvz_full")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    main(starttime=args.starttime, endtime=args.endtime, run_name=args.run_name,
         threads=args.threads)
