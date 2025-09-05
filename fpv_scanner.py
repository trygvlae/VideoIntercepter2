import argparse
import math
import sys
import time
from typing import Dict, List, Tuple, Set

from video_player import VideoTransmitter
from vtx_controller import VTXController

import numpy as np

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_RX
except Exception as exc:  # pragma: no cover
    print("ERROR: Failed to import SoapySDR. Install with: pip install SoapySDR", file=sys.stderr)
    raise


# FPV band/channel plan (MHz)
FPV_BANDS_MHZ: Dict[str, List[int]] = {
    # A/"Boscam A"
    "A": [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725],
    # B/"Boscam B"
    "B": [5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866],
    # E
    "E": [5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945],
    # F/"FatShark"/"ImmersionRC"
    "F": [5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880],
    # R/"Raceband"
    "R": [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917],
}


# Default configuration (edit here to change standardverdier)
DEFAULT_GAIN_DB: float = 30.0            # Total RX gain (LNA+VGA)
DEFAULT_SAMPLE_RATE_MSPS: float = 10.0   # Sample rate in MSPS
DEFAULT_THRESHOLD_DB: float = 8.0        # Peak over median (dB)
DEFAULT_MIN_BW_MHZ: float = 6.0          # Minimum bandwidth (MHz)
DEFAULT_SAMPLES: int = 16384             # Samples per channel measurement
DEFAULT_SETTLE_MS: int = 50              # Settle time after tuning (ms)
DEFAULT_NOISE_PERCENTILE: float = 30.0   # Percentile for noise floor (e.g., 30 = robust below-median)
DEFAULT_SMOOTH_MHZ: float = 1.0          # Spectral smoothing width (MHz), 0 disables
DEFAULT_ENABLE_AMP: bool = False         # Enable HackRF RF AMP (if available)
DEFAULT_DEBUG: bool = False              # Print per-channel debug metrics
DEFAULT_LOOP_INTERVAL_S: float = 1.0     # Delay between scan passes in continuous mode
DEFAULT_CONFIRMATIONS: int = 2           # Times in a row required to trigger action
DEFAULT_PLAY_DURATION_S: float = 10.0    # Seconds to transmit/play video before resuming detection
DEFAULT_ACTIVATE: bool = True            # If True: set VTX and play to VTX; if False: local playback only
DEFAULT_RESTRICT: str = ""              # Comma-separated R-band channels to restrict, e.g., "R1,R3"
DEFAULT_RESTRICT_TOL_MHZ: float = 6.0    # Channels within this MHz of restricted freqs are also blocked

#test

def setup_hackrf(sample_rate: float, total_gain_db: float, enable_amp: bool) -> Tuple[SoapySDR.Device, object]:
    """Create and configure HackRF via SoapySDR.

    Splits total_gain_db across LNA and VGA conservatively.
    Returns device and an initialized RX stream (CF32).
    """
    dev = SoapySDR.Device({"driver": "hackrf"})

    # Select RX antenna if available
    try:
        dev.setAntenna(SOAPY_SDR_RX, 0, "RX")
    except Exception:
        pass

    # Configure sample rate and (optional) baseband filter bandwidth
    dev.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    try:
        dev.setBandwidth(SOAPY_SDR_RX, 0, sample_rate)
    except Exception:
        pass

    # RF AMP (front-end)
    try:
        dev.setGainMode(SOAPY_SDR_RX, 0, False)
    except Exception:
        pass
    try:
        dev.setGain(SOAPY_SDR_RX, 0, "AMP", 1.0 if enable_amp else 0.0)
    except Exception:
        pass

    # Split total gain across LNA and VGA within device limits
    remaining = max(0.0, float(total_gain_db))
    lna = min(remaining, 40.0)
    remaining -= lna
    vga = min(remaining, 62.0)
    try:
        dev.setGain(SOAPY_SDR_RX, 0, "LNA", lna)
    except Exception:
        pass
    try:
        dev.setGain(SOAPY_SDR_RX, 0, "VGA", vga)
    except Exception:
        pass

    # Use complex float32 stream
    rx_stream = dev.setupStream(SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
    dev.activateStream(rx_stream)
    return dev, rx_stream


def close_hackrf(dev: SoapySDR.Device, rx_stream: object) -> None:
    try:
        dev.deactivateStream(rx_stream)
    except Exception:
        pass
    try:
        dev.closeStream(rx_stream)
    except Exception:
        pass


def tune_frequency(dev: SoapySDR.Device, frequency_hz: float) -> None:
    dev.setFrequency(SOAPY_SDR_RX, 0, float(frequency_hz))


def read_samples(
    dev: SoapySDR.Device,
    rx_stream: object,
    num_samples: int,
    settle_time_s: float,
) -> np.ndarray:
    """Read complex baseband samples after tuning.

    Discards samples during settle_time_s, then returns num_samples samples as complex64.
    """
    # Discard/settle
    if settle_time_s > 0.0:
        deadline = time.time() + settle_time_s
        tmp_len = min(4096, num_samples)
        tmp = np.empty(tmp_len, dtype=np.complex64)
        while time.time() < deadline:
            _ = _read_into_buffer(dev, rx_stream, tmp)

    # Read desired block (may require multiple reads)
    out = np.empty(num_samples, dtype=np.complex64)
    filled = 0
    while filled < num_samples:
        filled += _read_into_buffer(dev, rx_stream, out[filled:])
    return out


def _read_into_buffer(dev: SoapySDR.Device, rx_stream: object, buff: np.ndarray) -> int:
    total = 0
    target = buff.shape[0]
    # Soapy expects a list of buffers
    view = buff
    while total < target:
        elems = int(min(4096, target - total))
        ret = dev.readStream(rx_stream, [view[total:total + elems]], elems)

        # Normalize return to an integer sample count across bindings
        n_read = 0
        try:
            # Newer bindings may return StreamResult object
            if hasattr(SoapySDR, "StreamResult") and isinstance(ret, SoapySDR.StreamResult):
                n_read = int(getattr(ret, "ret", 0) or 0)
            elif isinstance(ret, tuple):
                n_read = int((ret[0] if ret else 0) or 0)
            else:
                n_read = int(ret or 0)
        except Exception:
            n_read = 0

        # Handle negative error codes (timeout/overflow) as non-fatal
        if n_read is None:
            n_read = 0
        if n_read < 0:
            # Optional: back-off on timeout
            try:
                if n_read == SoapySDR.SOAPY_SDR_TIMEOUT:
                    time.sleep(0.001)
            except Exception:
                pass
            # Ignore and continue
            n_read = 0
        if n_read == 0:
            # Brief back-off to avoid busy loop
            time.sleep(0.001)
            continue
        total += int(n_read)
    return total


def compute_spectrum(samples: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum in dB and associated frequency axis (Hz)."""
    n = int(samples.shape[0])
    if n == 0:
        return np.array([]), np.array([])

    # Apply Hann window to reduce leakage
    window = np.hanning(n).astype(np.float32)
    windowed = samples * window

    # FFT and shift to center DC
    spectrum = np.fft.fft(windowed)
    spectrum = np.fft.fftshift(spectrum)
    power = np.abs(spectrum) ** 2

    # Convert to dB scale (relative); add epsilon to avoid log(0)
    eps = 1e-20
    power_db = 10.0 * np.log10(power + eps)

    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    freqs = np.fft.fftshift(freqs)
    return power_db.astype(np.float32), freqs.astype(np.float32)


def smooth_spectrum(power_db: np.ndarray, freqs_hz: np.ndarray, smooth_width_hz: float) -> np.ndarray:
    """Apply simple moving-average smoothing over a frequency window width in Hz."""
    if smooth_width_hz <= 0.0 or power_db.size == 0:
        return power_db
    # Compute kernel size in bins (~width / bin_width)
    if freqs_hz.size < 2:
        return power_db
    bin_hz = float(abs(freqs_hz[1] - freqs_hz[0]))
    if bin_hz <= 0:
        return power_db
    k = max(1, int(round(smooth_width_hz / bin_hz)))
    k = min(k, power_db.size - 1)
    if k <= 1:
        return power_db
    kernel = np.ones(k, dtype=np.float32) / float(k)
    # Pad at edges to keep length consistent
    padded = np.pad(power_db, (k // 2, k - 1 - k // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.astype(np.float32)


def detect_video(
    power_db: np.ndarray,
    freqs_hz: np.ndarray,
    median_offset_db: float,
    min_bandwidth_hz: float,
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE,
) -> Tuple[bool, float, float]:
    """Detect presence of analog video signal.

    Returns tuple: (detected, peak_db_over_median, estimated_bandwidth_hz)
    """
    if power_db.size == 0 or freqs_hz.size == 0:
        return False, 0.0, 0.0

    # Robust noise estimate using percentile (e.g., 30th percentile)
    p = float(np.clip(noise_percentile, 1.0, 49.0))
    median_db = float(np.percentile(power_db, p))
    peak_idx = int(np.argmax(power_db))
    peak_db = float(power_db[peak_idx])
    peak_over_median = peak_db - median_db

    # Condition 1: strong peak above median noise
    if peak_over_median < median_offset_db:
        return False, peak_over_median, 0.0

    # Condition 2: bandwidth of contiguous region above (median + offset) at least min_bandwidth
    threshold = median_db + median_offset_db
    above = power_db >= threshold

    # Find contiguous region around the peak where above==True
    left = peak_idx
    while left - 1 >= 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right + 1 < above.size and above[right + 1]:
        right += 1

    # Estimate bandwidth from frequency axis
    bw_hz = float(abs(freqs_hz[right] - freqs_hz[left])) if right > left else 0.0

    detected = bw_hz >= min_bandwidth_hz
    return detected, peak_over_median, bw_hz


def scan_all_channels(
    sample_rate: float,
    total_gain_db: float,
    threshold_offset_db: float,
    min_bandwidth_mhz: float,
    samples_per_measurement: int,
    settle_ms: int,
    enable_amp: bool,
    smooth_mhz: float,
    noise_percentile: float,
    debug: bool,
) -> Tuple[List[Tuple[str, int, int]], Dict[Tuple[str, int], float]]:
    """Scan all FPV bands/channels.

    Returns:
      - detections: list of (band, ch, freq_mhz)
      - strengths: dict mapping (band, ch) -> peak_over_median (dB) for this pass
    """
    dev, rx_stream = setup_hackrf(sample_rate=sample_rate, total_gain_db=total_gain_db, enable_amp=enable_amp)

    detections: List[Tuple[str, int, int]] = []
    strengths: Dict[Tuple[str, int], float] = {}
    min_bw_hz = float(min_bandwidth_mhz) * 1e6
    settle_time_s = max(0.0, settle_ms / 1000.0)

    try:
        for band_name, freqs_mhz in FPV_BANDS_MHZ.items():
            for ch_idx, f_mhz in enumerate(freqs_mhz, start=1):
                f_hz = float(f_mhz) * 1e6
                try:
                    tune_frequency(dev, f_hz)
                except Exception as exc:
                    print(f"[ERROR] Failed to tune {band_name} CH{ch_idx} ({f_mhz} MHz): {exc}")
                    continue

                samples = read_samples(
                    dev=dev,
                    rx_stream=rx_stream,
                    num_samples=samples_per_measurement,
                    settle_time_s=settle_time_s,
                )

                power_db, freqs_hz = compute_spectrum(samples, sample_rate=sample_rate)
                if smooth_mhz > 0:
                    power_db = smooth_spectrum(power_db, freqs_hz, smooth_mhz * 1e6)

                detected, peak_over_median, bw_hz = detect_video(
                    power_db=power_db,
                    freqs_hz=freqs_hz,
                    median_offset_db=threshold_offset_db,
                    min_bandwidth_hz=min_bw_hz,
                    noise_percentile=noise_percentile,
                )

                if detected:
                    print(f"[DETECTED] Band {band_name} - CH{ch_idx} ({f_mhz} MHz)  "
                          f"peak+{peak_over_median:.1f} dB, bw {bw_hz/1e6:.1f} MHz")
                    detections.append((band_name, ch_idx, int(f_mhz)))
                    strengths[(band_name, ch_idx)] = float(peak_over_median)
                else:
                    if debug:
                        print(f"[CLEAR]    Band {band_name} - CH{ch_idx} ({f_mhz} MHz)  "
                              f"peak+{peak_over_median:.1f} dB, bw {bw_hz/1e6:.1f} MHz")
                    else:
                        print(f"[CLEAR]    Band {band_name} - CH{ch_idx} ({f_mhz} MHz)")

    finally:
        close_hackrf(dev, rx_stream)

    return detections, strengths


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Scan 5.8 GHz FPV bands with HackRF and detect analog video signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN_DB, help="Total RX gain in dB (LNA+VGA)")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_MSPS, help="Sample rate in MSPS")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_DB, help="Detection threshold above median noise in dB")
    parser.add_argument("--min-bw", type=float, default=DEFAULT_MIN_BW_MHZ, help="Minimum bandwidth for detection in MHz")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Samples per channel measurement")
    parser.add_argument("--settle-ms", type=int, default=DEFAULT_SETTLE_MS, help="Settle time after tuning in milliseconds")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_ENABLE_AMP, help="Enable RF AMP (front-end) if available")
    parser.add_argument("--noise-percentile", type=float, default=DEFAULT_NOISE_PERCENTILE, help="Percentile used for noise floor estimate (1-49)")
    parser.add_argument("--smooth", type=float, default=DEFAULT_SMOOTH_MHZ, help="Smoothing width in MHz for spectrum (0 disables)")
    parser.add_argument("--debug", action="store_true", default=DEFAULT_DEBUG, help="Print per-channel metrics when CLEAR")
    parser.add_argument("--single", action="store_true", help="Run one pass and exit (otherwise continuous)")
    parser.add_argument("--loop-interval", type=float, default=DEFAULT_LOOP_INTERVAL_S, help="Delay between scan passes in seconds")
    parser.add_argument("--confirmations", type=int, default=DEFAULT_CONFIRMATIONS, help="Detections in a row required to trigger action")
    parser.add_argument("--play-duration", type=float, default=DEFAULT_PLAY_DURATION_S, help="Seconds to transmit/play video before resuming detection")
    parser.add_argument("--video", type=str, default="video.mp4", help="MP4 file to play when threshold reached (same folder)")
    parser.add_argument("--vtx", type=str, default="noop", choices=["noop", "cmd"], help="VTX control method")
    parser.add_argument("--vtx-cmd", type=str, default="", help="Command template to set frequency (use {freq_mhz})")
    parser.add_argument("--restrict", type=str, default=DEFAULT_RESTRICT, help="Comma-separated restricted R-band channels (e.g. R1,R3)")
    parser.add_argument("--restrict-tol", type=float, default=DEFAULT_RESTRICT_TOL_MHZ, help="Restrict tolerance in MHz for nearby channels")
    act_group = parser.add_mutually_exclusive_group()
    act_group.add_argument("--activate", dest="activate", action="store_true", help="Set VTX and transmit on trigger")
    act_group.add_argument("--no-activate", dest="activate", action="store_false", help="Do not set VTX; local playback only")
    parser.set_defaults(activate=DEFAULT_ACTIVATE)

    args = parser.parse_args(argv)

    sample_rate_sps = float(args.sample_rate) * 1e6

    print("=== FPV Analog Scanner (HackRF + SoapySDR) ===")
    print(f"Sample rate: {args.sample_rate:.2f} MSPS, Gain: {args.gain:.1f} dB, "
          f"Threshold: +{args.threshold:.1f} dB over median, Min BW: {args.min_bw:.1f} MHz")
    print(f"Samples/measurement: {args.samples}, Settle: {args.settle_ms} ms, AMP: {bool(args.amp)}")
    print(f"Noise Pctl: {args.noise_percentile:.0f}, Smooth: {args.smooth:.1f} MHz, Debug: {bool(args.debug)}, Activate TX: {bool(args.activate)}")
    if bool(args.activate):
        print(f"VTX mode: {args.vtx}, Command: {args.vtx_cmd or '(none)'}")
    else:
        print(f"VTX mode (dry-run): {args.vtx}, Command (not used): {args.vtx_cmd or '(none)'}")
    if not args.single:
        print("Continuous scan mode enabled")
    print("Scanning all 40 channels...\n")

    # Parse restricted channels into a set of restricted frequencies with tolerance
    restricted_freqs: Set[int] = set()
    if str(args.restrict).strip():
        for token in str(args.restrict).split(','):
            token = token.strip().upper()
            if token.startswith('R') and token[1:].isdigit():
                r_idx = int(token[1:])
                if 1 <= r_idx <= 8:
                    restricted_freqs.add(int(FPV_BANDS_MHZ['R'][r_idx - 1]))

    def is_restricted(band: str, ch: int) -> bool:
        try:
            f_mhz = int(FPV_BANDS_MHZ[band][ch - 1])
        except Exception:
            return False
        tol = float(args.restrict_tol)
        for rf in restricted_freqs:
            if abs(f_mhz - rf) <= tol:
                return True
        return False

    def run_once() -> Tuple[List[Tuple[str, int, int]], Dict[Tuple[str, int], float]]:
        return scan_all_channels(
            sample_rate=sample_rate_sps,
            total_gain_db=args.gain,
            threshold_offset_db=args.threshold,
            min_bandwidth_mhz=args.min_bw,
            samples_per_measurement=args.samples,
            settle_ms=args.settle_ms,
            enable_amp=bool(args.amp),
            smooth_mhz=float(args.smooth),
            noise_percentile=float(args.noise_percentile),
            debug=bool(args.debug),
        )

    if args.single:
        detections, strengths = run_once()
        print("\n--- RESULTAT ---")
        if detections:
            for band, ch, f_mhz in detections:
                print(f"Analog FPV video på Band {band}, Kanal {ch} ({f_mhz} MHz)")
        else:
            print("Ingen analoge FPV-signaler detektert.")
    else:
        # Track consecutive detections per (band, ch)
        consecutive: Dict[Tuple[str, int], int] = {}
        triggered: Set[Tuple[str, int]] = set()
        tx = VideoTransmitter(video_path=str(args.video))
        vtx = VTXController(method=str(args.vtx), command_template=(str(args.vtx_cmd) or None))
        try:
            while True:
                detections, strengths = run_once()
                print("\n--- RESULTAT ---")
                if detections:
                    for band, ch, f_mhz in detections:
                        print(f"Analog FPV video på Band {band}, Kanal {ch} ({f_mhz} MHz)")
                        key = (band, ch)
                        prev = consecutive.get(key, 0)
                        consecutive[key] = prev + 1
                    # Decay or reset channels not detected this pass
                    detected_keys = {(b, c) for b, c, _ in detections}
                    for key in list(consecutive.keys()):
                        if key not in detected_keys:
                            consecutive[key] = 0
                    # Check triggers
                    # Build candidates that reached confirmation threshold this pass
                    candidates: List[Tuple[str, int, float]] = []
                    for key, count in list(consecutive.items()):
                        if count >= int(args.confirmations) and key not in triggered:
                            s = strengths.get(key, 0.0)
                            if not is_restricted(key[0], key[1]):
                                candidates.append((key[0], key[1], s))
                            else:
                                if bool(args.debug):
                                    try:
                                        f_mhz = FPV_BANDS_MHZ[key[0]][key[1] - 1]
                                    except Exception:
                                        f_mhz = 0
                                    print(f"[SKIP] Restricted: Band {key[0]} CH{key[1]} ({f_mhz} MHz)")
                    if candidates:
                        # Choose the strongest by peak-over-median dB
                        band, ch, s = max(candidates, key=lambda t: t[2])
                        if bool(args.activate):
                            print(f"[TRIGGER] Strongest: Band {band} CH{ch} (peak+{s:.1f} dB) -> set VTX + play {args.video} for {args.play_duration:.1f}s")
                            vtx.set_band_channel(band, ch)
                        else:
                            try:
                                freq_mhz = FPV_BANDS_MHZ[band][ch - 1]
                            except Exception:
                                freq_mhz = 0
                            print(
                                f"[TRIGGER] Strongest: Band {band} CH{ch} (peak+{s:.1f} dB) -> local play {args.video} for {args.play_duration:.1f}s (no TX). "
                                f"Would TX on Band {band} CH{ch} at {freq_mhz} MHz"
                            )
                        tx.play_for(float(args.play_duration))
                        consecutive.clear()
                        triggered.clear()
                else:
                    print("Ingen analoge FPV-signaler detektert.")
                    # Reset all counts when nothing is detected
                    consecutive.clear()
                time.sleep(max(0.0, float(args.loop_interval)))
        except KeyboardInterrupt:
            print("\nAvslutter kontinuerlig skanning...")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))



