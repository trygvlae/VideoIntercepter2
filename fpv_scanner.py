import argparse
import math
import sys
import time
from typing import Dict, List, Tuple

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


def setup_hackrf(sample_rate: float, total_gain_db: float) -> Tuple[SoapySDR.Device, object]:
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
        if isinstance(ret, tuple):
            n_read, _flags, _time = ret
        else:
            n_read = ret
        if n_read is None:
            n_read = 0
        if n_read < 0:
            # Timeout or overflow; treat as zero read and continue
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


def detect_video(
    power_db: np.ndarray,
    freqs_hz: np.ndarray,
    median_offset_db: float,
    min_bandwidth_hz: float,
) -> Tuple[bool, float, float]:
    """Detect presence of analog video signal.

    Returns tuple: (detected, peak_db_over_median, estimated_bandwidth_hz)
    """
    if power_db.size == 0 or freqs_hz.size == 0:
        return False, 0.0, 0.0

    median_db = float(np.median(power_db))
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
) -> List[Tuple[str, int, int]]:
    """Scan all FPV bands/channels. Returns list of detections as (band, ch, freq_mhz)."""
    dev, rx_stream = setup_hackrf(sample_rate=sample_rate, total_gain_db=total_gain_db)

    detections: List[Tuple[str, int, int]] = []
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
                detected, peak_over_median, bw_hz = detect_video(
                    power_db=power_db,
                    freqs_hz=freqs_hz,
                    median_offset_db=threshold_offset_db,
                    min_bandwidth_hz=min_bw_hz,
                )

                if detected:
                    print(f"[DETECTED] Band {band_name} - CH{ch_idx} ({f_mhz} MHz)  "+
                          f"peak+{peak_over_median:.1f} dB, bw {bw_hz/1e6:.1f} MHz")
                    detections.append((band_name, ch_idx, int(f_mhz)))
                else:
                    print(f"[CLEAR]    Band {band_name} - CH{ch_idx} ({f_mhz} MHz)")

    finally:
        close_hackrf(dev, rx_stream)

    return detections


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Scan 5.8 GHz FPV bands with HackRF and detect analog video signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gain", type=float, default=30.0, help="Total RX gain in dB (LNA+VGA)")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Sample rate in MSPS")
    parser.add_argument("--threshold", type=float, default=8.0, help="Detection threshold above median noise in dB")
    parser.add_argument("--min-bw", type=float, default=6.0, help="Minimum bandwidth for detection in MHz")
    parser.add_argument("--samples", type=int, default=16384, help="Samples per channel measurement")
    parser.add_argument("--settle-ms", type=int, default=50, help="Settle time after tuning in milliseconds")

    args = parser.parse_args(argv)

    sample_rate_sps = float(args.sample_rate) * 1e6

    print("=== FPV Analog Scanner (HackRF + SoapySDR) ===")
    print(f"Sample rate: {args.sample_rate:.2f} MSPS, Gain: {args.gain:.1f} dB, "
          f"Threshold: +{args.threshold:.1f} dB over median, Min BW: {args.min_bw:.1f} MHz")
    print(f"Samples/measurement: {args.samples}, Settle: {args.settle_ms} ms")
    print("Scanning all 40 channels...\n")

    detections = scan_all_channels(
        sample_rate=sample_rate_sps,
        total_gain_db=args.gain,
        threshold_offset_db=args.threshold,
        min_bandwidth_mhz=args.min_bw,
        samples_per_measurement=args.samples,
        settle_ms=args.settle_ms,
    )

    print("\n--- RESULTAT ---")
    if detections:
        for band, ch, f_mhz in detections:
            print(f"Analog FPV video på Band {band}, Kanal {ch} ({f_mhz} MHz)")
    else:
        print("Ingen analoge FPV-signaler detektert.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))



