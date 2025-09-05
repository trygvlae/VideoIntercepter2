import os
import shlex
import subprocess
from typing import Dict, List, Optional

"""
Raspberry Pi wiring notes (for VTX control and video out):

- Composite video to VTX:
  The analog video signal comes from the 3.5 mm A/V jack on the Raspberry Pi.
  Use the official TRRS pinout for your board revision (tip is usually video).
  Connect the VTX video-in to the Pi's composite video output on the jack.
  Ensure common ground between Pi and VTX.

- SmartAudio/one-wire UART control to VTX:
  Recommended GPIO: BCM GPIO14 (TXD0, physical pin 8) as the transmit line.
  This allows using the primary UART for SmartAudio-style control.
  Voltage level: Pi GPIO is 3.3 V. Many VTX units accept 3.3 V logic, but
  consult the Alpha 16 specs. If 5 V is required, use a level shifter.
  Also connect ground between Pi and VTX.

  If you cannot use the primary UART, you can configure another UART overlay
  (e.g., mini UART) and map its TX pin, or implement bit-banged one-wire on
  a different GPIO via a userspace tool. Update the command template accordingly.

Power: Do not power the VTX from a Pi 5 V pin unless within current limits;
prefer a dedicated, filtered regulator per VTX requirements.
"""


# FPV band/channel plan (MHz) duplicated to avoid circular imports
FPV_BANDS_MHZ: Dict[str, List[int]] = {
    "A": [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725],
    "B": [5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866],
    "E": [5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945],
    "F": [5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880],
    "R": [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917],
}


class VTXController:
    """Pluggable VTX controller.

    method:
      - "noop": only prints actions (default)
      - "cmd":  executes a shell command template to set frequency

    For method "cmd", provide a command template with {freq_mhz}, e.g.:
      "python3 ./set_vtx.py --freq {freq_mhz}"
    """

    def __init__(self, method: str = "noop", command_template: Optional[str] = None) -> None:
        self.method = method.strip().lower() if method else "noop"
        if self.method not in ("noop", "cmd"):
            self.method = "noop"
        self.command_template = command_template or os.environ.get("VTX_SET_CMD", "")

    def set_band_channel(self, band: str, channel: int) -> None:
        band = str(band).upper()
        ch_idx = int(channel)
        if band not in FPV_BANDS_MHZ or not (1 <= ch_idx <= 8):
            print(f"[VTX] Invalid band/channel: {band} CH{channel}")
            return
        freq_mhz = FPV_BANDS_MHZ[band][ch_idx - 1]
        # If using command mode and template provided, allow band/channel placeholders
        if self.method == "cmd" and self.command_template:
            cmd = self.command_template.format(freq_mhz=int(freq_mhz), band=band, channel=ch_idx)
            self._run_cmd(cmd)
            return
        self.set_frequency_mhz(freq_mhz)
        # For clarity in dry-run vs active modes, print the intended action with frequency
        if self.method == "noop":
            print(f"[VTX] Would set {band} CH{ch_idx} -> {freq_mhz} MHz")

    def set_frequency_mhz(self, freq_mhz: int) -> None:
        if self.method == "noop":
            print(f"[VTX] (noop) Set frequency to {freq_mhz} MHz")
            return

        if self.method == "cmd":
            if not self.command_template:
                print("[VTX] cmd method selected but no command template provided (set --vtx-cmd or VTX_SET_CMD)")
                return
            cmd = self.command_template.format(freq_mhz=int(freq_mhz))
            self._run_cmd(cmd)
            return

        # Fallback
        print(f"[VTX] Unknown method {self.method}, noop: set {freq_mhz} MHz")

    def _run_cmd(self, cmd: str) -> None:
        try:
            args = shlex.split(cmd)
            subprocess.run(args, check=False)
        except Exception as exc:
            print(f"[VTX] Failed to execute command: {cmd} -> {exc}")


