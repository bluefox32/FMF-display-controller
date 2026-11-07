"""Re-verification of fmf_display_controller_fixed.py: regression
checks (unchanged behaviour) + fix-specific assertions."""
import sys
import numpy as np
sys.path.insert(0, "/home/hatch/workspace/fmf_verify")
import fmf_display_controller_fixed as fmf

ok_all = []
def check(name, ok, detail=""):
    ok_all.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

hw = fmf.HardwareConfig()
layers = [fmf.LayerConfig(0, 490), fmf.LayerConfig(1, 530), fmf.LayerConfig(2, 625)]
opt = fmf.OpticalEngine(layers)
ce = fmf.ColorSpaceEngine(hw, opt)

# regression: CIE xy
ref = {(255,255,255): (0.3127, 0.3290), (255,0,0): (0.64, 0.33),
       (0,255,0): (0.30, 0.60), (0,0,255): (0.15, 0.06)}
worst = max(max(abs(ce.rgb_to_cie_xy(k)[0]-v[0]), abs(ce.rgb_to_cie_xy(k)[1]-v[1]))
            for k, v in ref.items())
check("cie_xy_regression", worst < 0.002, f"worst {worst:.4f}")

# FIX1: LUT-hit colour no longer swaps R/B.
# Quench domain: strong channel -> LOW voltage on that layer.
v = ce.rgb_to_voltages((224, 32, 32))   # red-dominant, on LUT grid
check("fix1_lut_red_dominant", v[2] < v[0] and v[0] == v[1],
      f"[B,G,R] voltages = {list(v)} (red layer lowest = least quenched)")

# FIX2: composed output monotonic over the full range, max at 255
amps = {}
for val in [0, 32, 64, 128, 192, 224, 255]:
    vv = ce.rgb_to_voltages((0, 0, val))              # blue request
    px = np.array([val, 0, 0], dtype=np.uint8)        # layer order [B,G,R]
    amps[val] = float(opt.calculate_output_spectrum(px, vv).max())
mono = all(amps[b] > amps[a] for a, b in zip(sorted(amps), sorted(amps)[1:]))
check("fix2_output_monotonic", mono, f"{ {k: round(x,3) for k,x in amps.items()} }")

# FIX3: currents span the range, clip only at full scale
c = ce.rgb_to_currents((128, 0, 0))
check("fix3_current_midscale", abs(c[2] - 128/255*5) < 0.05 and c[0] == 0,
      f"(128,0,0) -> {np.round(c,2)} A in [B,G,R]")
clip = tot = 0
for r in range(0, 256, 17):
    for g in range(0, 256, 17):
        for b in range(0, 256, 17):
            cc = ce.rgb_to_currents((r, g, b)); tot += 1
            if np.any(cc >= hw.current_max_A - 1e-9) and max(r, g, b) < 255: clip += 1
check("fix3_no_interior_clip", clip == 0, f"interior clips {clip}/{tot}")

# FIX4: protocol layout, quantisation, full-size frame
class FakeSerial:
    def __init__(self): self.buf = b""
    def write(self, data): self.buf += bytes(data); return len(data)
drv = fmf.RealDisplayDriver(hw, fmf.DriveMode.ELECTRIC_FIELD)
drv.is_connected = True; drv.serial_conn = FakeSerial()
drv.write_frame(np.array([[[1000, 500, 0]]], dtype=np.uint16))
buf = drv.serial_conn.buf
ln = int.from_bytes(buf[2:6], 'little')
crc_wire = int.from_bytes(buf[6+ln:6+ln+2], 'little')
check("fix4_layout", buf[:2] == b'\xaa\x55' and ln == 3 and buf[6] == 255
      and buf[7] == 128 and crc_wire == (sum(buf[6:6+ln]) & 0xFFFF),
      f"len={ln} data={buf[6:9].hex()} crc_ok={crc_wire == (sum(buf[6:9]) & 0xFFFF)}")
drv.serial_conn = FakeSerial()
big = np.zeros((1080, 1920, 3), dtype=np.uint16)
sent = drv.write_frame(big)
ln = int.from_bytes(drv.serial_conn.buf[2:6], 'little')
check("fix4_full_frame", sent and ln == big.size, f"LEN={ln} for {big.size} values")

print(f"\n== {sum(ok_all)}/{len(ok_all)} passed ==")
