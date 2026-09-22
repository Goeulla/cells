"""
plot_height_velocity_profile.py -- visualize the theoretical parabolic (Poiseuille)
velocity profile between parallel plates that cell_speed_binning_exit_v7_clean.py's
--shear_stress_pa/--chamber_height_um/--medium_viscosity_pa_s back-calculation
inverts (Oh et al. 2015, J Cell Sci 128:3731-3743, Eqn 3):

    vx = tau_wall/(h*mu) * (h^2/4 - y^2)

where y is measured from the channel centerline. Plotted here directly in terms
of height above the bottom substrate (0 at the bottom wall, h at the top), so it
lines up with est_height_above_bottom_um in per_object_csv -- if you pass that
CSV in, each counted cell's own (speed, est_height_above_bottom_um) is overlaid
on the same axes as a direct sanity check: a point should sit close to the curve,
since that's literally the relationship its height was solved from. A point that
sits far off the curve either had its speed capped/clipped by something else in
the pipeline, or is evidence the flow isn't behaving like simple parabolic shear
at that point (e.g. still within an entrance/settling region).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
AMBER = "#c97a1a"


def velocity_profile(height_above_bottom_um, chamber_height_um, shear_stress_pa, viscosity_pa_s):
    h_m = chamber_height_um * 1e-6
    y_from_center_m = height_above_bottom_um * 1e-6 - h_m / 2.0
    vx_m_s = shear_stress_pa / (h_m * viscosity_pa_s) * (h_m**2 / 4.0 - y_from_center_m**2)
    return vx_m_s * 1e6  # um/s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shear_stress_pa", type=float, required=True)
    ap.add_argument("--chamber_height_um", type=float, required=True)
    ap.add_argument("--medium_viscosity_pa_s", type=float, required=True)
    ap.add_argument("--per_object_csv", type=str, default=None,
                    help="Optional: overlay real counted cells' (speed, "
                         "est_height_above_bottom_um) from a per_object_csv "
                         "produced with the same three physical parameters.")
    ap.add_argument("--out_png", type=str, default="height_velocity_profile.png")
    args = ap.parse_args()

    h = args.chamber_height_um
    height_um = np.linspace(0, h, 400)
    vx_um_s = velocity_profile(height_um, h, args.shear_stress_pa, args.medium_viscosity_pa_s)
    vx_max = vx_um_s.max()

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    ax.plot(vx_um_s, height_um, color=BLUE, linewidth=2, zorder=3,
            label="Theoretical profile (parabolic flow)")

    ax.axhline(0, color="#9a9890", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(h, color="#9a9890", linewidth=1, linestyle="--", zorder=1)
    ax.text(vx_max * 1.02, 2, "bottom wall", fontsize=9, color="#6b6a63", va="bottom")
    ax.text(vx_max * 1.02, h - 2, "top wall", fontsize=9, color="#6b6a63", va="top")
    ax.axhline(h / 2, color="#c7c5bd", linewidth=1, linestyle=":", zorder=1)

    if args.per_object_csv:
        df = pd.read_csv(args.per_object_csv)
        if "est_height_above_bottom_um" in df.columns:
            d = df.dropna(subset=["est_height_above_bottom_um"]).copy()
            speed_um_s = d["speed"] * 1e6 if (d["speed"].abs() < 1).mean() > 0.5 else d["speed"]
            ax.scatter(speed_um_s, d["est_height_above_bottom_um"],
                       s=28, color=AMBER, alpha=0.75, zorder=4,
                       edgecolors="white", linewidths=0.6,
                       label=f"Counted cells (n={len(d)})")
        else:
            print("Warning: no est_height_above_bottom_um column in that CSV -- "
                  "was it generated with --shear_stress_pa/--chamber_height_um/"
                  "--medium_viscosity_pa_s/--m_per_px set?")

    ax.set_xlabel("x-velocity (µm/s)")
    ax.set_ylabel("Height above bottom substrate (µm)")
    ax.set_title("Velocity vs. height across the channel gap\n"
                 f"h={h:.0f} µm, τ={args.shear_stress_pa:g} Pa, "
                 f"μ={args.medium_viscosity_pa_s:g} Pa·s", fontsize=12)
    ax.set_ylim(-h * 0.05, h * 1.05)
    ax.set_xlim(0, vx_max * 1.15)
    ax.grid(True, color="#e8e6e0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(args.out_png)
    print(f"Wrote: {args.out_png}  (peak mid-channel velocity: {vx_max:.2f} µm/s at h/2={h/2:.1f} µm)")


if __name__ == "__main__":
    main()
