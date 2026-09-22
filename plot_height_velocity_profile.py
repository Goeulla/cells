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

--particle_radius_um additionally overlays where the NAIVE curve above is known
to be wrong: that curve treats the cell as a massless point tracer in the
UNDISTURBED flow, which ignores hydrodynamic wall drag -- a real, finite-sized
particle within about 2 radii of the wall (y < 2a) moves measurably slower than
the undisturbed local fluid velocity there (Goldman, Cox & Brenner 1967). We only
have a rigorously defensible correction at one specific point, not a continuous
function across the whole band: a sphere actually in contact with the wall
(y=a) translates at approximately V ~= 0.7 * shear_rate_wall * a (the same
relation used for predicted_wall_velocity_m_s in the main script). The plot
shades the y<2a region to flag it as unreliable for the naive curve, and marks
the two competing predictions at y=a side by side -- it deliberately does NOT
draw a continuous "corrected" curve through that band, since we don't have a
verified closed-form for the correction factor at intermediate 0<y<2a
(that requires Goldman-Cox-Brenner's full tabulated/numerical results, not just
the contact-point approximation).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
AMBER = "#c97a1a"
RED = "#c0392b"


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
    ap.add_argument("--particle_radius_um", type=float, default=None,
                    help="Cell/particle radius in micrometers. If set, shades the "
                         "near-wall region (y < 2*radius) where the naive curve "
                         "ignores hydrodynamic wall drag, and marks the naive vs. "
                         "wall-drag-corrected velocity prediction at y=radius "
                         "(a sphere in contact with the wall) side by side. If not "
                         "set but --per_object_csv has a mean_r column, its mean "
                         "is used automatically.")
    ap.add_argument("--out_png", type=str, default="height_velocity_profile.png")
    args = ap.parse_args()

    h = args.chamber_height_um
    height_um = np.linspace(0, h, 400)
    vx_um_s = velocity_profile(height_um, h, args.shear_stress_pa, args.medium_viscosity_pa_s)
    vx_max = vx_um_s.max()

    df = None
    if args.per_object_csv:
        df = pd.read_csv(args.per_object_csv)

    radius_um = args.particle_radius_um
    if radius_um is None and df is not None and "mean_r" in df.columns:
        radius_um = df["mean_r"].mean() * 1e6  # mean_r is in meters when present
        print(f"Using mean_r from --per_object_csv as particle radius: {radius_um:.3f} µm")

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)

    if radius_um is not None:
        near_wall_um = min(2 * radius_um, h)
        ax.axhspan(0, near_wall_um, color="#f4ded6", zorder=0,
                   label=f"Near-wall zone y<2a (a={radius_um:.2f} µm) -- naive curve unreliable here")

    ax.plot(vx_um_s, height_um, color=BLUE, linewidth=2, zorder=3,
            label="Naive profile (undisturbed flow, no wall drag)")

    ax.axhline(0, color="#9a9890", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(h, color="#9a9890", linewidth=1, linestyle="--", zorder=1)
    ax.text(vx_max * 1.02, 2, "bottom wall", fontsize=9, color="#6b6a63", va="bottom")
    ax.text(vx_max * 1.02, h - 2, "top wall", fontsize=9, color="#6b6a63", va="top")
    ax.axhline(h / 2, color="#c7c5bd", linewidth=1, linestyle=":", zorder=1)

    if radius_um is not None:
        shear_rate_wall = args.shear_stress_pa / args.medium_viscosity_pa_s  # 1/s
        v_corrected_um_s = 0.7 * shear_rate_wall * (radius_um * 1e-6) * 1e6
        v_naive_at_a = velocity_profile(np.array([radius_um]), h, args.shear_stress_pa,
                                        args.medium_viscosity_pa_s)[0]
        ax.plot([v_naive_at_a, v_corrected_um_s], [radius_um, radius_um],
                color=RED, linewidth=1, linestyle="-", zorder=4, alpha=0.6)
        ax.scatter([v_naive_at_a], [radius_um], s=60, color=BLUE, zorder=5,
                   edgecolors="white", linewidths=1, marker="o",
                   label=f"Naive prediction at y=a: {v_naive_at_a:.1f} µm/s")
        ax.scatter([v_corrected_um_s], [radius_um], s=70, color=RED, zorder=5,
                   edgecolors="white", linewidths=1, marker="D",
                   label=f"Wall-drag-corrected at y=a (contact, 0.7×γ̇×a): "
                         f"{v_corrected_um_s:.1f} µm/s")

    if df is not None:
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
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
             fontsize=9, ncol=1)
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"Wrote: {args.out_png}  (peak mid-channel velocity: {vx_max:.2f} µm/s at h/2={h/2:.1f} µm)")


if __name__ == "__main__":
    main()
