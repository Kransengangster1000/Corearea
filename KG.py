import io
import json
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path


# =====================================================================
# GEOMETRISKA HJÄLPFUNKTIONER (Sutherland-Hodgman & Polygonintegration)
# =====================================================================
def clip_polygon_halfplane(pts, a, b, c):
    # Klipper en 2D-polygon (pts: Nx2) så att endast den del där a + b*y + c*z <= 0 behålls.
    if len(pts) == 0:
        return np.empty((0, 2))

    def inside(p):
        return (a + b * p[0] + c * p[1]) <= 1e-12

    def intersect(p1, p2):
        v1 = a + b * p1[0] + c * p1[1]
        v2 = a + b * p2[0] + c * p2[1]
        denom = v1 - v2
        if abs(denom) < 1e-15:
            return p1
        t = v1 / denom
        return p1 + t * (p2 - p1)

    out = []
    p_prev = pts[-1]
    in_prev = inside(p_prev)

    for p_curr in pts:
        in_curr = inside(p_curr)
        if in_curr:
            if not in_prev:
                out.append(intersect(p_prev, p_curr))
            out.append(p_curr)
        elif in_prev:
            out.append(intersect(p_prev, p_curr))
        p_prev = p_curr
        in_prev = in_curr

    return np.array(out) if len(out) >= 3 else np.empty((0, 2))


def polygon_integrals(pts):
    # Exakta ytegenskaper via Greens sats för en godtycklig 2D-polygon (moturs orienterad).
    if len(pts) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    y, z = pts[:, 0], pts[:, 1]
    yn, zn = np.roll(y, -1), np.roll(z, -1)
    cr = y * zn - yn * z

    A = 0.5 * np.sum(cr)
    if abs(A) < 1e-14:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    Sy = np.sum((y + yn) * cr) / 6.0
    Sz = np.sum((z + zn) * cr) / 6.0
    Izz = np.sum((y**2 + y * yn + yn**2) * cr) / 12.0
    Iyy = np.sum((z**2 + z * zn + zn**2) * cr) / 12.0
    Iyz = np.sum((y * zn + 2 * y * z + 2 * yn * zn + yn * z) * cr) / 24.0
    return A, Sy, Sz, Izz, Iyy, Iyz


# =====================================================================
# HUVUDKLASS FÖR TVÄRSNITTS- OCH SPÄNNINGSANALYS
# =====================================================================
class CrossSectionNavier:
    def __init__(self, vertices):
        pts = np.asarray(vertices, dtype=float)
        if len(pts) < 3:
            raise ValueError("Tvärsnittet måste ha minst 3 hörn.")
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            raise ValueError("Tvärsnittet måste ha minst 3 unika hörn.")

        if self._signed_area(pts) < 0:
            pts = pts[::-1]

        self.pts_orig = pts
        self._compute_section_properties()
        self._compute_core_boundaries()

    @staticmethod
    def _signed_area(pts):
        y, z = pts[:, 0], pts[:, 1]
        return 0.5 * np.sum(y * np.roll(z, -1) - np.roll(y, -1) * z)

    def _compute_section_properties(self):
        A, Sy, Sz, _, _, _ = polygon_integrals(self.pts_orig)
        if abs(A) < 1e-12:
            raise ValueError("Tvärsnittets area är noll (punkterna ligger på en linje).")

        self.A = A
        self.y_tp = Sy / A
        self.z_tp = Sz / A

        # Koordinater i tyngdpunktssystemet (y, z)
        self.pts = self.pts_orig - np.array([self.y_tp, self.z_tp])
        _, _, _, self.I_zz, self.I_yy, self.I_yz = polygon_integrals(self.pts)

        self.I_mat = np.array([
            [self.I_zz, self.I_yz],
            [self.I_yz, self.I_yy]
        ])

        # Huvudtröghetsmoment och vinkel
        avg = 0.5 * (self.I_yy + self.I_zz)
        diff = 0.5 * (self.I_yy - self.I_zz)
        R = np.hypot(diff, self.I_yz)
        self.I_1 = avg + R
        self.I_2 = avg - R
        self.alpha_rad = 0.5 * np.arctan2(-2.0 * self.I_yz, self.I_yy - self.I_zz)
        self.alpha_deg = np.degrees(self.alpha_rad)

    def _compute_core_boundaries(self):
        # 1. Konvext hölje och ordinarie kärngräns (100 % kontakt)
        hull = ConvexHull(self.pts)
        hull_pts = self.pts[hull.vertices]

        core_vertices = []
        n = len(hull_pts)
        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            M = np.vstack([p1, p2])
            rhs = np.array([-1.0, -1.0])
            a_vec = np.linalg.solve(M, rhs)
            e_vec = -(1.0 / self.A) * (self.I_mat @ a_vec)
            core_vertices.append(e_vec)

        self.hull_pts = hull_pts
        self.core_pts = np.array(core_vertices)

        # 2. Yttre gräns för sprucket tvärsnitt där neutrala lagret skär TP (ca 50 % aktiv area)
        angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        half_core = []
        for th in angles:
            ny, nz = np.cos(th), np.sin(th)
            clipped = clip_polygon_halfplane(self.pts, 0.0, ny, nz)
            if len(clipped) >= 3:
                Ac, Syc, Szc, Izzc, Iyyc, Iyzc = polygon_integrals(clipped)
                denom = ny * Syc + nz * Szc
                if abs(denom) > 1e-12:
                    ey = (ny * Izzc + nz * Iyzc) / denom
                    ez = (ny * Iyzc + nz * Iyyc) / denom
                    half_core.append([ey, ez])
        self.half_core_pts = np.array(half_core) if half_core else self.core_pts

    def calculate_linear(self, N, M_y, M_z):
        sigma_0 = N / self.A
        c_vec = np.linalg.solve(self.I_mat, np.array([M_z, M_y]))
        c_y, c_z = c_vec[0], c_vec[1]
        v_stress = sigma_0 + c_y * self.pts[:, 0] + c_z * self.pts[:, 1]
        s_max, s_min = float(np.max(v_stress)), float(np.min(v_stress))

        if s_max <= 1e-9:
            active_ratio = 100.0
            clipped = self.pts
        elif s_min >= -1e-9:
            active_ratio = 0.0
            clipped = np.empty((0, 2))
        else:
            clipped = clip_polygon_halfplane(self.pts, sigma_0, c_y, c_z)
            Ac, _, _, _, _, _ = polygon_integrals(clipped)
            active_ratio = 100.0 * (Ac / self.A)

        return {
            "mode": "Linjär (Navier)",
            "converged": True,
            "sigma_0": sigma_0,
            "c_y": c_y,
            "c_z": c_z,
            "v_stress": v_stress,
            "s_max": s_max,
            "s_min": s_min,
            "active_area_pct": active_ratio,
            "compressed_poly": clipped,
            "message": "Linjär elastisk beräkning enligt Naviers formel."
        }

    def calculate_cracked(self, N, M_y, M_z):
        lin = self.calculate_linear(N, M_y, M_z)
        if N >= -1e-9:
            return {
                "mode": "Sprucket (Kräver N < 0)",
                "converged": False,
                "sigma_0": np.nan,
                "c_y": np.nan,
                "c_z": np.nan,
                "v_stress": np.full(len(self.pts), np.nan),
                "s_max": np.nan,
                "s_min": np.nan,
                "active_area_pct": 0.0,
                "compressed_poly": np.empty((0, 2)),
                "message": "Ingen tryckkraft (N >= 0) – ett ospänt tvärsnitt utan dragkapacitet kan inte uppta ren dragkraft."
            }

        if lin["s_max"] <= 1e-7:
            return {
                **lin,
                "mode": "Sprucket (100 % kontakt)",
                "converged": True,
                "message": "Hela tvärsnittet är i tryck (kraften angriper inom kärnan)."
            }

        e_y = M_z / N
        e_z = M_y / N

        hull_path = Path(self.hull_pts)
        if not hull_path.contains_point((e_y, e_z), radius=-1e-5):
            return {
                "mode": "Sprucket (Stjälpning!)",
                "converged": False,
                "sigma_0": np.nan,
                "c_y": np.nan,
                "c_z": np.nan,
                "v_stress": np.full(len(self.pts), np.nan),
                "s_max": np.nan,
                "s_min": np.nan,
                "active_area_pct": 0.0,
                "compressed_poly": np.empty((0, 2)),
                "message": "Kraftangreppspunkten ligger utanför tvärsnittets konvexa hölje – jämvikt utan drag är omöjlig (stjälpning)!"
            }

        L_scale = max(np.ptp(self.pts[:, 0]), np.ptp(self.pts[:, 1]))
        stress_scale = abs(N) / self.A

        def residuals(params):
            s0 = params[0] * stress_scale
            cy = params[1] * (stress_scale / L_scale)
            cz = params[2] * (stress_scale / L_scale)

            clipped = clip_polygon_halfplane(self.pts, s0, cy, cz)
            if len(clipped) < 3:
                return np.array([10.0, 10.0, 10.0])

            Ac, Syc, Szc, Izzc, Iyyc, Iyzc = polygon_integrals(clipped)
            if Ac < 1e-12:
                return np.array([10.0, 10.0, 10.0])

            N_int = s0 * Ac + cy * Syc + cz * Szc
            Mz_int = s0 * Syc + cy * Izzc + cz * Iyzc
            My_int = s0 * Szc + cy * Iyzc + cz * Iyyc

            r1 = (N_int - N) / abs(N)
            r2 = (Mz_int - M_z) / (abs(N) * L_scale)
            r3 = (My_int - M_y) / (abs(N) * L_scale)
            return np.array([r1, r2, r3])

        x0 = np.array([
            lin["sigma_0"] / stress_scale,
            lin["c_y"] * L_scale / stress_scale,
            lin["c_z"] * L_scale / stress_scale
        ])

        sol = least_squares(residuals, x0, method="lm", max_nfev=600)
        if not sol.success or np.linalg.norm(sol.fun) > 1e-3:
            ang = np.arctan2(e_z, e_y)
            x0_alt = np.array([-0.5, -2.0 * np.cos(ang), -2.0 * np.sin(ang)])
            sol = least_squares(residuals, x0_alt, method="trf", max_nfev=1000)

        s0 = sol.x[0] * stress_scale
        cy = sol.x[1] * (stress_scale / L_scale)
        cz = sol.x[2] * (stress_scale / L_scale)
        clipped = clip_polygon_halfplane(self.pts, s0, cy, cz)
        Ac, _, _, _, _, _ = polygon_integrals(clipped)
        active_pct = 100.0 * (Ac / self.A)

        v_raw = s0 + cy * self.pts[:, 0] + cz * self.pts[:, 1]
        v_stress = np.minimum(0.0, v_raw)
        s_min = float(np.min(v_stress))
        s_max = 0.0
        converged = bool(np.linalg.norm(sol.fun) < 1e-2 and Ac > 1e-9)

        if not converged:
            return {
                "mode": "Sprucket (Stjälpning!)",
                "converged": False,
                "sigma_0": np.nan,
                "c_y": np.nan,
                "c_z": np.nan,
                "v_stress": np.full(len(self.pts), np.nan),
                "s_max": np.nan,
                "s_min": np.nan,
                "active_area_pct": 0.0,
                "compressed_poly": np.empty((0, 2)),
                "message": "Kunde inte finna jämvikt utan drag (stjälpning inträffar)."
            }

        return {
            "mode": "Sprucket (No-Tension)",
            "converged": True,
            "sigma_0": s0,
            "c_y": cy,
            "c_z": cz,
            "v_stress": v_stress,
            "s_max": s_max,
            "s_min": s_min,
            "active_area_pct": active_pct,
            "compressed_poly": clipped,
            "message": "Jämvikt beräknad för sprucket tvärsnitt (0 % dragkapacitet)."
        }


# =====================================================================
# PLOTTNING OCH PDF-RAPPORTGENERERING
# =====================================================================
def create_analysis_figure(section, results_list, active_idx, show_principal=True, show_half_core=True, show_vertices=True):
    s = section
    res = results_list[active_idx]

    pad = 0.28 * max(np.ptp(s.pts[:, 0]), np.ptp(s.pts[:, 1]))
    y_min, y_max = np.min(s.pts[:, 0]) - pad, np.max(s.pts[:, 0]) + pad
    z_min, z_max = np.min(s.pts[:, 1]) - pad, np.max(s.pts[:, 1]) + pad

    yy, zz = np.meshgrid(np.linspace(y_min, y_max, 180), np.linspace(z_min, z_max, 180))
    grid_pts = np.column_stack([yy.ravel(), zz.ravel()])
    inside_mask = Path(s.pts).contains_points(grid_pts).reshape(yy.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8), dpi=110)

    poly_c = np.vstack([s.pts, s.pts[0]])
    hull_c = np.vstack([s.hull_pts, s.hull_pts[0]])
    core_c = np.vstack([s.core_pts, s.core_pts[0]])

    # --- GRAF 1: Tvärsnitt, Kärngräns, Huvudaxlar & Alla Lastfall ---
    ax1.fill(poly_c[:, 0], poly_c[:, 1], color="#d9e3f0", alpha=0.7, label="Tvärsnitt")
    ax1.plot(poly_c[:, 0], poly_c[:, 1], color="#1f3a60", lw=2)
    ax1.plot(hull_c[:, 0], hull_c[:, 1], "k--", lw=1, alpha=0.35, label="Konvext hölje (Stjälpgräns)")

    if show_half_core and len(s.half_core_pts) >= 3:
        hc_c = np.vstack([s.half_core_pts, s.half_core_pts[0]])
        ax1.plot(hc_c[:, 0], hc_c[:, 1], color="#ff7f0e", ls="-.", lw=1.6, label="Andra kärnan (NL genom TP)")
        ax1.fill(hc_c[:, 0], hc_c[:, 1], color="#ff7f0e", alpha=0.10)

    ax1.fill(core_c[:, 0], core_c[:, 1], color="#2ca02c", alpha=0.40, label="Kärngräns (100 % tryck)")
    ax1.plot(core_c[:, 0], core_c[:, 1], color="#1b611b", lw=2)
    ax1.plot(0, 0, "ko", markersize=5, label="TP (0,0)")

    # Huvudtröghetsaxlar (1 och 2)
    if show_principal:
        L_ax = 0.42 * max(np.ptp(s.pts[:, 0]), np.ptp(s.pts[:, 1]))
        ca, sa = np.cos(s.alpha_rad), np.sin(s.alpha_rad)
        ax1.plot([-L_ax * ca, L_ax * ca], [-L_ax * sa, L_ax * sa], color="#8c564b", ls="--", lw=1.3,
                 label=f"Huvudaxlar (α={s.alpha_deg:.1f}°)")
        ax1.plot([-L_ax * (-sa), L_ax * (-sa)], [-L_ax * ca, L_ax * ca], color="#8c564b", ls=":", lw=1.3)

    # Markera hörnen med H1, H2...
    if show_vertices:
        for i, (vy, vz) in enumerate(s.pts):
            ax1.plot(vy, vz, "o", color="#1f3a60", markersize=4)
            ax1.annotate(f"H{i+1}", (vy, vz), textcoords="offset points", xytext=(4, 4),
                         fontsize=7.5, color="#1f3a60", weight="bold")

    # Plotta samtliga lastfalls angreppspunkter
    cmap_lc = plt.get_cmap("tab10")
    for idx, item in enumerate(results_list):
        N_i, My_i, Mz_i = item["N"], item["M_y"], item["M_z"]
        name_i = item["name"]
        if abs(N_i) > 1e-6:
            ey_i, ez_i = Mz_i / N_i, My_i / N_i
            is_sel = (idx == active_idx)
            marker = "X" if is_sel else "o"
            ms = 10 if is_sel else 6.5
            color_i = "red" if is_sel else cmap_lc(idx % 10)
            zord = 10 if is_sel else 7
            ax1.plot(ey_i, ez_i, marker=marker, color=color_i, markersize=ms,
                     markeredgecolor="black", markeredgewidth=0.8, linestyle="None",
                     label=f"{name_i} (Aktiv)" if is_sel else name_i, zorder=zord)
            ax1.annotate(name_i, (ey_i, ez_i), textcoords="offset points", xytext=(6, -6),
                         fontsize=8, weight="bold" if is_sel else "normal", color=color_i, zorder=zord)

    ax1.axhline(0, color="gray", lw=0.7, ls=":")
    ax1.axvline(0, color="gray", lw=0.7, ls=":")
    ax1.set_aspect("equal")
    ax1.set_xlim(y_min, y_max)
    ax1.set_ylim(z_min, z_max)
    ax1.set_title("Tvärsnitt, Kärngräns & Lastangrepp")
    ax1.set_xlabel("y [m] (från TP)")
    ax1.set_ylabel("z [m] (från TP)")
    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- GRAF 2: Spänningsfördelning för valt lastfall ---
    if not res["converged"]:
        # Fysikalisk visualisering av STJÄLPNING (0 % kontaktyta)
        ax2.fill(poly_c[:, 0], poly_c[:, 1], facecolor="#fff0f0", edgecolor="darkred",
                 hatch="///", lw=1.5, label="100 % Sprucken / Glipa (0 % kontakt)")
        ax2.plot(poly_c[:, 0], poly_c[:, 1], color="black", lw=2)

        # Markera tippkant / närmaste hörn mot kraftangreppspunkten
        if res["e_y"] is not None and res["e_z"] is not None:
            dists = np.hypot(s.pts[:, 0] - res["e_y"], s.pts[:, 1] - res["e_z"])
            idx_tip = int(np.argmin(dists))
            ax2.plot(s.pts[idx_tip, 0], s.pts[idx_tip, 1], "rv", markersize=11,
                     markeredgecolor="black", label=f"Tippkant / Stjälphörn: H{idx_tip+1}")

        ax2.text(
            0.5 * (y_min + y_max), 0.5 * (z_min + z_max),
            "STJÄLPNING!\n(0 % kontaktyta – jämvikt\nutan drag existerar ej)",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#990000",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffe6e6", edgecolor="#cc0000", lw=1.5),
            zorder=15
        )
    else:
        s0, cy, cz = res["sigma_0"], res["c_y"], res["c_z"]
        sigma_raw = s0 + cy * yy + cz * zz
        if res["is_cracked_model"]:
            sigma_grid = np.minimum(0.0, sigma_raw)
        else:
            sigma_grid = sigma_raw

        sigma_masked = np.where(inside_mask, sigma_grid, np.nan)
        s_max, s_min = res["s_max"], res["s_min"]
        abs_max = max(abs(s_min), abs(s_max), 1.0)

        mesh = ax2.pcolormesh(yy, zz, sigma_masked, cmap="coolwarm",
                              vmin=-abs_max, vmax=abs_max, shading="auto")
        cbar = fig.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Spänning σ [kPa] (+ Drag / - Tryck)")

        ax2.plot(poly_c[:, 0], poly_c[:, 1], color="black", lw=2)

        # Skravera dragzon / sprucken zon (där sigma_raw > 0)
        raw_masked = np.where(inside_mask, sigma_raw, np.nan)
        if np.nanmax(raw_masked) > 1e-6:
            max_raw = float(np.nanmax(raw_masked))
            lbl_zone = "Sprucken zon (σ = 0)" if res["is_cracked_model"] else "Dragzon (σ > 0)"
            ax2.contourf(yy, zz, raw_masked,
                         levels=[0.0, max(max_raw * 1.05, 1e-3)],
                         colors=["none"], hatches=["///"], extend="max")
            ax2.fill([], [], facecolor="none", edgecolor="darkred", hatch="///", label=lbl_zone)

        # Neutrala lagret (sigma = 0)
        if np.hypot(cy, cz) > 1e-12:
            if abs(cz) >= abs(cy):
                nl_y = np.array([y_min, y_max])
                nl_z = -(s0 + cy * nl_y) / cz
            else:
                nl_z = np.array([z_min, z_max])
                nl_y = -(s0 + cz * nl_z) / cy
            ax2.plot(nl_y, nl_z, "k--", lw=2, label="Neutrala lagret (σ = 0)")

        # Markera hörn med minsta (störst tryck) och största spänning
        idx_min = int(np.argmin(res["v_stress"]))
        idx_max = int(np.argmax(res["v_stress"]))
        ax2.plot(s.pts[idx_min, 0], s.pts[idx_min, 1], "bv", markersize=8,
                 label=f"Min: H{idx_min+1} ({s_min:+.1f} kPa)")
        ax2.plot(s.pts[idx_max, 0], s.pts[idx_max, 1], "r^", markersize=8,
                 label=f"Max: H{idx_max+1} ({s_max:+.1f} kPa)")

    ax2.axhline(0, color="gray", lw=0.7, ls=":")
    ax2.axvline(0, color="gray", lw=0.7, ls=":")
    ax2.set_aspect("equal")
    ax2.set_xlim(y_min, y_max)
    ax2.set_ylim(z_min, z_max)
    ax2.set_title(f"Spänning: {res['name']} [{res['mode']}]")
    ax2.set_xlabel("y [m] (från TP)")
    ax2.set_ylabel("z [m] (från TP)")
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def generate_pdf_report(project_name, author_name, section, results_list, active_idx, fig_main):
    buf = io.BytesIO()
    s = section

    with PdfPages(buf) as pdf:
        # --- SIDA 1: Sammanfattning, Tvärsnittsdata & Lastfallstabell ---
        fig_p1 = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 stående
        fig_p1.patch.set_facecolor("white")

        ax_t = fig_p1.add_axes([0.06, 0.52, 0.88, 0.43])
        ax_t.axis("off")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        ax_t.text(0.0, 0.96, "BERÄKNINGSRAPPORT – KÄRNGRÄNS & SPÄNNINGSANALYS",
                  fontsize=13, weight="bold", color="#1f3a60")
        ax_t.text(0.0, 0.90, f"Projekt: {project_name}    |    Handläggare: {author_name}    |    Datum: {now_str}",
                  fontsize=9.5, color="#333333")
        ax_t.plot([0.0, 1.0], [0.87, 0.87], color="#1f3a60", lw=1.5)

        prop_text = (
            "1. TVÄRSNITTSEGENSKAPER (Tyngdpunkts- och huvudaxelsystem)\n"
            f"   Area A:                  {s.A:.4f} m²\n"
            f"   Tyngdpunkt (y_tp, z_tp): ({s.y_tp:.4f}, {s.z_tp:.4f}) m\n"
            f"   Tröghetsmoment I_yy:     {s.I_yy:.5e} m⁴   (böjning kring y-axel / i z-led)\n"
            f"   Tröghetsmoment I_zz:     {s.I_zz:.5e} m⁴   (böjning kring z-axel / i y-led)\n"
            f"   Deviationsmoment I_yz:   {s.I_yz:.5e} m⁴\n"
            f"   Huvudtröghetsmoment I_1: {s.I_1:.5e} m⁴\n"
            f"   Huvudtröghetsmoment I_2: {s.I_2:.5e} m⁴\n"
            f"   Huvudaxelvinkel α:       {s.alpha_deg:+.2f}°"
        )
        ax_t.text(0.0, 0.44, prop_text, fontsize=9, family="monospace", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f7fa", edgecolor="#c5d1e0"))

        ax_t.text(0.0, 0.36, "2. SAMMANSTÄLLNING AV LASTFALL", fontsize=10.5, weight="bold", color="#1f3a60")

        col_labels = ["Lastfall", "Modell", "N [kN]", "My [kNm]", "Mz [kNm]", "e_y [m]", "e_z [m]",
                      "σ_min [kPa]", "σ_max [kPa]", "Kontakt [%]"]
        table_data = []
        for r in results_list:
            ey_s = f"{r['e_y']:+.3f}" if r["e_y"] is not None else "-"
            ez_s = f"{r['e_z']:+.3f}" if r["e_z"] is not None else "-"
            smin_s = f"{r['s_min']:+.1f}" if r["converged"] else "Stjälpning"
            smax_s = f"{r['s_max']:+.1f}" if r["converged"] else "Stjälpning"
            table_data.append([
                r["name"],
                "Sprucket" if r["is_cracked_model"] else "Navier",
                f"{r['N']:.1f}",
                f"{r['M_y']:.1f}",
                f"{r['M_z']:.1f}",
                ey_s,
                ez_s,
                smin_s,
                smax_s,
                f"{r['active_area_pct']:.1f}%"
            ])

        tbl = ax_t.table(cellText=table_data, colLabels=col_labels,
                         loc="bottom", bbox=[0.0, -0.05, 1.0, 0.38], cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.2)
        for (row_i, col_i), cell in tbl.get_celld().items():
            if row_i == 0:
                cell.set_facecolor("#1f3a60")
                cell.set_text_props(color="white", weight="bold")
            elif row_i - 1 == active_idx:
                cell.set_facecolor("#e8f0fe")

        # Infoga figurerna på nedre halvan av Sida 1
        buf_img = io.BytesIO()
        fig_main.savefig(buf_img, format="png", dpi=160, bbox_inches="tight")
        buf_img.seek(0)
        img_arr = plt.imread(buf_img)

        ax_img = fig_p1.add_axes([0.04, 0.04, 0.92, 0.44])
        ax_img.imshow(img_arr)
        ax_img.axis("off")

        pdf.savefig(fig_p1)
        plt.close(fig_p1)

        # --- SIDA 2: Detaljerad Hörntabell för aktivt lastfall ---
        fig_p2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig_p2.patch.set_facecolor("white")
        ax_p2 = fig_p2.add_axes([0.06, 0.06, 0.88, 0.88])
        ax_p2.axis("off")

        active_res = results_list[active_idx]
        ax_p2.text(0.0, 0.96, f"3. HÖRNKOORDINATER OCH HÖRNSPÄNNINGAR – {active_res['name']} ({active_res['mode']})",
                   fontsize=11.5, weight="bold", color="#1f3a60")
        ax_p2.plot([0.0, 1.0], [0.94, 0.94], color="#1f3a60", lw=1.2)

        v_headers = ["Hörn", "y_orig [m]", "z_orig [m]", "y_TP [m]", "z_TP [m]",
                     "σ (Navier) [kPa]", "σ (Vald modell) [kPa]", "Status (Vald modell)"]
        v_rows = []
        lin_v = s.calculate_linear(active_res["N"], active_res["M_y"], active_res["M_z"])["v_stress"]

        for i in range(len(s.pts)):
            sig_val = active_res["v_stress"][i]
            sig_lin = lin_v[i]
            if not active_res["converged"]:
                sig_val_str = "Ej jämvikt"
                st_txt = "STJÄLPNING (Glipa)"
            else:
                sig_val_str = f"{sig_val:+.2f}"
                st_txt = "TRYCK" if sig_val < -1e-6 else ("DRAG" if sig_val > 1e-6 else "Sprucken (0)")

            v_rows.append([
                f"H{i+1}",
                f"{s.pts_orig[i, 0]:.4f}",
                f"{s.pts_orig[i, 1]:.4f}",
                f"{s.pts[i, 0]:+.4f}",
                f"{s.pts[i, 1]:+.4f}",
                f"{sig_lin:+.2f}",
                sig_val_str,
                st_txt
            ])

        tbl_h = max(0.18, min(0.75, 0.045 * (len(v_rows) + 1)))
        tbl2 = ax_p2.table(cellText=v_rows, colLabels=v_headers,
                           loc="upper center", bbox=[0.0, 0.90 - tbl_h, 1.0, tbl_h], cellLoc="center")
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        for (row_i, col_i), cell in tbl2.get_celld().items():
            if row_i == 0:
                cell.set_facecolor("#1f3a60")
                cell.set_text_props(color="white", weight="bold")

        pdf.savefig(fig_p2)
        plt.close(fig_p2)

    buf.seek(0)
    return buf.getvalue()


# =====================================================================
# STREAMLIT APPLIKATION (HUVUDGRÄNSSNITT)
# =====================================================================
st.set_page_config(page_title="Kärngräns & Spänningsanalys", layout="wide")

PRESETS = {
    "T-tvärsnitt": "0.15, 0.0\n0.35, 0.0\n0.35, 0.45\n0.50, 0.45\n0.50, 0.60\n0.00, 0.60\n0.00, 0.45\n0.15, 0.45",
    "L-tvärsnitt (Osymmetriskt)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.15\n0.15, 0.15\n0.15, 0.5\n0.0, 0.5",
    "Rektangel (0.4 x 0.6 m)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.6\n0.0, 0.6",
    "Trapets (Damm/Fundament)": "0.0, 0.0\n4.0, 0.0\n3.2, 5.0\n0.4, 5.0",
    "Hålkäl / Avfasat fundament": "0.0, 0.0\n2.0, 0.0\n2.0, 1.2\n1.5, 1.8\n0.5, 1.8\n0.0, 1.2"
}

DEFAULT_LOADS = [
    {"Lastfall": "LF1 (Bruks)", "N [kN]": -150.0, "My [kNm]": 5.0, "Mz [kNm]": 4.0},
    {"Lastfall": "LF2 (Karakteristisk)", "N [kN]": -120.0, "My [kNm]": 12.0, "Mz [kNm]": -8.0},
    {"Lastfall": "LF3 (Brott / Extrem)", "N [kN]": -100.0, "My [kNm]": 18.0, "Mz [kNm]": 11.0},
]

if "coords_text" not in st.session_state:
    st.session_state["coords_text"] = PRESETS["T-tvärsnitt"]
if "loads_df" not in st.session_state:
    st.session_state["loads_df"] = pd.DataFrame(DEFAULT_LOADS)
if "project_name" not in st.session_state:
    st.session_state["project_name"] = "Tvärsnittskontroll"
if "author_name" not in st.session_state:
    st.session_state["author_name"] = ""

# --- SIDOPANEL: Projektinfo, Spara / Ladda JSON & Visningsval ---
with st.sidebar:
    st.header("Projekt & Export")
    project_name = st.text_input("Projektnamn:", value=st.session_state["project_name"])
    author_name = st.text_input("Handläggare / Signatur:", value=st.session_state["author_name"])
    st.session_state["project_name"] = project_name
    st.session_state["author_name"] = author_name

    st.divider()
    st.subheader("Beräkningsmodell")
    use_cracked = st.checkbox(
        "Sprucket tvärsnitt (No-Tension / 0 drag)",
        value=False,
        help="Om ikryssad antas materialet/fogen sakna draghållfasthet (σ ≤ 0). Neutrala lagret itereras fram så att jämvikt råder över den komprimerade delen."
    )

    st.subheader("Grafinställningar")
    show_principal = st.checkbox("Visa huvudtröghetsaxlar (I₁, I₂)", value=True)
    show_half_core = st.checkbox("Visa andra kärnan (NL genom TP)", value=True)
    show_vertices = st.checkbox("Visa hörnnummer (H1, H2...)", value=True)

    st.divider()
    st.subheader("Dela / Spara indata (JSON)")
    uploaded_json = st.file_uploader("Ladda in sparad JSON-fil:", type=["json"])
    if uploaded_json is not None:
        try:
            data_in = json.load(uploaded_json)
            if "coords_text" in data_in:
                st.session_state["coords_text"] = data_in["coords_text"]
            if "loads" in data_in:
                st.session_state["loads_df"] = pd.DataFrame(data_in["loads"])
            if "project_name" in data_in:
                st.session_state["project_name"] = data_in["project_name"]
            st.success("Indata inläst!")
        except Exception as e:
            st.error(f"Kunde inte läsa JSON-fil: {e}")

# --- HUVUDVY ---
st.title("Interaktiv Kärngräns- och Spänningsberäknare (Navier & No-Tension)")

col_left, col_right = st.columns([1.15, 2.05], gap="large")

with col_left:
    st.subheader("1. Tvärsnittsgeometri (y, z) [m]")
    preset_sel = st.selectbox("Välj mall (eller redigera koordinater nedan):", list(PRESETS.keys()))
    if st.button("Ladda vald mall"):
        st.session_state["coords_text"] = PRESETS[preset_sel]
        st.rerun()

    coords_raw = st.text_area(
        "Hörnkoordinater y, z [m] (en punkt per rad):",
        value=st.session_state["coords_text"],
        height=145
    )
    st.session_state["coords_text"] = coords_raw

    st.subheader("2. Lastfall (Lägg till / ändra rader)")
    st.caption("Konvention: N < 0 ger tryckspänning (σ < 0), N > 0 ger dragspänning (σ > 0).")
    edited_df = st.data_editor(
        st.session_state["loads_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="loads_editor"
    )
    st.session_state["loads_df"] = edited_df

# Tolka tvärsnittskoordinater
pts = []
try:
    for line in coords_raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(";", ",").split(",")
        pts.append((float(parts[0].strip()), float(parts[1].strip())))
    section = CrossSectionNavier(pts)
except Exception as e:
    st.error(f"Fel i tvärsnittsdata: {e}")
    st.stop()

# Beräkna alla lastfall i tabellen
results_list = []
for idx, row in edited_df.iterrows():
    name_str = str(row.get("Lastfall", f"LF{idx+1}")).strip() or f"LF{idx+1}"
    try:
        N_val = float(row.get("N [kN]", -100.0))
        My_val = float(row.get("My [kNm]", 0.0))
        Mz_val = float(row.get("Mz [kNm]", 0.0))
        if np.isnan(N_val) or np.isnan(My_val) or np.isnan(Mz_val):
            continue
    except (ValueError, TypeError):
        continue

    if use_cracked:
        res_calc = section.calculate_cracked(N_val, My_val, Mz_val)
    else:
        res_calc = section.calculate_linear(N_val, My_val, Mz_val)

    e_y = (Mz_val / N_val) if abs(N_val) > 1e-6 else None
    e_z = (My_val / N_val) if abs(N_val) > 1e-6 else None

    results_list.append({
        "name": name_str,
        "N": N_val,
        "M_y": My_val,
        "M_z": Mz_val,
        "e_y": e_y,
        "e_z": e_z,
        "is_cracked_model": use_cracked,
        **res_calc
    })

if not results_list:
    st.warning("Lägg till minst ett giltigt lastfall i tabellen.")
    st.stop()

# Skapa JSON-exportknapp i sidopanelen nu när indata är verifierad
with st.sidebar:
    export_payload = {
        "project_name": project_name,
        "author_name": author_name,
        "coords_text": coords_raw,
        "loads": edited_df.to_dict(orient="records")
    }
    st.download_button(
        label="Ladda ner indata (.json)",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name=f"{project_name.lower().replace(' ', '_')}_indata.json",
        mime="application/json",
        use_container_width=True
    )

with col_left:
    st.subheader("3. Välj aktivt lastfall för detaljvy")
    lc_options = []
    for r in results_list:
        if r["converged"]:
            lc_options.append(f"{r['name']} (σ_min={r['s_min']:+.1f} kPa, Kontakt={r['active_area_pct']:.0f}%)")
        else:
            lc_options.append(f"{r['name']} (STJÄLPNING! Kontakt=0%)")
    lc_options.append("Dimensionerande (Störst tryck / Lägst kontakt)")

    sel_option = st.selectbox("Visa spänningskontur och hörndata för:", lc_options)
    if sel_option == "Dimensionerande (Störst tryck / Lägst kontakt)":
        non_conv = [i for i, r in enumerate(results_list) if not r["converged"]]
        if non_conv:
            active_idx = non_conv[0]
        else:
            active_idx = int(np.argmin([r["s_min"] for r in results_list]))
    else:
        active_idx = lc_options.index(sel_option)

    active_res = results_list[active_idx]

    if not active_res["converged"]:
        st.error(f"{active_res['name']}: {active_res['message']}")
    elif active_res["active_area_pct"] < 99.99:
        st.warning(
            f"{active_res['name']}: Kraften angriper utanför kärnan! "
            f"Aktiv tryckt area: {active_res['active_area_pct']:.1f} %"
        )
    else:
        st.success(f"{active_res['name']}: Hela tvärsnittet är i TRYCK (100 % kontakt).")

    s = section
    e_str = f"({active_res['e_y']:+.3f}, {active_res['e_z']:+.3f}) m" if active_res["e_y"] is not None else "Oändlig (N = 0)"
    smin_txt = f"{active_res['s_min']:+.2f} kPa" if active_res["converged"] else "Ej jämvikt (Stjälpning!)"
    smax_txt = f"{active_res['s_max']:+.2f} kPa" if active_res["converged"] else "Ej jämvikt (Stjälpning!)"

    st.code(
        f"Area A:          {s.A:.4f} m²\n"
        f"TP (y, z):       ({s.y_tp:.4f}, {s.z_tp:.4f}) m\n"
        f"I_yy, I_zz:      {s.I_yy:.4e}, {s.I_zz:.4e} m⁴\n"
        f"I_yz:            {s.I_yz:.4e} m⁴\n"
        f"I_1, I_2:        {s.I_1:.4e}, {s.I_2:.4e} m⁴\n"
        f"Huvudvinkel α:   {s.alpha_deg:+.2f}°\n"
        f"{'-'*38}\n"
        f"Aktivt lastfall: {active_res['name']}\n"
        f"Modell:          {active_res['mode']}\n"
        f"Angreppspunkt:   {e_str}\n"
        f"σ_min (Tryck):   {smin_txt}\n"
        f"σ_max (Drag):    {smax_txt}\n"
        f"Tryckt area:     {active_res['active_area_pct']:.1f} %"
    )

with col_right:
    fig_main = create_analysis_figure(
        section, results_list, active_idx,
        show_principal=show_principal,
        show_half_core=show_half_core,
        show_vertices=show_vertices
    )
    st.pyplot(fig_main)

    pdf_bytes = generate_pdf_report(project_name, author_name, section, results_list, active_idx, fig_main)
    plt.close(fig_main)

    st.download_button(
        label="📄 Ladda ner komplett PDF-beräkningsrapport (A4)",
        data=pdf_bytes,
        file_name=f"{project_name.lower().replace(' ', '_')}_rapport.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    # Sammanställningstabell för alla lastfall
    st.subheader("Sammanställning av samtliga lastfall")
    summary_rows = []
    for r in results_list:
        summary_rows.append({
            "Lastfall": r["name"],
            "Modell": r["mode"],
            "N [kN]": round(r["N"], 2),
            "My [kNm]": round(r["M_y"], 2),
            "Mz [kNm]": round(r["M_z"], 2),
            "e_y [m]": round(r["e_y"], 4) if r["e_y"] is not None else None,
            "e_z [m]": round(r["e_z"], 4) if r["e_z"] is not None else None,
            "σ_min [kPa]": f"{r['s_min']:+.2f}" if r["converged"] else "Stjälpning (Ej jämvikt)",
            "σ_max [kPa]": f"{r['s_max']:+.2f}" if r["converged"] else "Stjälpning (Ej jämvikt)",
            "Tryckt area [%]": round(r["active_area_pct"], 1)
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Expanderbar hörntabell
    with st.expander(f"Visa hörnkoordinater och spänningar i varje hörn ({active_res['name']})", expanded=False):
        lin_v = section.calculate_linear(active_res["N"], active_res["M_y"], active_res["M_z"])["v_stress"]
        corner_rows = []
        for i in range(len(section.pts)):
            sig_v = active_res["v_stress"][i]
            if not active_res["converged"]:
                sig_str = "Ej jämvikt (Stjälpning)"
                st_str = "STJÄLPNING (100 % glipa)"
            else:
                sig_str = f"{sig_v:+.2f}"
                st_str = "TRYCK" if sig_v < -1e-6 else ("DRAG" if sig_v > 1e-6 else "Sprucken (0)")

            corner_rows.append({
                "Hörn": f"H{i+1}",
                "y_orig [m]": round(float(section.pts_orig[i, 0]), 4),
                "z_orig [m]": round(float(section.pts_orig[i, 1]), 4),
                "y_TP [m]": round(float(section.pts[i, 0]), 4),
                "z_TP [m]": round(float(section.pts[i, 1]), 4),
                "σ (Navier) [kPa]": f"{lin_v[i]:+.2f}",
                "σ (Vald modell) [kPa]": sig_str,
                "Status (Vald modell)": st_str
            })
        st.dataframe(pd.DataFrame(corner_rows), use_container_width=True, hide_index=True)
