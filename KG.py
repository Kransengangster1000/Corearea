import streamlit as st
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.path import Path


class CrossSectionNavier:
    """Beräkningsmotor för tvärsnittsegenskaper, kärngräns och Naviers formel."""
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
        self._compute_core_boundary()

    @staticmethod
    def _signed_area(pts):
        y, z = pts[:, 0], pts[:, 1]
        return 0.5 * np.sum(y * np.roll(z, -1) - np.roll(y, -1) * z)

    def _compute_section_properties(self):
        y, z = self.pts_orig[:, 0], self.pts_orig[:, 1]
        y_n, z_n = np.roll(y, -1), np.roll(z, -1)
        cross = y * z_n - y_n * z

        self.A = 0.5 * np.sum(cross)
        if abs(self.A) < 1e-12:
            raise ValueError("Tvärsnittets area är noll (punkterna ligger på en linje).")

        self.y_tp = np.sum((y + y_n) * cross) / (6.0 * self.A)
        self.z_tp = np.sum((z + z_n) * cross) / (6.0 * self.A)

        self.pts = self.pts_orig - np.array([self.y_tp, self.z_tp])
        yc, zc = self.pts[:, 0], self.pts[:, 1]
        yc_n, zc_n = np.roll(yc, -1), np.roll(zc, -1)
        cross_c = yc * zc_n - yc_n * zc

        self.I_zz = np.sum((yc**2 + yc * yc_n + yc_n**2) * cross_c) / 12.0
        self.I_yy = np.sum((zc**2 + zc * zc_n + zc_n**2) * cross_c) / 12.0
        self.I_yz = np.sum((yc * zc_n + 2 * yc * zc + 2 * yc_n * zc_n + yc_n * zc) * cross_c) / 24.0

        self.I_mat = np.array([
            [self.I_zz, self.I_yz],
            [self.I_yz, self.I_yy]
        ])

    def _compute_core_boundary(self):
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

    def calculate_stress(self, N, M_y, M_z):
        """
        N > 0 = Dragkraft (ger positiv spänning sigma > 0)
        N < 0 = Tryckkraft (ger negativ spänning sigma < 0)
        """
        sigma_0 = N / self.A
        c_vec = np.linalg.solve(self.I_mat, np.array([M_z, M_y]))
        return sigma_0, c_vec[0], c_vec[1]


# --- STREAMLIT WEBBAPP ---
st.set_page_config(page_title="Kärngräns & Navier", layout="wide")
st.title("Kärngräns- och Spänningsberäknare (Navier)")

presets = {
    "T-tvärsnitt": "0.15, 0.0\n0.35, 0.0\n0.35, 0.45\n0.50, 0.45\n0.50, 0.60\n0.00, 0.60\n0.00, 0.45\n0.15, 0.45",
    "L-tvärsnitt (Osymmetriskt)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.15\n0.15, 0.15\n0.15, 0.5\n0.0, 0.5",
    "Rektangel (0.4 x 0.6 m)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.6\n0.0, 0.6",
    "Trapets": "0.0, 0.0\n0.6, 0.0\n0.45, 0.5\n0.15, 0.5"
}

col_ctrl, col_plot = st.columns([1, 2.2])

with col_ctrl:
    st.subheader("1. Tvärsnittets koordinater (y, z) [m]")
    preset_choice = st.selectbox("Välj mall:", list(presets.keys()))
    
    coords_raw = st.text_area(
        "Koordinater (ändra fritt, en punkt per rad):",
        value=presets[preset_choice],
        height=160
    )

    st.subheader("2. Belastning")
    N = st.number_input("Normalkraft N [kN] (+Drag / -Tryck):", min_value=-5000.0, max_value=5000.0, value=-100.0, step=10.0)
    M_y = st.number_input("Moment My [kNm] (böjning i z-led):", min_value=-2000.0, max_value=2000.0, value=0.0, step=5.0)
    M_z = st.number_input("Moment Mz [kNm] (böjning i y-led):", min_value=-2000.0, max_value=2000.0, value=0.0, step=5.0)

# Tolka koordinater och beräkna
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

s = section
sigma_0, c_y, c_z = s.calculate_stress(N, M_y, M_z)
v_stress = sigma_0 + c_y * s.pts[:, 0] + c_z * s.pts[:, 1]
s_max, s_min = np.max(v_stress), np.min(v_stress)

with col_ctrl:
    st.subheader("3. Beräkningsresultat")
    if s_max > 1e-6:
        st.error("Status: DRAGZON FÖRELIGGER (σ > 0)")
    else:
        st.success("Status: Hela tvärsnittet i TRYCK")

    e_str = f"({M_z/N:+.3f}, {M_y/N:+.3f}) m" if abs(N) > 1e-6 else "Oändlig (N = 0)"
    st.code(
        f"Area A:       {s.A:.4f} m²\n"
        f"TP (y, z):    ({s.y_tp:.3f}, {s.z_tp:.3f}) m\n"
        f"I_zz:         {s.I_zz:.5e} m⁴\n"
        f"I_yy:         {s.I_yy:.5e} m⁴\n"
        f"I_yz:         {s.I_yz:.5e} m⁴\n"
        f"{'-'*32}\n"
        f"Angreppspkt:  {e_str}\n"
        f"sigma_max:    {s_max:+.2f} kPa\n"
        f"sigma_min:    {s_min:+.2f} kPa"
    )

with col_plot:
    pad = 0.25 * max(np.ptp(s.pts[:, 0]), np.ptp(s.pts[:, 1]))
    y_min, y_max = np.min(s.pts[:, 0]) - pad, np.max(s.pts[:, 0]) + pad
    z_min, z_max = np.min(s.pts[:, 1]) - pad, np.max(s.pts[:, 1]) + pad

    yy, zz = np.meshgrid(np.linspace(y_min, y_max, 180), np.linspace(z_min, z_max, 180))
    grid_pts = np.column_stack([yy.ravel(), zz.ravel()])
    inside_mask = Path(s.pts).contains_points(grid_pts).reshape(yy.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), dpi=100)

    poly_c = np.vstack([s.pts, s.pts[0]])
    hull_c = np.vstack([s.hull_pts, s.hull_pts[0]])
    core_c = np.vstack([s.core_pts, s.core_pts[0]])

    # Graf 1: Tvärsnitt & Kärngräns
    ax1.fill(poly_c[:, 0], poly_c[:, 1], color="#d9e3f0", alpha=0.7, label="Tvärsnitt")
    ax1.plot(poly_c[:, 0], poly_c[:, 1], color="#1f3a60", lw=2)
    ax1.plot(hull_c[:, 0], hull_c[:, 1], "k--", lw=1, alpha=0.4, label="Konvext hölje")
    ax1.fill(core_c[:, 0], core_c[:, 1], color="#2ca02c", alpha=0.4, label="Kärngräns (Kärna)")
    ax1.plot(core_c[:, 0], core_c[:, 1], color="#1b611b", lw=2)
    ax1.plot(0, 0, "ko", markersize=5, label="TP (0,0)")

    if abs(N) > 1e-6:
        e_y, e_z = M_z / N, M_y / N
        ax1.plot(e_y, e_z, "rX", markersize=10, markeredgewidth=2, label="Kraftangrepp (e)")

    ax1.axhline(0, color="gray", lw=0.7, ls=":")
    ax1.axvline(0, color="gray", lw=0.7, ls=":")
    ax1.set_aspect("equal")
    ax1.set_xlim(y_min, y_max)
    ax1.set_ylim(z_min, z_max)
    ax1.set_title("Tvärsnitt & Kärngräns")
    ax1.set_xlabel("y [m]")
    ax1.set_ylabel("z [m]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Graf 2: Spänningsfördelning
    sigma_grid = sigma_0 + c_y * yy + c_z * zz
    sigma_masked = np.where(inside_mask, sigma_grid, np.nan)

    abs_max = max(abs(s_min), abs(s_max), 1.0)
    mesh = ax2.pcolormesh(yy, zz, sigma_masked, cmap="coolwarm",
                          vmin=-abs_max, vmax=abs_max, shading="auto")

    cbar = fig.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Spänning [kPa] (+ Drag / - Tryck)")

    ax2.plot(poly_c[:, 0], poly_c[:, 1], color="black", lw=2)

    if s_max > 1e-6:
        ax2.contourf(yy, zz, sigma_masked,
                     levels=[0.0, max(s_max * 1.05, 1e-3)],
                     colors=["none"], hatches=["///"], extend="max")
        ax2.fill([], [], facecolor="none", edgecolor="darkred", hatch="///", label="Dragzon (σ > 0)")

    if np.hypot(c_y, c_z) > 1e-12:
        if abs(c_z) >= abs(c_y):
            nl_y = np.array([y_min, y_max])
            nl_z = -(sigma_0 + c_y * nl_y) / c_z
        else:
            nl_z = np.array([z_min, z_max])
            nl_y = -(sigma_0 + c_z * nl_z) / c_y
        ax2.plot(nl_y, nl_z, "k--", lw=2, label="Neutrala lagret (σ = 0)")

    ax2.axhline(0, color="gray", lw=0.7, ls=":")
    ax2.axvline(0, color="gray", lw=0.7, ls=":")
    ax2.set_aspect("equal")
    ax2.set_xlim(y_min, y_max)
    ax2.set_ylim(z_min, z_max)
    ax2.set_title("Spänningsfördelning & Dragzon")
    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
