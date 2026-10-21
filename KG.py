import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path

# Försök importera paket för AI-skissanalys (valfritt)
try:
    from PIL import Image
    from pydantic import BaseModel, Field
    from google import genai
    from google.genai import types

    class Point2D(BaseModel):
        y: float = Field(description="Horisontell koordinat (bredd) i meter")
        z: float = Field(description="Vertikal koordinat (höjd) i meter")

    class ExtractedCrossSection(BaseModel):
        explanation: str = Field(description="Kort förklaring av hur måtten tolkades.")
        vertices: list[Point2D] = Field(description="Hörnkoordinater (y, z) i meter runt ytterkanten.")

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


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


class NavierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interaktiv Kärngräns- och Spänningsberäknare (Navier)")
        self.geometry("1380x820")

        self.presets = {
            "T-tvärsnitt": "0.15, 0.0\n0.35, 0.0\n0.35, 0.45\n0.50, 0.45\n0.50, 0.60\n0.00, 0.60\n0.00, 0.45\n0.15, 0.45",
            "L-tvärsnitt (Osymmetriskt)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.15\n0.15, 0.15\n0.15, 0.5\n0.0, 0.5",
            "Rektangel (0.4 x 0.6 m)": "0.0, 0.0\n0.4, 0.0\n0.4, 0.6\n0.0, 0.6",
            "Trapets": "0.0, 0.0\n0.6, 0.0\n0.45, 0.5\n0.15, 0.5"
        }

        # Startar med N = -100.0 kN (Tryckkraft!) och My=0, Mz=0
        self.N_var = tk.DoubleVar(value=-100.0)
        self.My_var = tk.DoubleVar(value=0.0)
        self.Mz_var = tk.DoubleVar(value=0.0)
        self.api_key_var = tk.StringVar(value="")
        self._updating_sliders = False

        self._build_ui()
        self.load_section_from_text()

    def _build_ui(self):
        ctrl_frame = ttk.Frame(self, padding=10, width=370)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        ctrl_frame.pack_propagate(False)

        # 1. Tvärsnittsgeometri (Manuellt + AI-skiss)
        geom_box = ttk.LabelFrame(ctrl_frame, text="1. Tvärsnittets koordinater (y, z) [m]", padding=8)
        geom_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(geom_box, text="Välj mall eller skriv/ändra koordinater manuellt:").pack(anchor="w")
        self.preset_cb = ttk.Combobox(geom_box, values=list(self.presets.keys()), state="readonly")
        self.preset_cb.current(0)
        self.preset_cb.pack(fill=tk.X, pady=3)
        self.preset_cb.bind("<<ComboboxSelected>>", self.on_preset_change)

        self.coords_text = tk.Text(geom_box, height=7, width=30, font=("Consolas", 10))
        self.coords_text.pack(fill=tk.X, pady=3)
        self.coords_text.insert("1.0", self.presets["T-tvärsnitt"])

        ttk.Button(geom_box, text="Uppdatera tvärsnitt från text", command=self.load_section_from_text).pack(fill=tk.X, pady=2)

        # AI-skissuppladdning
        ai_frame = ttk.Frame(geom_box)
        ai_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(ai_frame, text="Gemini API-nyckel (för skissanalys):").pack(anchor="w")
        ttk.Entry(ai_frame, textvariable=self.api_key_var, show="*").pack(fill=tk.X, pady=2)
        ttk.Button(ai_frame, text="Ladda upp handritad skiss (AI)...", command=self.analyze_sketch_image).pack(fill=tk.X, pady=2)

        # 2. Belastning (Manuell sifferruta + reglage)
        load_box = ttk.LabelFrame(ctrl_frame, text="2. Belastning (Skriv värde + Enter eller dra)", padding=8)
        load_box.pack(fill=tk.X, pady=(0, 8))

        self._create_load_control(load_box, "Normalkraft N [kN] (+Drag / -Tryck):", self.N_var, -500.0, 500.0)
        self._create_load_control(load_box, "Moment My [kNm] (böjning i z-led):", self.My_var, -100.0, 100.0)
        self._create_load_control(load_box, "Moment Mz [kNm] (böjning i y-led):", self.Mz_var, -100.0, 100.0)

        # 3. Resultat
        res_box = ttk.LabelFrame(ctrl_frame, text="3. Beräkningsresultat", padding=8)
        res_box.pack(fill=tk.BOTH, expand=True)

        self.res_label = tk.Label(res_box, text="", justify=tk.LEFT, anchor="nw", font=("Consolas", 9))
        self.res_label.pack(fill=tk.BOTH, expand=True)

        # Höger panel: Grafer
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9.5, 6.5), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.cbar = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_load_control(self, parent, label_text, var, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)

        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X)
        ttk.Label(top_row, text=label_text).pack(side=tk.LEFT)

        entry = ttk.Entry(top_row, width=9, justify="right")
        entry.pack(side=tk.RIGHT)
        entry.insert(0, f"{var.get():.2f}")

        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X, pady=2)

        def on_slider_move(_):
            if self._updating_sliders:
                return
            entry.delete(0, tk.END)
            entry.insert(0, f"{var.get():.2f}")
            self.update_plot()

        def on_entry_commit(_=None):
            try:
                val = float(entry.get().replace(",", "."))
                self._updating_sliders = True
                var.set(val)
                self._updating_sliders = False
                self.update_plot()
            except ValueError:
                pass

        slider.configure(command=on_slider_move)
        entry.bind("<Return>", on_entry_commit)
        entry.bind("<FocusOut>", on_entry_commit)
        entry.bind("<KeyRelease>", on_entry_commit)

    def analyze_sketch_image(self):
        if not AI_AVAILABLE:
            messagebox.showwarning(
                "Saknar AI-paket",
                "För att analysera skisser i Tkinter, öppna Kommandotolken (cmd) och kör:\n\n"
                "py -m pip install pillow google-genai pydantic"
            )
            return

        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("API-nyckel saknas", "Skriv in din Gemini API-nyckel i fältet ovanför knappen först.")
            return

        file_path = filedialog.askopenfilename(
            title="Välj bild på handritad skiss",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            client = genai.Client(api_key=api_key)
            prompt = (
                "Analysera denna handritade skiss av ett tvärsnitt. "
                "Identifiera formen och alla utskrivna mått. "
                "Placera origo (0, 0) i det nedre vänstra hörnet "
                "och räkna ut (y, z)-koordinaterna i METER för alla hörn i ordning runt profilens ytterkant "
                "(y är horisontell axel åt höger, z är vertikal axel uppåt). "
                "Om måtten är i mm eller cm, omvandla dem till meter."
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedCrossSection,
                    temperature=0.1,
                ),
            )
            parsed = response.parsed
            new_coords = "\n".join([f"{pt.y:.4f}, {pt.z:.4f}" for pt in parsed.vertices])
            self.coords_text.delete("1.0", tk.END)
            self.coords_text.insert("1.0", new_coords)
            self.load_section_from_text()
            messagebox.showinfo("Skiss analyserad", parsed.explanation)
        except Exception as e:
            messagebox.showerror("Fel vid bildanalys", f"Kunde inte tolka skissen:\n{e}")

    def on_preset_change(self, _):
        choice = self.preset_cb.get()
        self.coords_text.delete("1.0", tk.END)
        self.coords_text.insert("1.0", self.presets[choice])
        self.load_section_from_text()

    def load_section_from_text(self):
        raw = self.coords_text.get("1.0", tk.END).strip()
        pts = []
        try:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(";", ",").split(",")
                pts.append((float(parts[0].strip()), float(parts[1].strip())))
            self.section = CrossSectionNavier(pts)
        except Exception as e:
            messagebox.showerror("Fel i tvärsnittsdata", f"Kunde inte tolka koordinaterna:\n{e}")
            return

        s = self.section
        pad = 0.25 * max(np.ptp(s.pts[:, 0]), np.ptp(s.pts[:, 1]))
        self.y_min, self.y_max = np.min(s.pts[:, 0]) - pad, np.max(s.pts[:, 0]) + pad
        self.z_min, self.z_max = np.min(s.pts[:, 1]) - pad, np.max(s.pts[:, 1]) + pad

        self.yy, self.zz = np.meshgrid(
            np.linspace(self.y_min, self.y_max, 180),
            np.linspace(self.z_min, self.z_max, 180)
        )
        grid_pts = np.column_stack([self.yy.ravel(), self.zz.ravel()])
        self.inside_mask = Path(s.pts).contains_points(grid_pts).reshape(self.yy.shape)

        self.update_plot()

    def update_plot(self):
        if not hasattr(self, "section"):
            return

        s = self.section
        N = self.N_var.get()
        M_y = self.My_var.get()
        M_z = self.Mz_var.get()

        sigma_0, c_y, c_z = s.calculate_stress(N, M_y, M_z)
        v_stress = sigma_0 + c_y * s.pts[:, 0] + c_z * s.pts[:, 1]
        s_max, s_min = np.max(v_stress), np.min(v_stress)

        e_str = f"({M_z/N:+.3f}, {M_y/N:+.3f}) m" if abs(N) > 1e-6 else "Oändlig (N = 0)"
        status = "DRAGZON FÖRELIGGER (sigma > 0)" if s_max > 1e-6 else "Hela tvärsnittet i TRYCK"
        info = (
            f"Area A:       {s.A:.4f} m²\n"
            f"TP (y, z):    ({s.y_tp:.3f}, {s.z_tp:.3f}) m\n"
            f"I_zz:         {s.I_zz:.5e} m⁴\n"
            f"I_yy:         {s.I_yy:.5e} m⁴\n"
            f"I_yz:         {s.I_yz:.5e} m⁴\n"
            f"{'-'*34}\n"
            f"Angreppspkt:  {e_str}\n"
            f"sigma_max:    {s_max:+.2f} kPa\n"
            f"sigma_min:    {s_min:+.2f} kPa\n"
            f"Status:\n  {status}"
        )
        self.res_label.config(text=info, fg="#b30000" if s_max > 1e-6 else "#1b611b")

        self.ax1.clear()
        self.ax2.clear()

        poly_c = np.vstack([s.pts, s.pts[0]])
        hull_c = np.vstack([s.hull_pts, s.hull_pts[0]])
        core_c = np.vstack([s.core_pts, s.core_pts[0]])

        self.ax1.fill(poly_c[:, 0], poly_c[:, 1], color="#d9e3f0", alpha=0.7, label="Tvärsnitt")
        self.ax1.plot(poly_c[:, 0], poly_c[:, 1], color="#1f3a60", lw=2)
        self.ax1.plot(hull_c[:, 0], hull_c[:, 1], "k--", lw=1, alpha=0.4, label="Konvext hölje")
        self.ax1.fill(core_c[:, 0], core_c[:, 1], color="#2ca02c", alpha=0.4, label="Kärngräns (Kärna)")
        self.ax1.plot(core_c[:, 0], core_c[:, 1], color="#1b611b", lw=2)
        self.ax1.plot(0, 0, "ko", markersize=5, label="TP (0,0)")

        if abs(N) > 1e-6:
            e_y, e_z = M_z / N, M_y / N
            self.ax1.plot(e_y, e_z, "rX", markersize=10, markeredgewidth=2, label="Kraftangrepp (e)")

        self.ax1.axhline(0, color="gray", lw=0.7, ls=":")
        self.ax1.axvline(0, color="gray", lw=0.7, ls=":")
        self.ax1.set_aspect("equal")
        self.ax1.set_xlim(self.y_min, self.y_max)
        self.ax1.set_ylim(self.z_min, self.z_max)
        self.ax1.set_title("Tvärsnitt & Kärngräns")
        self.ax1.set_xlabel("y [m]")
        self.ax1.set_ylabel("z [m]")
        self.ax1.legend(loc="upper right", fontsize=8)
        self.ax1.grid(True, alpha=0.3)

        sigma_grid = sigma_0 + c_y * self.yy + c_z * self.zz
        sigma_masked = np.where(self.inside_mask, sigma_grid, np.nan)

        abs_max = max(abs(s_min), abs(s_max), 1.0)
        mesh = self.ax2.pcolormesh(self.yy, self.zz, sigma_masked, cmap="coolwarm",
                                   vmin=-abs_max, vmax=abs_max, shading="auto")

        if self.cbar is None:
            self.cbar = self.fig.colorbar(mesh, ax=self.ax2, fraction=0.046, pad=0.04)
            self.cbar.set_label("Spänning [kPa] (+ Drag / - Tryck)")
        else:
            self.cbar.update_normal(mesh)

        self.ax2.plot(poly_c[:, 0], poly_c[:, 1], color="black", lw=2)

        if s_max > 1e-6:
            self.ax2.contourf(self.yy, self.zz, sigma_masked,
                              levels=[0.0, max(s_max * 1.05, 1e-3)],
                              colors=["none"], hatches=["///"], extend="max")
            self.ax2.fill([], [], facecolor="none", edgecolor="darkred", hatch="///", label="Dragzon (σ > 0)")

        if np.hypot(c_y, c_z) > 1e-12:
            if abs(c_z) >= abs(c_y):
                nl_y = np.array([self.y_min, self.y_max])
                nl_z = -(sigma_0 + c_y * nl_y) / c_z
            else:
                nl_z = np.array([self.z_min, self.z_max])
                nl_y = -(sigma_0 + c_z * nl_z) / c_y
            self.ax2.plot(nl_y, nl_z, "k--", lw=2, label="Neutrala lagret (σ = 0)")

        self.ax2.axhline(0, color="gray", lw=0.7, ls=":")
        self.ax2.axvline(0, color="gray", lw=0.7, ls=":")
        self.ax2.set_aspect("equal")
        self.ax2.set_xlim(self.y_min, self.y_max)
        self.ax2.set_ylim(self.z_min, self.z_max)
        self.ax2.set_title("Spänningsfördelning & Dragzon")
        self.ax2.set_xlabel("y [m]")
        self.ax2.set_ylabel("z [m]")
        self.ax2.legend(loc="upper right", fontsize=8)
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = NavierApp()
    app.mainloop()
