
Gemini

Ny chatt
Bibliotek
Planerad
Studenter

Mer
Ny anteckningsbok
Crack Control for Early Thermal Stress in Massive Waterworks
Designing Reinforced Concrete Frame Corners for Bending Moments
Finite Element Appraisal of Concrete Strut and Tie Methods
Alla anteckningsböcker
TBE-dagbok och aktivitetslogg
Motsägelser i Eurokod 2 om SLS
Brigade plus och ska simulera krympning för betong
Mobilanpassa kärngränsapp via Streamlit
Varning för Jelvora Stockholm
Oklar start på konversation
Löprundor och träningsanalys
Makita DHR243 för borrning i berg
3D-karta över SL-trafiken
Bilars olika värdeminskning efter år
Hitta tillverkningsnummer för golvbyte
Interaktiv svensk makrokalkylator
Ny mixerkanna
Initiating Phone Call
Förklaring av Z-Library-länk
Koppla in USB för lagring
Skillnader mellan Lexus NX 450h+-årsmodellerna
Vad lönar sig vid bilköp?
BMW iX1 vs X1
Namnlös
Dosering för 25-liters dunk
Skapande av retro affischprompter
Billån eller sparande?
Starta din läranderesa
Sammanfattning av träningspass
Prisbedömning ej möjlig
Oklart kontextunderlag förfrågan
Effektiv E-modul och krympning
Vinterkonservering av Evinrude fyrtakt
Prisfråga om Kia EV6
Moms- och beräkningskontroll för faktura

Aktivitet
Hägersten, Sverige
Utifrån dina platser (hem)
Konversation med Gemini
jag vill få tillgång till detta program via mobilen, sen vill jag ta bort funktionen att den ska kunna läsa av en handskiss, jag har github om det hjälper

För att du ska kunna köra programmet på mobilen behöver gränssnittet bytas ut från Tkinter (som kräver en lokal datorskärm) till Streamlit (som redan finns med i din requirements.txt). Eftersom du har GitHub kan du ladda upp koden där och koppla den till Streamlit Community Cloud helt gratis – då får du en webblänk som du kan öppna i mobilen och spara som en app-ikon på hemskärmen.   
PY
+ 1

Här är den kompletta lösningen där AI-skissfunktionen är helt borttagen och gränssnittet är mobilanpassat.

1. Uppdaterad requirements.txt
Rensa bort de onödiga AI-paketen (pillow, google-genai, pydantic) så att appen startar snabbare i molnet. Din requirements.txt ska bara innehålla:   
TXT

Plaintext
streamlit
numpy
scipy
matplotlib
2. Ny programkod (app.py eller KG.py)
Ersätt innehållet i din Python-fil (KG.py) med koden nedan. Beräkningsmotorn (CrossSectionNavier) är exakt densamma som tidigare, men gränssnittet är ombyggt för Streamlit och pekfingrar:   
PY
+ 1

Python
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
3. Så publicerar du appen till mobilen via GitHub
Lägg upp filerna på GitHub:

Logga in på ditt GitHub-konto och skapa ett nytt repository (t.ex. karngrans-app).

Ladda upp din uppdaterade KG.py och requirements.txt till detta repository.

Koppla till Streamlit Community Cloud:

Gå till share.streamlit.io och logga in med ditt GitHub-konto.

Klicka på "Create app" (eller "New app").

Välj ditt repository (karngrans-app), branch (main) och skriv in KG.py under Main file path.

Klicka på "Deploy!".

Öppna på mobilen:

Efter ca 1–2 minuter är appen klar och du får en egen länk (t.ex. [https://din-app.streamlit.app](https://din-app.streamlit.app)).

Öppna länken i webbläsaren på mobilen (Safari eller Chrome) och välj "Lägg till på hemskärmen" i menyn, så fungerar den precis som en vanlig mobilapp.

Du är på precis rätt ställe i Streamlit Community Cloud! Fyll i fälten på skärmen enligt stegen nedan för att publicera din app.   
PNG

Repository: Klicka i rutan där det står kransengangster1000/repo och välj det GitHub-repository där du har lagt upp dina filer (om det inte dyker upp i listan kan du klicka på Paste GitHub URL uppe till höger och klistra in länken direkt från GitHub). Du ser att steget lyckats när ditt riktiga repo-namn visas i fältet.   
PNG
+ 1

Branch: Klicka i rutan där det står master och kontrollera att rätt gren är vald. Nya repon på GitHub heter nästan alltid main istället för master. Rätt branch brukar dyka upp automatiskt i listan så fort du har valt ditt repository i steg 1.   
PNG
+ 1

Main file path: Radera streamlit_app.py och skriv in KG.py (eller det exakta filnamnet du gav din Python-fil på GitHub, tänk på stora och små bokstäver). Du vet att det är rätt när fältet inte längre visar någon röd varningstext om att filen saknas.   
PNG
+ 1

App URL (optional): Här kan du skriva in ett eget kort namn, exempelvis karngrans, så blir din webbadress [https://karngrans.streamlit.app](https://karngrans.streamlit.app). Lämnar du fältet tomt skapar Streamlit en adress automatiskt.   
PNG
+ 1

Starta appen: Klicka på den blå knappen Deploy längst ner. Sidan kommer nu att byta vy och visa en laddningsruta ("Your app is in the oven") och efter cirka 1–2 minuter dyker dina grafer och beräkningsfält upp direkt på skärmen.   
PNG

Har du redan hunnit ladda upp den nya koden och requirements.txt till ditt GitHub-konto, eller vill du ha vägledning för hur du skapar repot och lägger in filerna där först?   
TXT

Den röda varningen "This file does not exist" under Main file path beror på att Streamlit letar efter standardfilnamnet streamlit_app.py, medan din fil på GitHub heter KG.py (eller det namn du gav den vid uppladdningen).   
PNG
+ 1

Följ dessa steg för att rätta till filnamnet och starta appen:

Ändra Main file path: Klicka i den rödmarkerade rutan där det står streamlit_app.py, radera texten och skriv in KG.py (var noga med stora bokstäver KG och små bokstäver .py om filen heter så på GitHub). Du ser att steget lyckats när den röda ramen och texten "This file does not exist" försvinner och knappen Deploy längst ner blir blå. (Tips: När du raderar texten i rutan brukar Streamlit visa en rullgardinsmeny med de Python-filer som finns i ditt repo Kransengangster1000/Corearea – du kan klicka direkt på rätt fil i listan).   
PNG
+ 3

Förenkla webbadressen (Valfritt): I rutan App URL (optional) kan du byta ut den långa slumpade texten corearea-izpswhybknfuzybe7jecar mot till exempel corearea eller karngrans. Du ser att adressen fungerar om texten "Domain is available" under rutan förblir grön.   
PNG
+ 1

Publicera appen: Klicka på knappen Deploy längst ner. Sidan övergår nu till en laddningsvy där Streamlit installerar paketen från requirements.txt, och inom ett par minuter visas din färdiga beräkningsapp.   
PNG
+ 1

Om varningen "This file does not exist" fortfarande visas trots att du skriver KG.py, kontrollera inne på GitHub i ditt repo Corearea att filerna verkligen är sparade (committade) i branchen main. Hur ser det ut när du byter ut texten i rutan?   
PNG
+ 3

Felmeddelandet beror på att filen KG.py på ditt GitHub-konto fortfarande innehåller den gamla koden som börjar med import tkinter as tk. Eftersom Streamlit-servern i molnet inte har någon lokal datorskärm för Tkinter kraschar programmet direkt på rad 1.   
PY
+ 2

Du behöver bara byta ut innehållet i KG.py direkt inne på GitHub mot den nya Streamlit-koden, så uppdaterar sig appen automatiskt:

Öppna filen på GitHub: Gå till ditt repository Kransengangster1000/Corearea på GitHub och klicka på filen KG.py. Du ser att du är rätt när du ser koden med import tkinter as tk högst upp.   
PNG
+ 4

Gå till redigeringsläget: Klicka på penn-ikonen (Edit this file) uppe till höger ovanför kodrutan. Textfältet blir nu redigerbart med en blinkande markör.

Byt ut koden: Markera all gammal kod i rutan (Ctrl + A), radera den, och klistra in den nya Streamlit-koden (som börjar med import streamlit as st från mitt första svar). Kontrollera att rad 1 nu säger import streamlit as st och att tkinter är helt borta.

Spara ändringen på GitHub: Klicka på den gröna knappen Commit changes... uppe till höger, och bekräfta genom att klicka på Commit changes i rutan som dyker upp. Du ser att ändringen är sparad när du kommer tillbaka till filvyn och rad 1 visar import streamlit as st.

Se appen ladda om: Gå tillbaka till fliken med din Streamlit-app. Så fort du har sparat på GitHub känner Streamlit av ändringen automatiskt inom några sekunder, det röda felmeddelandet försvinner och din interaktiva beräkningsapp dyker upp på skärmen.   
PNG
+ 1

(Tips: Om du inte redan har rensat requirements.txt på GitHub kan du göra likadant med den filen så att det bara står streamlit, numpy, scipy och matplotlib i den).   
TXT

Startar appen upp som den ska när du har klistrat in den nya koden och klickat på Commit changes?

kan du skapa båda filerna

Här är båda filerna färdiga att ladda ner och lägga upp på ditt GitHub-konto:

Ikon för PY
KG
 PY 
Ikon för TXT
requirements
 TXT 
Så lägger du in filerna på GitHub
Ladda ner filerna: Klicka på de två filerna ovan för att spara ner KG.py och requirements.txt på din dator. (Kontrollera att din dator inte lägger till (1) i filnamnet, t.ex. KG (1).py, om du redan har gamla filer med samma namn i mappen Hämtade filer).   
PY
+ 1

Öppna uppladdningen på GitHub: Gå till ditt repository Kransengangster1000/Corearea på GitHub, klicka på knappen Add file uppe till höger ovanför fillistan och välj Upload files.   
PNG

Dra in och ersätt filerna: Dra in dina två nedladdade filer (KG.py och requirements.txt) till uppladdningsrutan och klicka på den gröna knappen Commit changes längst ner. Du ser att steget lyckats när GitHub skriver över de gamla versionerna och visar meddelandet om att filerna nyss uppdaterats ("now").   
PY
+ 1

Kontrollera Streamlit-appen: Gå tillbaka till fliken med din Streamlit-app (och ladda om sidan med F5 om den inte startar om av sig själv). Det röda felmeddelandet försvinner nu och ersätts av din färdiga beräkningsapp.   
PNG

snyggt, jag vill kunna ha flera lastfall, skriva ut pdf:er. har du några förslag som är värda att lägga till. skapa inget nytt skript förens jag säger till

Att bygga in flera lastfall och PDF-export förvandlar verktyget från en enkel kalkylator till ett riktigt konstruktionshjälpmedel. Här är ett förslag på hur de två funktionerna kan utformas, följt av några skarpa tillägg som gör stor nytta i praktisk dimensionering:

1. Upplägg för dina önskade funktioner
Flera lastfall (Interaktiv tabell & plottning): Istället för enstaka inmatningsrutor lägger vi in en redigerbar tabell (st.data_editor) där du kan lägga till rader för exempelvis LF1 (Bruks), LF2 (Brott) och LF3 (Upplyft/Is). I kärngränsgrafen plottas alla lastfalls angreppspunkter (
e 
y
​
 ,e 
z
​
 ) samtidigt med olika färger/etiketter så att du direkt ser vilka lastfall som hamnar utanför kärnan. Du väljer sedan i en rullgardinsmeny vilket lastfall (eller "Dimensionerande/Värsta") som ska visas i spänningskonturen till höger.

PDF-beräkningsrapport: En knapp (Ladda ner PDF-rapport) som genererar ett färdigt, snyggt beräkningsblad (kan skapas direkt i minnet med Matplotlibs PdfPages utan krångliga extrapaket). PDF:en innehåller projektinfo/datum, tvärsnittsdata (
A,y 
tp
​
 ,z 
tp
​
 ,I 
yy
​
 ,I 
zz
​
 ,I 
yz
​
 ), en sammanställningstabell över alla lastfall med tillhörande 
σ 
max
​
 , 
σ 
min
​
  och status, samt högupplösta figurer.

2. Förslag på funktioner som är värda att lägga till
Sprucket tvärsnitt (Noll-drag / No-tension analysis):
Naviers formel förutsätter att materialet kan ta drag. För betongfogar, murverk, dammsektioner eller fundament mot berg/jord vill man ofta veta vad som händer när kraften kliver utanför kärngränsen och fogen spricker upp. Vi kan lägga till en kryssruta: "Beräkna som ospänningslös dragzon (sprucket tvärsnitt)". Då itererar programmet fram det verkliga neutrala lagret där dragzonen är inaktiv, visar det nya (högre) maximala kanttrycket 
σ 
c,max
​
  och anger hur många procent av tvärsnittsarean som är i kontakt (tryckt).   
PY
+ 1

Huvudtröghetsaxlar & huvudtröghetsmoment (
I 
1
​
 ,I 
2
​
 ,α):
För osymmetriska tvärsnitt (som L-tvärsnittet där 
I 
yz
​
 

=0) är det guld värt att rita ut de vridna huvudaxlarna i figuren och skriva ut vinkeln 
α samt 
I 
1
​
  och 
I 
2
​
  i resultatlistan.   
PY
+ 1

Andra kärnan (1/3-kontakt / Stjälpningskontroll):
Utöver den vanliga kärngränsen (100 % tryckt tvärsnitt, 0 % glipa) kan man rita ut en streckad yttre gränslinje för exempelvis halva tvärsnittet i tryck (neutrala lagret går genom tyngdpunkten). Det ger en direkt visuell referens för hur nära stjälpning konstruktionen är vid extrema lastfall.

Hörn- och koordinattabell med spänningar:
En utfällbar tabell som listar varje enskilt hörn 
(y 
i
​
 ,z 
i
​
 ) och dess exakta spänning 
σ 
i
​
  för det valda lastfallet, samt markerar vilket hörn som får störst tryck respektive drag i grafen.

Spara/Ladda indata på mobilen (JSON/CSV eller URL-parametrar):
Eftersom en webbapp nollställs när du stänger fliken kan det vara skönt med en knapp som sparar ner ditt tvärsnitt och dina lastfall till en liten fil (eller låter dig klistra in från Excel), så att du slipper knappa in koordinaterna på nytt i mobilen.

Vilka av dessa tillägg (utöver flera lastfall och PDF-utskrift) vill du att vi tar med innan jag skriver koden?

allt

Du har stoppat svaret
allt, och det behöver inte anpassas till mobil men jag vill dela med mig av programmet

Här är de nya, kompletta filerna med samtliga funktioner inbyggda och optimerade för datorvy och enkel delning med kollegor:

Ikon för PY
KG
 PY 
[file-tag: code-generated-file-32e0684e-8d5a-4a12-a61e-32e0684e8d5a] (Uppdaterad med pandas för den interaktiva lastfallstabellen)

Vad som nu finns i programmet
Interaktiv lastfallstabell & flerpunktsplottning: Du kan lägga till, ta bort och ändra hur många lastfall du vill direkt i tabellen (och även kopiera/klistra in rader direkt från Excel). Alla lastfalls angreppspunkter 
(e 
y
​
 ,e 
z
​
 ) ritas ut samtidigt i kärngränsgrafen med etiketter, och du kan välja i en rullgardinsmeny vilket lastfall (eller Dimensionerande / Störst tryckspänning) som ska visas i detalj.

Sprucket tvärsnitt (No-Tension / 0 dragkapacitet): I vänstermenyn finns kryssrutan "Sprucket tvärsnitt (No-Tension / 0 drag)". När du slår på den itererar programmet fram jämviktsläget där dragzonen är inaktiv (
σ≤0), räknar ut det förhöjda kanttrycket 
σ 
min
​
  samt anger exakt hur många procent av tvärsnittsarean som är i kontakt. Om kraften hamnar utanför tvärsnittets konvexa hölje varnar programmet för stjälpning.

Huvudtröghetsaxlar & Andra kärnan: Programmet beräknar och ritar ut huvudtröghetsaxlarna (
I 
1
​
 ,I 
2
​
  och vinkeln 
α) samt "andra kärnan" (den orangea streckade gränslinjen där det spruckna tvärsnittets neutrala lager skär tyngdpunkten).

Komplett PDF-beräkningsrapport (A4): En knapp direkt under grafen genererar en 2-sidig PDF med projektnamn, handläggare, datum, alla tvärsnittskonstanter, sammanställningstabell för samtliga lastfall, högupplösta figurer samt en fullständig hörntabell med både Navier-spänning och sprucken spänning i varje hörn (H1, H2...).

Dela med kollegor (Länk + JSON-projektfiler): Alla som får din streamlit.app-länk kan använda programmet direkt i sin webbläsare utan att installera Python. Via sidopanelen kan ni dessutom spara ner ett helt beräkningsfall som en .json-fil och skicka till varandra för att ladda in exakt samma tvärsnitt och lastfall.

Uppdatera ditt program på GitHub
Ladda ner filerna: Klicka på filerna KG.py och requirements.txt högst upp i detta svar så att de sparas på din dator. Se till att de heter exakt KG.py och requirements.txt utan siffror som (1) eller _2 i filnamnet.   
PY
+ 3

Ladda upp till GitHub: Gå till ditt repository Kransengangster1000/Corearea på GitHub, klicka på Add file 
→ Upload files, dra in båda filerna och klicka på den gröna knappen Commit changes. Du ser att uppladdningen lyckats när fillistan visar att båda filerna uppdaterats alldeles nyss.   
PNG

Öppna och dela appen: Gå till din Streamlit-flik – den laddar nu automatiskt in den nya versionen inom cirka 30 sekunder. Kopiera webbadressen högst upp i webbläsaren (.streamlit.app) för att skicka programmet till vem du vill.   
PNG
+ 2

fattar inte pandas, ska det läggas till requirements

pandas ska ligga i requirements.txt, och det ligger redan färdigt i filen jag skapade åt dig i förra svaret.   
TXT
+ 1

Pandas är ett standardpaket i Python för att hantera tabeller. I det nya programmet (KG.py) används import pandas as pd för att skapa den interaktiva tabellen där du kan lägga till och ändra flera lastfall (st.data_editor) samt resultat- och hörntabellerna.   
PY
+ 1

Så här ska din requirements.txt se ut på GitHub (vilket är exakt vad som står i filen du laddar ner):   
TXT

Plaintext
streamlit
numpy
scipy
matplotlib
pandas
Du behöver alltså inte skriva in något själv om du laddar ner bägge filerna (KG.py och requirements.txt) från mitt förra svar och laddar upp dem till GitHub – då kommer Streamlit-servern automatiskt att installera pandas åt dig.

vad är andra kärnan

Andra kärnan (även kallad utvidgad kärna eller halva kontaktgränsen) är en stabilitetsgräns för spruckna tvärsnitt som saknar draghållfasthet (t.ex. betongfogar, dammar, murverk eller fundament mot berg och jord). Den visar var normalkraften 
N kan angripa när man tillåter att fogen glipar upp, men kräver att det neutrala lagret inte passerar tvärsnittets tyngdpunkt (TP).   
PY
+ 2

Så här hänger de tre gränserna i ditt program ihop inifrån och ut:

Första kärnan (Gröna ytan – "Kärngränsen", 100 % kontakt):
Om kraftresultanten angriper innanför den gröna kärngränsen är hela tvärsnittet i tryck (
σ≤0 överallt). Det neutrala lagret (
σ=0) ligger utanför eller precis längs tvärsnittets ytterkant. Det uppstår inget drag och ingen spricka/glipa.   
PY
+ 2

Andra kärnan (Orange streckad linje – ca 50 % kontakt / NL genom TP):
Om kraften flyttar sig utanför den gröna kärnan men hålls innanför den orangea andra kärnan, spricker tvärsnittet upp på dragsidan, men det neutrala lagret ligger fortfarande på "rätt sida" om tyngdpunkten (minst ungefär halva tvärsnittet är i tryck).   
PY
+ 2

För ett rektangulärt tvärsnitt med bredden 
b har den vanliga kärnan bredden 
b/3 (excentricitet 
e≤b/6), medan andra kärnan har bredden 
2b/3 (excentricitet 
e≤b/3). Vid exakt 
e=b/6 har du en triangelformad spänningsfördelning över 100 % av bredden; vid 
e=b/3 (på andra kärnan) har du en triangelformad spänningsfördelning över exakt 50 % av bredden, och det neutrala lagret skär mitt i tyngdpunkten.

Konvexa höljet / Ytterkanten (Svart streckad linje – Stjälpningsgräns, 0 % kontakt):
Om kraften går utanför andra kärnan minskar den tryckta ytan snabbt och kanttrycket skjuter i höjden. Går kraften ända ut till tvärsnittets ytterkant (det konvexa höljet) går den tryckta arean mot noll och konstruktionen stjälper.   
PY
+ 2

Varför är "Andra kärnan" så användbar i praktiken?
Inom grundläggning (Eurokod 7), dammsäkerhet (RIDAS) och betongkonstruktioner har man ofta två olika krav beroende på lastfall:

Bruksgränstillstånd / Normala lastfall: Kraften ska ligga inom första kärnan (gröna ytan) så att ingen glipa uppstår (100 % tryckt yta).   
PY
+ 2

Olyckslast / Extrema lastfall / Karaktäristisk last: Man tillåter att fogen glipar upp, men kräver ofta att kraften håller sig inom andra kärnan (orangea linjen), det vill säga att minst halva basytan (eller fram till TP) är i kontakt så att konstruktionen inte får okontrollerade kanttryck eller närmar sig stjälpning.   
PY
+ 2

I programmet räknas den orangea kurvan fram exakt för godtyckliga tvärsnittsformer genom att låta det neutrala lagret rotera 
360 
∘
  genom tyngdpunkten (TP), klippa bort dragdelen och beräkna var tryckresultanten 
(e 
y
​
 ,e 
z
​
 ) hamnar för varje vinkel.   
PY
+ 2

förklara jättetydligt detta "Sprucket tvärsnitt (No-Tension / 0 dragkapacitet)"

Sprucket tvärsnitt (No-Tension / 0 dragkapacitet) är en beräkningsmodell för konstruktioner där fogen eller materialet bara kan ta tryck och absolut inget drag (
σ≤0).   
PY

Det bästa sättet att förstå skillnaden mellan Linjär (Navier) och Sprucket tvärsnitt (No-Tension) är att tänka på en styv betongkloss som vilar löst ovanpå en plan berghäll (utan bergförankringar eller lim).

1. Problemet med Naviers formel (Linjär modell)
Naviers vanliga formel (
σ= 
A
N
​
 +c 
y
​
 y+c 
z
​
 z) utgår från att tvärsnittet är fastlimmat i underlaget och tål hur mycket drag som helst.

Så länge normalkraften 
N (tryckkraften) angriper innanför kärngränsen (den gröna ytan) trycks hela undersidan mot berget. Då stämmer Naviers formel perfekt.   
PY
+ 2

Men om du flyttar kraften utanför kärngränsen vill klossen tippa lite. Naviers formel säger då att det uppstår dragspänning (
σ>0) i bakkanten för att hålla ner klossen.   
PY
+ 1

I verkligheten kan en ospänd betongfog, en damm mot berg eller en sula på jord inte suga sig fast i underlaget. Bakkanten lyfter istället (det bildas en glipa / spricka med 
σ=0), och hela tryckkraften 
N måste bäras av den lilla del av ytan som fortfarande har kontakt med underlaget.   
PY
+ 2

2. Vad händer i ett "Sprucket tvärsnitt (No-Tension)"?
När du kryssar i "Sprucket tvärsnitt (No-Tension / 0 drag)" i programmet slutar koden räkna med dragspänningar och gör precis som verkligheten:   
PY
+ 2

Sprickan öppnar sig (
σ=0): Alla delar av tvärsnittet som skulle ha fått dragspänning kopplas bort (blir helt spänningslösa, markerade med röd skravering i grafen).   
PY
+ 2

Aktiv kontaktyta minskar (
A 
c
​
 <A): Bara den del av tvärsnittet som ligger på trycksidan om det neutrala lagret (
σ=0) är kvar och bär last.   
PY
+ 2

Två jämviktskrav styr allt: För att klossen ska stå i jämvikt på den krympta kontaktytan måste spänningskilen på trycksidan uppfylla två saker samtidigt:   
PY
+ 2

Kraftjämvikt: Volymen av tryckspänningskilen måste vara exakt lika stor som din pålagda normalkraft 
N (
∫ 
A 
c
​
 
​
 σdA=N).   
PY
+ 2

Momentjämvikt: Tyngdpunkten (resultanten) för spänningskilen måste ligga på exakt samma punkt 
(e 
y
​
 ,e 
z
​
 ) som din angripande kraft (
∫ 
A 
c
​
 
​
 σydA=M 
z
​
  och 
∫ 
A 
c
​
 
​
 σzdA=M 
y
​
 ).   
PY
+ 2

Eftersom ytan som bär lasten blir mindre samtidigt som kraften 
N är lika stor, händer två viktiga saker jämfört med Naviers formel:

Det neutrala lagret flyttar sig inåt i tvärsnittet (sprickan växer mer än vad den linjära dragzonen antydde).

Kanttrycket (
σ 
min
​
 ) ökar kraftigt i den tryckta kanten.

3. Ett konkret sifferexempel (Rektangel 
1,0×1,0 m)
Tänk dig ett kvadratiskt fundament med bredden 
b=1,0 m (area 
A=1,0 m 
2
 ) och en tryckkraft 
N=−100 kN.

Kärngränsen går vid excentriciteten 
e=b/6=0,167 m från mitten.

Vad händer om vi ställer kraften utanför kärnan, till exempel vid 
e=0,25 m från mitten (dvs. 
0,25 m från den tryckta högerkanten)?

Storhet	Linjär modell (Navier)	Sprucket tvärsnitt (No-Tension)
Tryckt bredd (Kontakt)	0,83 m (83,3 % av ytan)	0,75 m (75,0 % av ytan)
Spänning i bakkant	+50,0 kPa (Drag – omöjligt utan lim!)	0,0 kPa (Glipa på 0,25 m)
Max tryck i framkant (σ 
min
​
 )	−250,0 kPa	−266,7 kPa (Högre verkligt tryck!)
Varför blir kontaktbredden exakt 
0,75 m i det spruckna fallet?
I ett rektangulärt tvärsnitt bildar tryckspänningen en triangel från 
σ=0 (vid sprickspetsen) till 
σ 
min
​
  (vid framkanten). Tyngdpunkten i en triangel ligger alltid på 1/3 av triangelns bas räknat från den höga kanten.

Eftersom vår kraft står 
0,25 m från framkanten måste triangelns tyngdpunkt också ligga 
0,25 m från framkanten.

Då måste triangelns totala bredd (den tryckta längden 
x) vara 
3×0,25 m=0,75 m (75 % kontakt).

Och medelspänningen över de 
0,75 m 
2
  är 
−100/0,75=−133,3 kPa, vilket ger kanttrycket 
2×(−133,3)=−266,7 kPa.

Om du skulle flytta ut kraften ännu längre, till 
e=0,40 m från mitten (endast 
0,10 m från framkanten):

Navier påstår att kanttrycket bara är 
−340 kPa.

Sprucket tvärsnitt visar verkligheten: bara 
3×0,10=0,30 m (30 % av ytan) är i kontakt, och det verkliga kanttrycket blir hela 
−666,7 kPa (nästan dubbelt så högt som Navier visar!).

4. Hur löser ditt program detta för krångliga former (T- och L-tvärsnitt)?
För en enkel rektangel med böjning i en riktning kan man räkna med regeln 
3×avst 
a
˚
 ndet till kanten. Men för ett T-tvärsnitt, ett L-tvärsnitt eller vid dubbelböjning (
M 
y
​
  och 
M 
z
​
  samtidigt) är den tryckta ytan inte längre en rektangel, utan kan bli en sned femhörning eller triangel i ett hörn.   
PY
+ 2

Därför gör funktionen calculate_cracked i din kod följande:   
PY
+ 2

Den lägger ut ett tänkt neutralt lager (
σ 
0
​
 +c 
y
​
 y+c 
z
​
 z=0) över tvärsnittet.   
PY
+ 2

Den klipper rent geometriskt bort hela dragdelen med algoritmen clip_polygon_halfplane så att bara den faktiska tryckta polygonen återstår.   
PY
+ 2

Den integrerar exakt area och tröghetsmoment över den återstående polygonen (polygon_integrals) och justerar automatiskt lutningen och läget på det neutrala lagret tills normalkraften 
N och momenten 
M 
y
​
 ,M 
z
​
  stämmer på millimetern.   
PY
+ 2

hänger inte med här. i LF3 hamnar resultanten utanför basarean, borde inte hela ytan bli dragen då. enligt bilden finns tryckspänningar

Anledningen till att det fortfarande finns tryckspänningar i LF3 (blå färg nere till vänster vid H1) beror på två saker: dels är du just nu i läget Linjär (Navier) (kryssrutan "Sprucket tvärsnitt" uppe till vänster är inte ikryssad), och dels kan ett tvärsnitt aldrig bli 100 % draget så länge normalkraften är en tryckkraft (
N=−100 kN).   
JPG
+ 1

Här är förklaringen till varför det ser ut som det gör i de båda modellerna när kraften hamnar utanför basarean:

1. Varför kan inte hela ytan bli dragen när 
N=−100 kN?
Tänk på ren vertikal kraftjämvikt: du trycker neråt på tvärsnittet med 
N=−100 kN.   
JPG

Om hela ytan hade haft dragspänningar (
σ>0 överallt), skulle summan av spänningarna peka uppåt (en ren dragkraft 
N>0).   
PY
+ 1

För att jämvikten 
∫σdA=−100 kN ska gå ihop, måste det alltid finnas en tryckzon (blå) som är minst 
100 kN större än dragzonen (röd).   
PY
+ 3

2. Vad händer i läget "Linjär (Navier)" (som är valt på din skärmbild)?
I Naviers formel antas tvärsnittet sitta på en fast inspänd balk som tål hur mycket drag och tryck som helst.   
PY
+ 2

Att det röda krysset (LF3) ligger nere till vänster utanför profilen betyder bara att du lägger på en tryckkraft på 
−100 kN via en tänkt "hävarm" som sticker ut till punkten 
(−0,11,−0,30) m.   
JPG

Eftersom hävarmen trycker neråt nere till vänster, trycks det nedre vänstra hörnet (H1) ihop hårt (
σ 
min
​
 =−7853 kPa, blått) samtidigt som den övre och högra delen (H5, H3) dras uppåt (
σ 
max
​
 =+5575 kPa, rött/skraffering).   
JPG

Tillsammans bildar det stora trycket nere vid H1 och det stora draget uppe vid H5 ett kraftpar (böjande moment) som balanserar den utflyttade kraften, medan det neutrala lagret (
σ=0, streckad linje) går nästan genom tvärsnittets mitt.   
JPG

3. Vad händer om du kryssar i "Sprucket tvärsnitt (No-Tension / 0 drag)"?
Om tvärsnittet istället är en ospänd fog eller ett fundament mot mark/berg som inte kan ta drag, händer det du tänker på – nämligen att tvärsnittet tappar kontakten och stjälper när kraften går utanför ytterkanten:   
PY
+ 2

Så fort kraften närmar sig den nedre/vänstra ytterkanten (vid H1) och du inte har någon dragkraft i bakkant (H3/H5) som kan hålla emot, krymper den tryckta blå zonen ihop mot själva hörnet H1.

När angreppspunkten passerar ytterkanten (det konvexa höljet, precis som LF3 gör i din bild), blir den tryckta arean 0 % – hela bakkanten lyfter och klossen tippar över kanten H1.   
JPG
+ 3

Testa att klicka i kryssrutan "Sprucket tvärsnitt (No-Tension / 0 drag)" ute i vänstra sidopanelen nu när du har LF3 valt – då kommer programmet direkt att slå larm med en röd varningsruta: "Sprucket (Stjälpning!): Kraftangreppspunkten ligger utanför tvärsnittets konvexa hölje – jämvikt utan drag är omöjlig" och sätta aktiv tryckt area till 0 %.   
JPG
+ 3

Vill du att programmet även i det linjära Navier-läget ska visa en tydlig notis när kraftangreppspunkten hamnar helt utanför tvärsnittets ytterkontur?

men stämmer den högra figuren med angiven sprucken zon

Nej, den högra figuren stämmer inte fysikaliskt för det spruckna läget när stjälpning inträffar – och det är väldigt skarpt fångat av dig!

Varför visar den högra figuren fel i LF2?
I LF2 ligger kraftangreppspunkten (
e 
y
​
 =−0,20 m,e 
z
​
 =−0,12 m) utanför tvärsnittets ytterkant (H1 ligger vid 
y=−0,1417 m).   
JPG

Programmet upptäcker helt rätt att det sker stjälpning och sätter Tryckt area [%] = 0 i tabellen samt skriver ut den röda varningsrutan.   
JPG

Felet i koden: När stjälpning sker och ingen sprucken jämvikt existerar (converged = False), råkade koden falla tillbaka på de linjära Navier-värdena (**lin) men satte ändå etiketten i legenden till "Sprucken zon (σ = 0)". Därför ritar den högra grafen ut den vanliga linjära Navier-spänningen (med både blått tryck 
−6921 kPa och rött drag 
+4732 kPa) trots att rubriken säger Sprucket (Stjälpning!).   
PY
+ 4

Jämför med LF1 i din tabell – där fungerar det spruckna tvärsnittet!
Titta på raden för LF1 i din sammanställningstabell på skärmbilden:   
JPG

I LF1 ligger angreppspunkten vid 
e 
y
​
 =−0,1333 m (precis innanför vänsterkanten H1 som ligger vid 
−0,1417 m).   
JPG

Där har det spruckna tvärsnittet hittat den verkliga jämvikten: 9,1 % av ytan (ett litet hörn nere vid H1) är i kontakt, 
σ 
max
​
 =0 kPa (ingen dragspänning någonstans!) och kanttrycket i H1 har skjutit i höjden till 
−42313 kPa.   
JPG

Om du byter rullgardinsmenyn under punkt 3 till LF1, kommer du att se en helt korrekt sprucken figur där 90,9 % av tvärsnittet är vitt/skrafferat (
σ=0) och bara en smal blå remsa längst ner vid H1 bär hela lasten.

Hur ska vi rätta till visningen vid stjälpning (LF2 och LF3)?
När kraften hamnar utanför tvärsnittets ytterkant i läget "Sprucket tvärsnitt" (dvs. Kontakt = 0 % / Stjälpning) bör programmet inte rita ut Naviers spänningsfält och kalla det för sprucket. Istället bör det göra följande:   
PY
+ 4

I den högra grafen: Skraffera hela tvärsnittet (100 %) som sprucket/glipa, markera tippkanten (det hörn som ligger närmast kraften, här linjen vid H1) och skriva en tydlig text mitt i figuren: "STJÄLPNING (0 % kontaktyta – jämvikt utan drag saknas)".

I tabellerna: Visa Ej jämvikt (Stjälpning) eller None / - på spänningarna i kolumnen för sprucken modell, så att man inte luras att tro att 
−6921 kPa och 
+4732 kPa är spruckna spänningar.   
JPG

Vill du att jag uppdaterar KG.py så att den högra grafen och tabellerna visar stjälpning på det sättet direkt?

ja

Här är den uppdaterade KG.py där både grafen, tabellerna och PDF-rapporten hanterar stjälpning helt fysikaliskt korrekt:

Ikon för PY
KG
 PY 
(Din requirements.txt på GitHub är redan rätt och behöver inte laddas upp igen).   
TXT

Vad som är ändrat vid stjälpning i läget "Sprucket tvärsnitt"
Högra grafen: När angreppspunkten kliver utanför tvärsnittets ytterkant (det konvexa höljet, som i ditt LF2 och LF3) ritas inte längre det linjära Navier-fältet upp av misstag. Istället skrafferas hela tvärsnittet (100 %) som sprucket/glipa, en tydlig röd ruta visas mitt i tvärsnittet med texten "STJÄLPNING! (0 % kontaktyta – jämvikt utan drag existerar ej)", och det hörn som konstruktionen tippar över (t.ex. H1) markeras som Tippkant / Stjälphörn.   
JPG

Sammanställnings- och hörntabellen: Istället för att visa vilseledande Navier-spänningar vid stjälpning står det nu Stjälpning (Ej jämvikt) under 
σ 
min
​
  och 
σ 
max
​
 , och i hörntabellen ser du tydligt skillnaden mellan vad en fast inspänd balk hade fått (σ (Navier)) och vad det ospända tvärsnittet får (Ej jämvikt (Stjälpning) / 100 % glipa).

Ladda ner KG.py ovan, kontrollera att den inte fått ett extra nummer i filnamnet (som KG (1).py), och ladda upp den till Kransengangster1000/Corearea på GitHub med Add file 
→ Upload files 
→ Commit changes.   
PNG


Gränsen återställs kl. 11:05 em
Se användning
Uppgradera



Gemini är AI och den kan göra misstag, inklusive om personer. Din integritet och GeminiÖppnas i ett nytt fönster

Bearbetad och analyserad i kod
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
KG.py
Visar KG.py.