#!/usr/bin/env python3
"""
Soporte nuevo del FR5: columna con cabezal en T y dos "hombros" a 45° (27-oct)
==============================================================================
    python fairino_description/mjcf/decoracion/crear_soporte_hombros.py

GENERA decoracion/soporte_hombros.xml (lo incluye fairino_mjcf.xml) a partir de las
MEDIDAS de abajo, y AJUSTA en fairino_mjcf.xml la pose de <body name="base_link">
para que el FR5 quede atornillado en la placa del HOMBRO DERECHO.

Cómo está colocado (la idea clave para no romper [3] [4] [5]):
  · El HOMBRO del FR5 (eje de joint2) se queda EXACTAMENTE donde estaba con la peana:
    (0, 0, 0.155). La base gira 45° hacia fuera y hacia abajo alrededor de ese punto,
    y el soporte (y el suelo) se colocan alrededor. Todo lo que el programa coloca en
    coordenadas del mundo (peana y techo del taladro, mesa, límites de altura...) sigue
    a la misma distancia del hombro del robot.
  · Solo visual: geoms estáticos SIN colisión (contype/conaffinity = 0, grupo 2), sin
    bodies, sin masa → no cambian qpos, ids de cuerpos, mocap ni la física.

Si cambiáis MEDIDAS (por ejemplo con las del plano), ejecutad el script otra vez: rehace
el soporte y la base del brazo. Imprime la nueva altura del SUELO; si cambia, hay que
ponerla también en config.py (SOPORTE_SUELO_Z) y en las z marcadas con [H] / [Z] de
fairino_mjcf.xml, decoracion/deco_escena_base.xml y decoracion/deco_taladro_peana.xml.
"""
import math
import os
import re

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_XML = os.path.join(AQUI, "soporte_hombros.xml")
RUTA_MJCF = os.path.join(os.path.dirname(AQUI), "fairino_mjcf.xml")

# ══════════════════════════════════════════════════════════════════════════════
# MEDIDAS (m) — estimadas de la foto y los vídeos; cambiadlas por las del plano
# ══════════════════════════════════════════════════════════════════════════════
MEDIDAS = dict(
    altura_total   = 1.40,    # suelo → cara de ARRIBA de la placa superior
    # ruedas giratorias (Ø100) bajo las 4 esquinas del bastidor
    rueda_r        = 0.050,
    rueda_ancho    = 0.032,
    rueda_alto     = 0.125,   # suelo → cara de abajo del bastidor
    # bastidor de tubo cuadrado
    bastidor_x     = 1.00,    # a lo largo de los hombros (eje X del mundo)
    bastidor_y     = 0.80,    # de delante a atrás
    tubo           = 0.050,
    pletina_x      = 0.20,    # pletina de delante a atrás sobre la que se atornilla la columna
    pletina_esp    = 0.008,
    pie            = (0.24, 0.22, 0.015),   # placa del pie de la columna
    # columna (tubo rectangular)
    columna_x      = 0.12,
    columna_y      = 0.15,
    ventanas       = ((0.50, 0.18, 0.075), (0.78, 0.12, 0.075)),  # (altura del centro, alto, ancho)
    # cabezal: placa de arriba + dos alas con placa de montaje inclinada
    cuello         = 0.07,    # de la arista de arriba de las alas a la cara de arriba de la placa superior
    placa_sup      = 0.22,    # lado de la placa de arriba
    placa          = 0.22,    # lado de las placas de los hombros
    placa_esp      = 0.012,
    agujero_r      = 0.045,   # agujero central (paso de cables)
    tal_margen     = 0.022,   # taladros de las esquinas, desde el borde
    tal_r          = 0.0045,
    gusset_esp     = 0.006,   # chapas triangulares de los lados de cada ala
    hombro_grados  = 45.0,    # inclinación de las placas de los hombros
    giro_base_grados = 0.0,   # 9-nov: giro de la base del FR5 sobre la placa (eje de joint1), como
                              # está atornillada de verdad; medirlo con diagnosticos/ajustar_gemelo.py
    disco          = (0.085, 0.006),   # adaptador entre placa y base del FR5 (radio, espesor)
)

# Colocación de la base: HOMBRO_MUNDO y BASE_A_HOMBRO son las del DISEÑO del soporte (con el
# d1 = 155 mm del modelo antiguo). NO cambiar: fijan dónde está la base (atornillada en la placa).
# 5-nov: el d1 REAL del FR5 v6 es D1_FR5 = 152 mm (está en fairino_mjcf.xml, upperarm_Link);
# con la base en el mismo sitio, el hombro real queda 3 mm más cerca de ella.
HOMBRO_MUNDO = (0.0, 0.0, 0.155)
BASE_A_HOMBRO = 0.155
D1_FR5 = 0.152

# Colores: "elecnor" = columna y alas azul corporativo, chapas grafito, acentos naranja
#          "real"    = acero negro, bastidor galvanizado y ruedas rojas (como en los vídeos)
COLORES = "elecnor"
LOGOS = True               # logos verticales en la columna (Elecnor delante, I+D+i detrás)


# ══════════════════════════════════════════════════════════════════════════════
def _fmt(*v):
    return " ".join(f"{(0.0 if abs(x) < 5e-9 else x):.5g}" for x in v)


def _add(a, b, k=1.0):
    return tuple(a[i] + k * b[i] for i in range(3))


def _quat_y(ang):
    return (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)


def _mat2quat(R):
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = ((R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s)
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = ((R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s)
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = ((R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def geometria(M=MEDIDAS):
    """Puntos clave en el MUNDO. El hombro derecho del soporte está en -X (la derecha
    del robot, que mira hacia -Y)."""
    th = math.radians(M["hombro_grados"])
    s, c = math.sin(th), math.cos(th)
    n = (-s, 0.0, -c)                       # normal de la placa derecha: hacia fuera y abajo
    u_in = (c, 0.0, -s)                     # a lo largo de la placa, hacia la columna y abajo
    base = _add(HOMBRO_MUNDO, n, -BASE_A_HOMBRO)
    e_d = M["disco"][1] if M["disco"] else 0.0
    cara = _add(base, n, -e_d)              # cara EXTERIOR de la placa (donde apoya el disco)
    cara_in = _add(cara, n, -M["placa_esp"])  # cara interior = hipotenusa del ala
    L = M["placa"]
    I = _add(cara_in, u_in, L / 2)          # esquina de abajo, contra la columna
    O = _add(cara_in, u_in, -L / 2)         # esquina de arriba, por fuera
    cx2 = M["columna_x"] / 2
    x_col = I[0] + cx2                      # cara derecha de la columna en x = I[0]
    z_o = O[2]                              # arista de arriba de las alas
    z_top = z_o + M["cuello"]
    z_suelo = z_top - M["altura_total"]
    # Base del FR5: z = n (eje de joint1) · x = +Y (atrás). Con joint2 = 0 el brazo apunta a
    # h(q1) = -cos(q1)·x - sin(q1)·y: joint1 = 0 → brazo al FRENTE (-Y), +90° → hacia fuera
    # y arriba, -90° → hacia dentro y abajo. El salto de ±175° de joint1 queda mirando hacia
    # ATRÁS, la dirección que el brazo humano casi no usa.
    x = (0.0, 1.0, 0.0)
    y = (n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0])
    # 9-nov: base girada sobre la placa (alrededor de n = eje de joint1): x' = x·cos + y·sin
    g = math.radians(M.get("giro_base_grados", 0.0))
    if g:
        x, y = (tuple(math.cos(g) * x[i] + math.sin(g) * y[i] for i in range(3)),
                tuple(-math.sin(g) * x[i] + math.cos(g) * y[i] for i in range(3)))
    R = [[x[0], y[0], n[0]], [x[1], y[1], n[1]], [x[2], y[2], n[2]]]
    return dict(th=th, n=n, u_in=u_in, base=base, cara=cara, cara_in=cara_in, I=I, O=O,
                x_col=x_col, z_o=z_o, z_top=z_top, z_suelo=z_suelo, base_quat=_mat2quat(R))


PALETAS = {
    "elecnor": dict(
        columna=('0.0 0.278 0.6 1', 'specular="0.5" shininess="0.6"'),
        chapa=('0.20 0.22 0.25 1', 'specular="0.6" shininess="0.6"'),
        bastidor=('0.20 0.22 0.25 1', 'specular="0.6" shininess="0.6"'),
        rueda=('1.0 0.498 0.0 1', 'specular="0.3" shininess="0.4"'),
        acento=('1.0 0.498 0.0 1', 'specular="0.4" shininess="0.5"'),
        herraje=('0.62 0.63 0.66 1', 'specular="0.8" shininess="0.7"'),
        hueco=('0.03 0.03 0.035 1', 'specular="0"'),
    ),
    "real": dict(
        columna=('0.15 0.15 0.16 1', 'specular="0.5" shininess="0.5"'),
        chapa=('0.24 0.23 0.22 1', 'specular="0.6" shininess="0.6"'),
        bastidor=('0.71 0.73 0.75 1', 'specular="0.8" shininess="0.7" reflectance="0.05"'),
        rueda=('0.80 0.12 0.12 1', 'specular="0.3" shininess="0.4"'),
        acento=('0.42 0.30 0.22 1', 'specular="0.3" shininess="0.3"'),
        herraje=('0.55 0.56 0.58 1', 'specular="0.8" shininess="0.7"'),
        hueco=('0.03 0.03 0.035 1', 'specular="0"'),
    ),
}


def generar(M=MEDIDAS):
    g = geometria(M)
    X0, Z0 = g["x_col"], g["z_suelo"]
    W = lambda x, y, z: (X0 + x, y, Z0 + z)          # local (columna, suelo) → mundo
    geoms, mallas = [], []
    A = 'contype="0" conaffinity="0" group="2"'

    def com(texto):
        geoms.append(f"<!-- {texto} -->")

    def box(nombre, pos, half, mat, quat=None):
        q = f' quat="{_fmt(*quat)}"' if quat else ""
        geoms.append(f'<geom name="sop_{nombre}" type="box" pos="{_fmt(*pos)}" size="{_fmt(*half)}"{q} '
                     f'material="sop_{mat}" {A}/>')

    def cil(nombre, a, b, r, mat):
        geoms.append(f'<geom name="sop_{nombre}" type="cylinder" fromto="{_fmt(*a, *b)}" size="{r:.5g}" '
                     f'material="sop_{mat}" {A}/>')

    # ── ruedas ──
    com("Ruedas giratorias bajo las esquinas del bastidor")
    hx, hy, t = M["bastidor_x"] / 2, M["bastidor_y"] / 2, M["tubo"]
    zb = M["rueda_alto"]
    r, w = M["rueda_r"], M["rueda_ancho"]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        cx, cy = sx * (hx - t / 2), sy * (hy - t / 2)
        cil(f"rueda{i}", W(cx - w / 2, cy, r), W(cx + w / 2, cy, r), r, "rueda")
        cil(f"rueda{i}_buje", W(cx - w / 2 - 0.005, cy, r), W(cx + w / 2 + 0.005, cy, r), 0.016, "herraje")
        alto_h = (zb - 0.008) - (r - 0.01)
        for k in (-1, 1):
            box(f"rueda{i}_horquilla{(k + 1) // 2}", W(cx + k * (w / 2 + 0.009), cy - 0.012, r - 0.01 + alto_h / 2),
                (0.003, 0.022, alto_h / 2), "herraje")
        box(f"rueda{i}_placa", W(cx, cy, zb - 0.004), (0.04, 0.04, 0.004), "herraje")

    # ── bastidor + pletina ──
    com("Bastidor de tubo cuadrado y pletina bajo la columna")
    zt = zb + t / 2
    box("bastidor_delante", W(0, -hy + t / 2, zt), (hx, t / 2, t / 2), "bastidor")
    box("bastidor_detras", W(0, hy - t / 2, zt), (hx, t / 2, t / 2), "bastidor")
    box("bastidor_derecha", W(-hx + t / 2, 0, zt), (t / 2, hy - t, t / 2), "bastidor")
    box("bastidor_izquierda", W(hx - t / 2, 0, zt), (t / 2, hy - t, t / 2), "bastidor")
    zp = zb + t + M["pletina_esp"] / 2
    box("pletina", W(0, 0, zp), (M["pletina_x"] / 2, hy, M["pletina_esp"] / 2), "bastidor")

    # ── pie y columna ──
    com("Columna")
    px, py, pz = M["pie"]
    z_pie = zp + M["pletina_esp"] / 2 + pz / 2
    box("pie", W(0, 0, z_pie), (px / 2, py / 2, pz / 2), "chapa")
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        a = W(sx * (px / 2 - 0.022), sy * (py / 2 - 0.022), z_pie + pz / 2)
        cil(f"pie_tornillo{i}", a, _add(a, (0, 0, 0.009)), 0.010, "herraje")
    cx2, cy2 = M["columna_x"] / 2, M["columna_y"] / 2
    e = M["placa_esp"]
    z_c0, z_c1 = z_pie + pz / 2, g["z_top"] - Z0 - e
    box("columna", W(0, 0, (z_c0 + z_c1) / 2), (cx2, cy2, (z_c1 - z_c0) / 2), "columna")
    box("columna_franja", W(0, 0, z_c0 + 0.03), (cx2 + 0.002, cy2 + 0.002, 0.012), "acento")
    for i, (zv, alto, ancho) in enumerate(M["ventanas"]):
        for s_, lado in ((-1, "derecha"), (1, "izquierda")):
            box(f"ventana{i}_{lado}", W(s_ * (cx2 + 0.0008), 0, zv), (0.001, ancho / 2, alto / 2), "hueco")

    # ── cabezal: placa de arriba ──
    com(f"Cabezal: placa de arriba y dos alas con placas de montaje a {M['hombro_grados']:.0f} grados")
    zt_loc = g["z_top"] - Z0
    Ls = M["placa_sup"]
    box("placa_arriba", W(0, 0, zt_loc - e / 2), (Ls / 2, Ls / 2, e / 2), "chapa")
    box("placa_arriba_franja", W(0, 0, zt_loc - e - 0.008), (cx2 + 0.002, cy2 + 0.002, 0.008), "acento")
    cil("placa_arriba_agujero", W(0, 0, zt_loc - 0.0005), W(0, 0, zt_loc + 0.0006), M["agujero_r"], "hueco")
    m_ = M["tal_margen"]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        cil(f"placa_arriba_taladro{i}", W(sx * (Ls / 2 - m_), sy * (Ls / 2 - m_), zt_loc - 0.0005),
            W(sx * (Ls / 2 - m_), sy * (Ls / 2 - m_), zt_loc + 0.0006), M["tal_r"], "hueco")

    # ── alas: dos chapas triangulares + placa de montaje inclinada ──
    th, L = g["th"], M["placa"]
    z_o = g["z_o"]
    gy = M["gusset_esp"]
    for s_, lado in ((-1, "derecha"), (1, "izquierda")):
        # en el mundo: el ala derecha usa la geometría calculada; la izquierda es su espejo
        # respecto al eje de la columna (x → 2·x_col − x)
        espejo = (lambda p: p) if s_ < 0 else (lambda p: (2 * X0 - p[0], p[1], p[2]))
        I, O = espejo(g["I"]), espejo(g["O"])
        top_in = (I[0], 0.0, z_o)
        for k, sy in ((0, -1), (1, 1)):
            y0, y1 = sy * cy2, sy * (cy2 - gy)
            vert = []
            for (x, z) in ((top_in[0], z_o), (O[0], z_o), (I[0], I[2])):
                vert += [_fmt(x, y0, z), _fmt(x, y1, z)]
            mallas.append(f'<mesh name="sop_ala_{lado}_{k}" vertex="{"  ".join(vert)}"/>')
            geoms.append(f'<geom name="sop_ala_{lado}_{k}" type="mesh" mesh="sop_ala_{lado}_{k}" '
                         f'material="sop_columna" {A}/>')
        n = (s_ * math.sin(th), 0.0, -math.cos(th))
        cara = espejo(g["cara"])
        centro = _add(cara, n, -e / 2)
        q = _quat_y(math.atan2(n[0], n[2]))             # su eje z → n
        box(f"placa_{lado}", centro, (L / 2, L / 2, e / 2), "chapa", quat=q)
        # agujero central y taladros en la cara exterior
        a, b = _add(cara, n, 0.0006), _add(cara, n, -0.0005)
        cil(f"placa_{lado}_agujero", b, a, M["agujero_r"], "hueco")
        u = (math.cos(th), 0.0, s_ * math.sin(th))          # eje de la placa en el plano XZ
        brazo_aqui = s_ < 0
        for k, (du, dy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            off = tuple(du * (L / 2 - m_) * u[i] for i in range(3))
            pa = (a[0] + off[0], dy * (L / 2 - m_), a[2] + off[2])
            pb = (b[0] + off[0], dy * (L / 2 - m_), b[2] + off[2])
            if brazo_aqui:            # placa con el robot: cabezas de tornillo
                cil(f"placa_{lado}_tornillo{k}", pb, _add(pa, n, 0.008), 0.009, "herraje")
            else:
                cil(f"placa_{lado}_taladro{k}", pb, pa, M["tal_r"], "hueco")
        if brazo_aqui and M["disco"]:
            r_d, e_d = M["disco"]
            cil(f"placa_{lado}_disco", cara, _add(cara, n, e_d), r_d, "acento")

    # ── logos ──
    if LOGOS:
        com("Logos verticales: Elecnor en la cara delantera (-Y), I+D+i en la trasera (+Y)")
        zl = 0.5 * (z_c0 + z_o - Z0)
        hl = min(0.16, 0.42 * (z_o - Z0 - z_c0), 2.4 * cx2)      # proporción 3:1, cabe en la cara
        half = (hl, hl / 3, 0.01)
        geoms.append(f'<geom name="sop_logo_elecnor" type="plane" pos="{_fmt(*W(0, -cy2 - 0.0015, zl))}" '
                     f'xyaxes="0 0 1 -1 0 0" size="{_fmt(*half)}" material="logo_elecnor_material" {A}/>')
        geoms.append(f'<geom name="sop_logo_idi" type="plane" pos="{_fmt(*W(0, cy2 + 0.0015, zl))}" '
                     f'xyaxes="0 0 1 1 0 0" size="{_fmt(*half)}" material="logo_departamento_material" {A}/>')

    pal = PALETAS[COLORES]
    lin = [
        "<!--",
        f"  Soporte del FR5: columna con cabezal en T y dos hombros a {M['hombro_grados']:.0f}°,",
        "  sobre bastidor con ruedas. El FR5 va en la placa del hombro DERECHO (-X).",
        "  GENERADO por decoracion/crear_soporte_hombros.py — para cambiar medidas o colores,",
        "  edita el script y vuelve a ejecutarlo (rehace también la base del brazo).",
        "  Solo visual: geoms estáticos sin colisión ni masa (como la peana de antes).",
        f"  Suelo en z = {Z0:.4f} · columna en x = {X0:.4f} · placa de arriba en z = {g['z_top']:.4f}",
        "-->",
        "<mujoco>",
        "  <asset>",
    ]
    for k, (rgba, acabado) in pal.items():
        lin.append(f'    <material name="sop_{k}" rgba="{rgba}" {acabado}/>')
    lin += ["    " + x for x in mallas]
    lin += ["  </asset>", "  <worldbody>"]
    lin += ["    " + x for x in geoms]
    lin += ["  </worldbody>", "</mujoco>", ""]
    with open(RUTA_XML, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lin))
    print(f"✓ {os.path.relpath(RUTA_XML)}: {sum(1 for x in geoms if x.startswith('<geom'))} geoms")
    return g


RE_BASE = re.compile(r'<body name="base_link" pos="[^"]*"(?: quat="[^"]*")?>')


def colocar_base(g):
    with open(RUTA_MJCF, encoding="utf-8") as fh:
        xml = fh.read()
    f8 = lambda *v: " ".join(f"{(0.0 if abs(x) < 5e-12 else x):.9g}" for x in v)
    nueva = f'<body name="base_link" pos="{f8(*g["base"])}" quat="{f8(*g["base_quat"])}">'
    xml2, n = RE_BASE.subn(nueva, xml, count=1)
    if n != 1:
        raise SystemExit('✗ No encuentro <body name="base_link" ...> en fairino_mjcf.xml')
    if xml2 != xml:
        with open(RUTA_MJCF, "w", encoding="utf-8") as fh:
            fh.write(xml2)
    print(f"✓ fairino_mjcf.xml: base del FR5 en ({_fmt(*g['base'])}), "
          f"inclinada {MEDIDAS['hombro_grados']:.0f}° (quat {_fmt(*g['base_quat'])})")


if __name__ == "__main__":
    geo = generar()
    colocar_base(geo)
    print(f"  Suelo en z = {geo['z_suelo']:.4f} m (config.py → SOPORTE_SUELO_Z y marcas [H]/[Z] "
          f"si ha cambiado) · hombro del FR5 a "
          f"{HOMBRO_MUNDO[2] + (BASE_A_HOMBRO - D1_FR5) * math.cos(math.radians(MEDIDAS['hombro_grados'])) - geo['z_suelo']:.3f}"
          f" m del suelo")
