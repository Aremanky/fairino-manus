#!/usr/bin/env python3
"""
Calibración guante MANUS → OrcaHand, IMITACIÓN 1:1  (manus_calibracion.json, versión 4)
=======================================================================================
    python calibraciones/calibrar_manus.py              # calibración guiada (~2 min, 9 poses)
    python calibraciones/calibrar_manus.py --ver        # ver en vivo los 20 ángulos de tu mano
    python calibraciones/calibrar_manus.py --probar     # ver en vivo tus ángulos y el ctrl 1:1 del robot
    python calibraciones/calibrar_manus.py --recalcular # rehacer el mapeo con las poses YA grabadas en
                                                        # manus_calibracion.json (sin guante; guarda copia).
                                                        # Sirve también para pasar una calibración v3 a 1:1.

QUÉ HACE (v4, 15-oct)
  El robot copia tus ángulos GRADO A GRADO (ver la cabecera de manus_guante.py):
      q_robot = q_ref + signo · (tu ángulo − tu ángulo con la mano abierta)
  La calibración ya NO estira tu recorrido al del robot, NO graba pinzas ni poses que
  luego se fuercen. Solo mide:
    · el CERO de cada sensor: tu mano ABIERTA = dedos rectos y juntos en el robot;
    · el SENTIDO de cada canal del guante (con las demás poses: puño, abanico, pulgar...);
    · el desfase del CMC del pulgar (sus ejes no son los tuyos): un número por
      articulación, ajustado con tus 3 poses del pulgar.
  Las pinzas de calibraciones anteriores se ignoran.

v5 (15-oct) — MODO "cierre" (config.MANUS_MAPEO, por defecto): con el 1:1 puro los dedos
  no tocaban la palma (proporciones de la OrcaHand: necesita ~97° de nudillo y tu puño son
  71-84°). Ahora cada flexión es una RECTA por dos poses tuyas: mano abierta → robot recto,
  PUÑO → robot con la yema en la palma; el pulgar cruzado → pulgar del robot apoyado en la
  palma (ver la cabecera de manus_guante.py). Por eso en "cierre" el PUÑO y las 3 poses del
  pulgar también son importantes: ciérralo como lo cerrarás al usarlo.
  "1:1" sigue disponible (MANUS_MAPEO = "1:1"); el mapeo se rehace al arrancar con las poses
  guardadas, así que cambiar de modo NO exige recalibrar.

CONSEJOS: mantén cada pose quieta ~3 s. La pose ABIERTA es la importante (es el cero de
todo): dedos estirados de verdad y juntos, pulgar estirado y relajado al lado del índice.
En las tres del pulgar exagera la DIFERENCIA entre ellas; en "plano" el pulgar va ESTIRADO
y sin tocar la palma (el 13-oct salió cruzado: era igual que "cruz").
"""
import json
import os
import shutil
import sys
import time

import numpy as np

# Este script vive en calibraciones/: la raíz del repo (config.py, skill_shield/) tiene que estar en sys.path
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from skill_shield.mano.manus_guante import (  # noqa: E402
    ERGO, NOMBRES, GuanteManus, MANUS_CALIBRACION, MANUS_MAPEO, angulos_a_ctrl,
    cargar_calibracion, construir_mapeo,
)

ESPERA_S, MEDIR_S = 2.0, 1.5

# (clave, texto que se muestra)
POSES = [
    ("abierta",    "mano ABIERTA: dedos ESTIRADOS del todo y JUNTOS (pegados entre sí), pulgar "
                   "estirado y relajado junto al índice.  ← ES EL CERO DE TODO"),
    ("puno",       "PUÑO cerrado y APRETADO (pulgar por fuera, como te salga)"),
    ("separados",  "dedos estirados y SEPARADOS al máximo (abanico), pulgar también abierto"),
    ("corazon_indice", "dedos estirados y JUNTOS, inclinados TODOS hacia el lado del PULGAR "
                       "(el CORAZÓN se inclina hacia el ÍNDICE)"),
    ("corazon_anular", "dedos estirados y JUNTOS, inclinados TODOS hacia el lado del MEÑIQUE "
                       "(el CORAZÓN se inclina hacia el ANULAR)"),
    ("pulgar_plano", "PULGAR ESTIRADO, tumbado en el PLANO de la palma y lo más SEPARADO del índice "
                     "(una 'L' plana; el pulgar NO toca la palma)"),
    ("pulgar_perp",  "PULGAR PERPENDICULAR a la palma (apuntando hacia delante de la palma), dedos estirados"),
    ("pulgar_cruz",  "PULGAR CRUZADO y APOYADO sobre la palma, con la yema en la base del MEÑIQUE, dedos estirados"),
    ("pulgar_flex",  "PULGAR DOBLADO al máximo (nudillo y punta), dedos estirados"),
]

# Dos poses del pulgar con los 4 canales del pulgar a menos de esto (°) son "la misma pose"
POSES_PULGAR = ("pulgar_plano", "pulgar_perp", "pulgar_cruz")
UMBRAL_GEMELAS_DEG = 6.0
_CANALES_PULGAR = [ERGO[c] for c in ("THUMB_CMC_SPREAD", "THUMB_CMC_STRETCH", "THUMB_MCP", "THUMB_IP")]


def medir(guante, segundos):
    muestras, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < segundos:
        ang, edad = guante.angulos()
        if ang is not None and edad < 0.2:
            muestras.append(ang)
        time.sleep(0.01)
    if len(muestras) < 10:
        return None, None
    m = np.array(muestras)
    return m.mean(axis=0), m.std(axis=0)


def pose_pulgar_gemela(medidas, clave):
    """Otra pose del pulgar (plano/perp/cruz) con los 4 canales del pulgar casi iguales que
    'clave' (una de las dos está mal hecha), o None."""
    if clave not in POSES_PULGAR or clave not in medidas:
        return None
    a = np.asarray(medidas[clave])[_CANALES_PULGAR]
    for otra in POSES_PULGAR:
        if otra != clave and otra in medidas:
            if float(np.abs(a - np.asarray(medidas[otra])[_CANALES_PULGAR]).max()) < UMBRAL_GEMELAS_DEG:
                return otra
    return None


def informar(informe):
    print(f"\nResultado (modo '{MANUS_MAPEO}'; base 1:1: tu ángulo desde la mano abierta = ángulo "
          f"del robot desde su reposo):")
    for k in sorted(informe):
        print(f"  ctrl[{k:2d}] {NOMBRES[k]:12s} {informe[k]}")


def comprobar(medidas, mapeo):
    """Lo que hará el robot en tus poses de calibración (grados, para compararlos con los tuyos)."""
    print("\nComprobación: robot en tus poses (GRADOS)")
    def fila(pose, ks):
        if pose in medidas:
            c = angulos_a_ctrl(medidas[pose], mapeo)
            print(f"  {pose:15s} " + "  ".join(f"{NOMBRES[k]}={np.degrees(c[k]):+5.0f}" for k in ks))
    for pose in ("abierta", "puno"):
        fila(pose, (11, 12, 8, 9, 5, 6, 2, 3))
    for pose in ("abierta", "separados", "corazon_indice", "corazon_anular"):
        fila(pose, (10, 7, 4, 1))
    for pose in ("abierta", "separados", "puno", "pulgar_plano", "pulgar_perp", "pulgar_cruz", "pulgar_flex"):
        fila(pose, (13, 14, 15, 16))
    print("  (t-cmc: − = pulgar hacia FUERA, + = hacia el MEÑIQUE · t-abd: + = en el plano de la "
          "palma, − = perpendicular · el resto: + = cerrar / hacia el pulgar)")
    print("  Yema en la palma ≈ mcp +97 y pip +103 · pulgar apoyado en la palma ≈ cmc +33, abd −18, mcp +80")


def guardar(ruta, mapeo, medidas, fecha=None, extra=None):
    d = {"version": 5, "modo": MANUS_MAPEO, "fecha": fecha or time.strftime("%Y-%m-%d %H:%M"),
         "mapeo": {str(k): v for k, v in mapeo.items()},
         "poses": len(medidas),
         "poses_crudas": {p: [float(x) for x in a] for p, a in medidas.items()}}
    d.update(extra or {})
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)


def recalcular(ruta=MANUS_CALIBRACION):
    """Rehace el mapeo 1:1 con las poses crudas ya grabadas (sin guante). Guarda copia."""
    if not os.path.exists(ruta):
        print(f"❌ No existe {ruta}")
        return 1
    with open(ruta, encoding="utf-8") as fh:
        d = json.load(fh)
    crudas = d.get("poses_crudas")
    if not crudas:
        print("❌ Esa calibración no tiene 'poses_crudas': hay que repetirla con el guante:  "
              "python calibraciones/calibrar_manus.py")
        return 1
    medidas = {p: np.asarray(a, dtype=float) for p, a in crudas.items()}
    if "abierta" not in medidas:
        print("❌ Falta la pose 'abierta' (el cero): repite la calibración.")
        return 1
    gem = pose_pulgar_gemela(medidas, "pulgar_cruz")
    if gem is not None:
        print(f"⚠️  En esta calibración '{gem}' y 'pulgar_cruz' son casi la misma pose para el guante "
              f"(una está mal hecha). Se compensa con otras poses; la próxima vez repítela.")
    mapeo, informe = construir_mapeo(medidas)
    informar(informe)
    comprobar(medidas, mapeo)
    copia = ruta + time.strftime(".antes_recalcular_%Y%m%d_%H%M%S")
    shutil.copy(ruta, copia)
    guardar(ruta, mapeo, medidas, fecha=d.get("fecha"),
            extra={"recalculado": time.strftime("%Y-%m-%d %H:%M")})
    print(f"\n💾 Mapeo (modo '{MANUS_MAPEO}') guardado en {ruta}\n   (copia de la anterior: {copia})")
    cargar_calibracion(verbose=True)
    return 0


def pedir_y_medir(guante, i, n, clave, texto):
    """Pide una pose y la mide. Devuelve la media (array) o None (saltada o sin datos)."""
    r = input(f"[{i}/{n}] {texto}  → ENTER (o 's' + ENTER para saltarla) ").strip().lower()
    if r == "s":
        print("   (saltada)")
        return None
    time.sleep(ESPERA_S)
    media, dispersion = medir(guante, MEDIR_S)
    if media is None:
        print("   ⚠️  sin datos, pose descartada")
        return None
    if dispersion.max() > 8.0:
        print(f"   ⚠️  te has movido bastante (hasta {dispersion.max():.0f}°): la uso igualmente")
    print("   ✓ medida")
    return media


def main():
    if "--recalcular" in sys.argv:
        return recalcular()

    guante = GuanteManus(tactil=False)
    if not guante.iniciar():
        return 1
    print("Esperando datos del guante...")
    t0 = time.perf_counter()
    while not guante.datos_frescos():
        if time.perf_counter() - t0 > 20:
            print("❌ No llegan datos del guante en 20 s (¿Dashboard abierto / dongle, guante encendido?).")
            guante.cerrar()
            return 1
        time.sleep(0.2)

    if "--ver" in sys.argv:
        print("Ángulos de tu mano en vivo, grados (Ctrl+C para salir)")
        nombres = sorted(ERGO, key=ERGO.get)
        try:
            while True:
                ang, _ = guante.angulos()
                print("\r" + "  ".join(f"{n.split('_', 1)[1][:9]}={a:6.1f}" for n, a in zip(nombres, ang))
                      if ang else "\rsin datos", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
        guante.cerrar()
        return 0

    if "--probar" in sys.argv:
        print("ctrl 1:1 de la OrcaHand que sale del guante (GRADOS; Ctrl+C para salir)")
        try:
            while True:
                obj = guante.ctrl_objetivo()
                if obj is None:
                    print("\rsin datos", end="", flush=True)
                else:
                    txt = " ".join(f"{NOMBRES[k][:6]}={np.degrees(obj[k]):+4.0f}" for k in range(1, 17))
                    print("\r" + txt + "   ", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
        guante.cerrar()
        return 0

    print(f"\n🧤 Calibración 1:1 guante → OrcaHand: {len(POSES)} poses. En cada una: colócate, "
          f"pulsa ENTER y mantén la pose ~{ESPERA_S + MEDIR_S:.0f} s.\n")
    medidas = {}
    textos = dict(POSES)
    for i, (clave, texto) in enumerate(POSES, 1):
        media = pedir_y_medir(guante, i, len(POSES), clave, texto)
        if media is None:
            continue
        medidas[clave] = media
        # Dos poses del pulgar iguales = una está mal hecha (13-oct: "plano" salió como "cruz")
        while True:
            gemela = pose_pulgar_gemela(medidas, clave)
            if gemela is None:
                break
            r = input(f"   ⚠️  '{clave}' sale casi IGUAL que '{gemela}' para el guante (pulgar a menos de "
                      f"{UMBRAL_GEMELAS_DEG:.0f}°): una de las dos está mal hecha.\n"
                      f"      ¿Cuál repites?  [1] {gemela}   [2] {clave}   [ENTER] ninguna: ").strip()
            if r not in ("1", "2"):
                break
            rep = gemela if r == "1" else clave
            nueva = pedir_y_medir(guante, i, len(POSES), rep, textos[rep])
            if nueva is None:
                break
            medidas[rep] = nueva
    guante.cerrar()
    if "abierta" not in medidas:
        print("❌ Falta la pose 'abierta' (es el cero de todo): no se guarda nada.")
        return 1

    mapeo, informe = construir_mapeo(medidas)
    informar(informe)
    comprobar(medidas, mapeo)
    guardar(MANUS_CALIBRACION, mapeo, medidas)
    print(f"\n💾 Guardado en {MANUS_CALIBRACION}. Skill Shield lo carga al arrancar.")
    cargar_calibracion(verbose=True)
    print("Retoque fino sin recalibrar: config.MANUS_GANANCIA multiplica cada articulación "
          "(1.0 = como sale de la calibración).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
