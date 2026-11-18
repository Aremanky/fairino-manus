"""
¿Qué cubos del mapa no dejan coger la pelota? (18-nov)

Con un plan fallido guardado (diagnosticos/plan_fallido_*.npz), lo vuelve a planificar con el código y el config
de ahora, apunta los cubos con los que choca la mano y los enseña:

    python diagnosticos/ver_estorbo.py diagnosticos/plan_fallido_20261118_132039.npz
    python diagnosticos/ver_estorbo.py                    # el último plan fallido

  · en la terminal: cada cubo que estorba, con su distancia a la pelota, su altura sobre la base de la pelota y
    hacia dónde está visto desde la cámara (derecha / izquierda, delante / detrás de la pelota);
  · diagnosticos/estorbo_<plan>.png: vista desde ARRIBA de 50 x 50 cm alrededor de la pelota, girada como la ve
    la cámara (la cámara abajo; derecha = derecha de la imagen de la web). Cubos del mapa coloreados por altura
    sobre la base de la pelota (gris: a su altura o por debajo, la mesa; amarillo → rojo: cada vez más altos), los
    que estorban con borde negro, la pelota en verde y el círculo de 15 cm que suele necesitar la mano abierta con
    sus márgenes.
"""
import glob
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))


def main():
    import cv2
    import config
    from banco_planes import planificar
    from skill_shield.autonomo import tarea as TA
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    rutas = sys.argv[1:] or sorted(glob.glob(os.path.join(RAIZ, "diagnosticos", "plan_fallido_*.npz")))[-1:]
    if not rutas:
        sys.exit("No hay planes fallidos guardados en diagnosticos/.")
    ruta = rutas[0]
    G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    d = np.load(ruta, allow_pickle=False)
    info = json.loads(str(d["info"]))
    if not info.get("pelota"):
        sys.exit("Este plan no es de una pelota.")
    T_cam = info.get("T_mundo_cam")
    if T_cam is None:
        from skill_shield.autonomo.calibracion import cargar
        T_cam = cargar(os.path.join(RAIZ, G.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")))["T_mundo_cam"]
    T_cam = np.asarray(T_cam, float)
    pl = TA.Planificador(cargar_cadena(G["RUTA_XML"], verbose=False), G, T_cam)
    try:
        planificar(pl, d, info)
        print("✅ Con el código y el config de ahora, este plan SÍ sale.")
    except TA.ErrorPlan as e:
        print(f"❌ {e}\n")
    diam = float(info.get("diametro") or 0.07)
    r = diam / 2
    T0 = np.asarray(d["T"], float).reshape(-1, 4, 4)[0]
    c = pl.apoyar(T0, diam, info.get("apoyo_z"), T_cam)[:3, 3]           # donde se planifica cogerla
    base = c[2] - r
    C = pl._arbol.data if pl._arbol is not None else np.zeros((0, 3))     # el mapa que usa el plan (sin la pelota)
    V = np.unique(np.round(np.asarray(pl._choques, float).reshape(-1, 3), 4), axis=0)
    # ejes «de la cámara» en horizontal: derecha de la imagen y hacia donde mira
    der = np.array([T_cam[0, 0], T_cam[1, 0], 0.0]); der /= np.linalg.norm(der)
    lej = np.array([T_cam[0, 2], T_cam[1, 2], 0.0]); lej -= der * (lej @ der); lej /= np.linalg.norm(lej)
    cerca = V[np.linalg.norm(V[:, :2] - c[:2], axis=1) < 0.30] if len(V) else V
    print(f"Pelota de {diam * 100:.1f} cm en ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) · {len(C)} cubos en el mapa del plan"
          f" · la mano ha chocado con {len(V)} cubos distintos ({len(cerca)} a menos de 30 cm de la pelota)")
    if len(cerca):
        print("\n   dist. (cm)  altura sobre la base (cm)   visto desde la cámara")
        orden = np.argsort(np.linalg.norm(cerca[:, :2] - c[:2], axis=1))
        for v in cerca[orden][:25]:
            w = v - c
            lado = f"{abs(100 * w @ der):4.0f} cm a la {'derecha' if w @ der > 0 else 'izquierda'}"
            fondo = f"{abs(100 * w @ lej):4.0f} cm {'detrás' if w @ lej > 0 else 'delante'}"
            print(f"   {100 * np.linalg.norm(w[:2]):8.1f}   {100 * (v[2] - base):12.1f}              {lado}, {fondo}")
    # vista desde arriba
    L, px = 0.50, 600
    img = np.full((px, px, 3), 255, np.uint8)

    def a_px(p):                                  # derecha de la cámara → derecha; alejándose de ella → arriba
        w = np.asarray(p, float) - c
        return np.array([w @ der, w @ lej]) / L * px * np.array([1, -1]) + px / 2
    lado_px = max(2, int(pl._res / L * px) - 1)
    C = np.asarray(pl._C, float).reshape(-1, 3)                          # con la mesa (gris), para ver el contexto
    sel = np.all(np.abs(C[:, :2] - c[:2]) < L / 2, axis=1)
    for v in C[sel][np.argsort(C[sel, 2])]:
        h = v[2] - base
        col = (190, 190, 190) if h < 0.02 else (0, int(max(0, 255 - 1800 * (h - 0.02))), 255)
        x, y = a_px(v).astype(int)
        cv2.rectangle(img, (x - lado_px // 2, y - lado_px // 2), (x + lado_px // 2, y + lado_px // 2), col, -1)
    for v in cerca:
        x, y = a_px(v).astype(int)
        cv2.rectangle(img, (x - lado_px // 2, y - lado_px // 2), (x + lado_px // 2, y + lado_px // 2), (0, 0, 0), 2)
    cx, cy = a_px(c).astype(int)
    cv2.circle(img, (cx, cy), int(r / L * px), (0, 170, 0), -1)
    cv2.circle(img, (cx, cy), int(0.15 / L * px), (0, 120, 0), 1)
    cv2.arrowedLine(img, (px // 2, px - 30), (px // 2, px - 70), (200, 0, 0), 2, tipLength=0.25)
    cv2.putText(img, "la camara mira hacia arriba", (px // 2 + 8, px - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1)
    cv2.putText(img, "gris: mesa · amarillo->rojo: mas alto · borde negro: estorba · circulo: 15 cm",
                (8, px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    salida = os.path.join(RAIZ, "diagnosticos", "estorbo_" + os.path.splitext(os.path.basename(ruta))[0] + ".png")
    cv2.imwrite(salida, img)
    print(f"\n🖼️  Vista desde arriba: {salida}")


if __name__ == "__main__":
    main()
