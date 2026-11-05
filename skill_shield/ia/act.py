"""
Cerebro IA: modelo ACT (Action Chunking Transformer), episodios, entrenamiento y carga.

(Extraído de Skill_Shield_Nivel_6.py.)

5-nov — BRAZO + MANO (formato 2): los episodios nuevos (skill_shield/ia/grabador.py) traen
cabecera y guardan los 6 joints del FR5 + los 17 de la OrcaHand (posición simulada q:… y
comando a:…) a ritmo fijo. La red es la misma (ACT "ciega": historial de posturas → bloque de
comandos futuros) pero con 23 dimensiones, y el .pth guarda los nombres de las columnas para
que la reproducción sepa qué joint es cada salida. Los episodios antiguos (formato 1: solo
mano, sin cabecera) se siguen pudiendo entrenar como antes.

Entrenar FUERA de la app (no congela cámaras ni física; [E] ahora lo lanza así también):
    python -m skill_shield.ia.act <tarea>          # una tarea → modelos/<tarea>.pth
    python -m skill_shield.ia.act --todas          # todas las tareas grabadas
"""
import os
import sys
import json
import math
from collections import deque
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# CEREBRO IA  —  ACT (Action Chunking Transformer) simplificado, sin imágenes
# ══════════════════════════════════════════════════════════════════════════════
# Basado en la arquitectura de ACT (Zhao et al., "Learning Fine-Grained Bimanual
# Manipulation with Low-Cost Hardware") — la misma familia de algoritmo que usan
# proyectos como ALOHA. Dos piezas nuevas respecto a nuestro MLP anterior:
#
# 1. CVAE con variable latente de "estilo" (z): un pequeño Transformer codificador,
#    que SOLO se usa durante el entrenamiento, resume la secuencia de acciones real
#    de cada episodio en un vector z. El decodificador aprende a partir de z, así
#    que nunca tiene que "promediar" entre gestos distintos — cada uno se explica
#    por su propio z. En inferencia no hay demostración real disponible, así que
#    se usa z=0 (la media del prior).
# 2. Transformer en vez de MLP: el decodificador usa "action queries" aprendidas
#    (una por cada paso futuro del bloque) que atienden, vía cross-attention, al
#    historial de postura + z, y producen el bloque de acciones futuras de golpe.
#
# Entradas : historial de VENTANA_IA fotogramas de qpos (+ z durante entrenamiento)
# Salidas  : bloque de CHUNK_IA fotogramas futuros de ctrl

VENTANA_IA     = 8    # nº de fotogramas de historial que ve la red en cada predicción
CHUNK_IA       = 30   # nº de fotogramas futuros que la red predice de una vez (ACT
                       # suele usar bloques más largos que un MLP simple, ya que el
                       # "temporal ensembling" en inferencia se encarga de suavizar)
DIM_LATENTE    = 32    # tamaño del vector de estilo z
DIM_MODELO     = 128   # dimensión interna del Transformer
N_CABEZAS      = 4     # nº de cabezas de atención
N_CAPAS_ENC    = 4     # capas del Transformer codificador de estilo
N_CAPAS_CTX    = 4     # capas del Transformer que codifica el contexto (historial+z)
N_CAPAS_DEC    = 1     # capas del decoder de acciones — LeRobot (implementación de
                       # referencia de ACT) usa 1 aquí a propósito: la implementación
                       # original tenía un bug que hacía que solo la 1ª capa de un
                       # decoder de 7 se usara realmente, y así es como se validó y
                       # funcionó en la práctica — lo replicamos en vez de "arreglarlo"
BETA_KL        = 10.0  # peso del término KL en la pérdida (estándar en ACT: 10-20)

IA_MAX_EPOCAS  = 8000  # tope de épocas del entrenamiento (las pruebas lo bajan)
IA_PACIENCIA   = 200   # épocas sin mejorar en validación antes de parar

EPISODIOS_DIR  = "episodios"  # cada TAREA tiene su propia subcarpeta: episodios/<tarea>/
                               # y dentro, cada grabación [G]...[G] es un episodio_XXXX.csv
MODELOS_DIR    = "modelos"    # un modelo .pth entrenado por cada tarea: modelos/<tarea>.pth


def _nombre_seguro(texto: str) -> str:
    """Convierte un nombre de tarea escrito por el usuario en un nombre de carpeta/
    archivo seguro (sin espacios ni caracteres raros)."""
    limpio = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in texto.strip())
    return limpio.lower() or "tarea_sin_nombre"


def _listar_tareas_grabadas(episodios_dir: str = EPISODIOS_DIR):
    """Devuelve los nombres de las subcarpetas de tareas que tengan al menos 1 episodio."""
    if not os.path.isdir(episodios_dir):
        return []
    tareas = []
    for nombre in sorted(os.listdir(episodios_dir)):
        ruta = os.path.join(episodios_dir, nombre)
        if os.path.isdir(ruta) and _listar_episodios(ruta):
            tareas.append(nombre)
    return tareas


def _listar_modelos_entrenados(modelos_dir: str = MODELOS_DIR):
    """Devuelve los nombres de tarea (sin '.pth') que ya tienen un modelo entrenado."""
    if not os.path.isdir(modelos_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(modelos_dir) if f.endswith(".pth"))



class ACTModel(nn.Module):
    """CVAE + Transformer para clonación de conducta con variable latente de estilo."""

    def __init__(self, n_joints: int = N_ACT, ventana: int = VENTANA_IA,
                 chunk: int = CHUNK_IA, dim: int = DIM_MODELO,
                 n_heads: int = N_CABEZAS, n_capas_enc: int = N_CAPAS_ENC,
                 n_capas_ctx: int = N_CAPAS_CTX, n_capas_dec: int = N_CAPAS_DEC,
                 dim_z: int = DIM_LATENTE, dim_accion: int = None):
        super().__init__()
        self.ventana, self.chunk, self.dim_z = ventana, chunk, dim_z
        dim_accion = n_joints if dim_accion is None else dim_accion   # 5-nov: pueden diferir

        # — Embeddings de entrada —
        self.embed_qpos   = nn.Linear(n_joints, dim)
        self.embed_action = nn.Linear(dim_accion, dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, 2 + chunk, dim))  # CLS + qpos + chunk

        # — Codificador de estilo (solo se usa con demostración real, en training) —
        capa_enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, batch_first=True)
        self.style_encoder = nn.TransformerEncoder(capa_enc, num_layers=n_capas_enc)
        self.to_mu_logvar  = nn.Linear(dim, dim_z * 2)

        # — Política: Transformer encoder (contexto) + decoder (action queries) —
        self.latent_proj    = nn.Linear(dim_z, dim)
        self.pos_embed_pol  = nn.Parameter(torch.zeros(1, ventana + 1, dim))
        capa_pol_enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, batch_first=True)
        self.policy_encoder = nn.TransformerEncoder(capa_pol_enc, num_layers=n_capas_ctx)

        self.action_queries = nn.Parameter(torch.zeros(1, chunk, dim))
        capa_pol_dec = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, batch_first=True)
        self.policy_decoder = nn.TransformerDecoder(capa_pol_dec, num_layers=n_capas_dec)

        self.cabeza_salida = nn.Linear(dim, dim_accion)

    def _codificar_estilo(self, qpos_actual: torch.Tensor, action_chunk: torch.Tensor):
        b = action_chunk.shape[0]
        act_emb  = self.embed_action(action_chunk)                    # (B, chunk, dim)
        qpos_emb = self.embed_qpos(qpos_actual).unsqueeze(1)           # (B, 1, dim)
        cls = self.cls_token.expand(b, -1, -1)                        # (B, 1, dim)
        seq = torch.cat([cls, qpos_emb, act_emb], dim=1) + self.pos_embed_enc
        salida_enc = self.style_encoder(seq)
        mu, logvar = self.to_mu_logvar(salida_enc[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def _decodificar(self, qpos_hist: torch.Tensor, z: torch.Tensor):
        b = qpos_hist.shape[0]
        hist_emb = self.embed_qpos(qpos_hist)                         # (B, ventana, dim)
        z_emb    = self.latent_proj(z).unsqueeze(1)                   # (B, 1, dim)
        contexto = torch.cat([hist_emb, z_emb], dim=1) + self.pos_embed_pol
        memoria  = self.policy_encoder(contexto)
        queries  = self.action_queries.expand(b, -1, -1)
        salida_dec = self.policy_decoder(queries, memoria)
        return self.cabeza_salida(salida_dec)                         # (B, chunk, n_joints)

    def forward(self, qpos_hist: torch.Tensor, action_chunk: torch.Tensor = None):
        """Si se pasa action_chunk (entrenamiento): usa el codificador de estilo real.
        Si no (inferencia): usa z=0 (media del prior), sin necesitar demostración."""
        b = qpos_hist.shape[0]
        if action_chunk is not None:
            qpos_actual = qpos_hist[:, -1, :]   # postura más reciente del historial
            mu, logvar = self._codificar_estilo(qpos_actual, action_chunk)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            mu = logvar = None
            z = torch.zeros(b, self.dim_z, device=qpos_hist.device)
        prediccion = self._decodificar(qpos_hist, z)
        return prediccion, mu, logvar


def _listar_episodios(episodios_dir: str = EPISODIOS_DIR):
    if not os.path.isdir(episodios_dir):
        return []
    return sorted(
        os.path.join(episodios_dir, f)
        for f in os.listdir(episodios_dir)
        if f.startswith("episodio_") and f.endswith(".csv")
    )


def _siguiente_numero_episodio(episodios_dir: str = EPISODIOS_DIR) -> int:
    existentes = _listar_episodios(episodios_dir)
    if not existentes:
        return 1
    numeros = []
    for ruta in existentes:
        nombre = os.path.basename(ruta)
        try:
            numeros.append(int(nombre.replace("episodio_", "").replace(".csv", "")))
        except ValueError:
            continue
    return (max(numeros) + 1) if numeros else 1


def leer_episodio(ruta):
    """Lee un episodio de cualquier formato. Devuelve dict(estado (T, De), accion (T, Da),
    tiempos (T,) o None, col_estado, col_accion, formato, fps) o None si no se puede leer.
      · formato 2 (grabador.py): cabecera t, t_reloj, q:<joint>…, a:<joint>…
      · formato 1 (antiguo, solo mano): 17 qpos + 17 ctrl [+ timestamp], sin cabecera"""
    with open(ruta, "r") as f:
        filas = list(csv.reader(f))
    if not filas:
        return None
    if filas[0] and filas[0][0] == "t":                         # ── formato 2
        cab = filas[0]
        ie = [i for i, c in enumerate(cab) if c.startswith("q:")]
        ia = [i for i, c in enumerate(cab) if c.startswith("a:")]
        datos = np.array([[float(v) for v in r] for r in filas[1:] if len(r) == len(cab)],
                         dtype=np.float64).reshape(-1, len(cab))
        fps = None
        meta = ruta[:-4] + ".json"
        if os.path.exists(meta):
            try:
                with open(meta) as fm:
                    fps = float(json.load(fm).get("fps"))
            except Exception:
                fps = None
        return dict(estado=datos[:, ie], accion=datos[:, ia], tiempos=datos[:, 0],
                    col_estado=[cab[i] for i in ie], col_accion=[cab[i] for i in ia],
                    formato=2, fps=fps)
    qpos, ctrl, tiempos = [], [], []                            # ── formato 1
    for row in filas:
        # 17 qpos + 17 ctrl + 1 timestamp = 35 · episodios aún más viejos: 34
        if len(row) == 2 * N_ACT + 1:
            vals = [float(v) for v in row]
            qpos.append(vals[:N_ACT]); ctrl.append(vals[N_ACT:2 * N_ACT]); tiempos.append(vals[-1])
        elif len(row) == 2 * N_ACT:
            vals = [float(v) for v in row]
            qpos.append(vals[:N_ACT]); ctrl.append(vals[N_ACT:])
    if not qpos:
        return None
    return dict(estado=np.array(qpos), accion=np.array(ctrl),
                tiempos=np.array(tiempos) if len(tiempos) == len(qpos) else None,
                col_estado=None, col_accion=None, formato=1, fps=None)


def guardar_episodio(buffer_filas, episodios_dir: str = EPISODIOS_DIR) -> str | None:
    """Guarda un episodio grabado (lista de filas qpos+ctrl) como un CSV independiente.
    Devuelve la ruta guardada, o None si el buffer estaba vacío."""
    if not buffer_filas:
        return None
    os.makedirs(episodios_dir, exist_ok=True)
    numero = _siguiente_numero_episodio(episodios_dir)
    ruta = os.path.join(episodios_dir, f"episodio_{numero:04d}.csv")
    with open(ruta, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(buffer_filas)
    return ruta


def preparar_datos(rutas):
    """Episodios → ventanas de entrenamiento (sin torch). Devuelve dict(x (N, ventana, De),
    y (N, chunk, Da), episodio (N,), n_episodios, formato, col_estado, col_accion,
    dt_objetivo) o None si no hay datos suficientes."""
    minimo_filas = VENTANA_IA + CHUNK_IA
    x_data, y_data, episodio_id_data = [], [], []
    episodios_usados, episodios_cortos = 0, 0
    deltas_tiempo = []   # diferencias reales entre fotogramas consecutivos (s)

    # 5-nov: episodios de formato 2 (brazo + mano) y 1 (solo mano). No se pueden mezclar en
    # un mismo modelo: si hay de los dos, se usan los NUEVOS (y se avisa).
    leidos = [(r, leer_episodio(r)) for r in rutas]
    leidos = [(r, e) for r, e in leidos if e is not None]
    if any(e["formato"] == 2 for _, e in leidos):
        viejos = sum(1 for _, e in leidos if e["formato"] != 2)
        leidos = [(r, e) for r, e in leidos if e["formato"] == 2]
        cols = (leidos[0][1]["col_estado"], leidos[0][1]["col_accion"])
        distintos = [r for r, e in leidos if (e["col_estado"], e["col_accion"]) != cols]
        leidos = [(r, e) for r, e in leidos if (e["col_estado"], e["col_accion"]) == cols]
        if viejos:
            print(f"⚠️  {viejos} episodio(s) del formato antiguo (solo mano) ignorados: "
                  "este modelo es de brazo + mano.")
        if distintos:
            print(f"⚠️  {len(distintos)} episodio(s) con otras articulaciones ignorados.")
        formato, col_estado, col_accion = 2, cols[0], cols[1]
        fps_meta = leidos[0][1]["fps"]
        print(f"🦾 Formato 2: {len(col_estado)} articulaciones (brazo + mano).")
    else:
        formato, col_estado, col_accion, fps_meta = 1, None, None, None

    for ruta, ep in leidos:
        qpos_seq, ctrl_seq = ep["estado"].tolist(), ep["accion"].tolist()
        tiempos = [] if ep["tiempos"] is None else ep["tiempos"].tolist()

        if len(qpos_seq) <= minimo_filas:
            episodios_cortos += 1
            continue
        episodios_usados += 1

        # — Ritmo real de grabación: diferencias entre timestamps consecutivos —
        if len(tiempos) >= 2:
            deltas_tiempo.extend(t2 - t1 for t1, t2 in zip(tiempos, tiempos[1:]))

        # Ventanas SOLO dentro de este episodio — nunca cruzan a otro episodio
        for i in range(VENTANA_IA, len(qpos_seq) - CHUNK_IA + 1):
            ventana = qpos_seq[i - VENTANA_IA:i]
            bloque  = ctrl_seq[i:i + CHUNK_IA]
            x_data.append(ventana)
            y_data.append(bloque)
            episodio_id_data.append(episodios_usados)  # id = nº de episodio válido, no de fila

    # Mediana en vez de media: robusta frente a algún salto puntual de la cámara
    DT_POR_DEFECTO = 1.0 / 15.0   # ~15 fps, si no hay timestamps (episodios viejos)
    if fps_meta:                                     # formato 2: ritmo fijo de la grabadora
        dt_objetivo = 1.0 / fps_meta
    elif deltas_tiempo:
        dt_objetivo = float(np.median(deltas_tiempo))
        # Descartamos valores absurdos (p.ej. si hubo una pausa real en medio)
        if not (0.01 <= dt_objetivo <= 0.5):
            dt_objetivo = DT_POR_DEFECTO
    else:
        dt_objetivo = DT_POR_DEFECTO
    print(f"⏱  Ritmo real de grabación detectado: {dt_objetivo*1000:.1f} ms/fotograma "
          f"(~{1/dt_objetivo:.1f} fps)")

    if episodios_cortos:
        print(f"⚠️  {episodios_cortos} episodio(s) descartado(s) por ser más cortos "
              f"que el mínimo ({minimo_filas + 1} filas).")

    if not x_data:
        print("❌ Ningún episodio tiene suficientes fotogramas. Graba más datos.")
        return None

    return dict(x=np.array(x_data, dtype=np.float32),            # (N, ventana, De)
                y=np.array(y_data, dtype=np.float32),            # (N, chunk, Da)
                episodio=np.array(episodio_id_data), n_episodios=episodios_usados,
                formato=formato, col_estado=col_estado, col_accion=col_accion,
                dt_objetivo=dt_objetivo)


def entrenar_ia(episodios_dir: str = EPISODIOS_DIR, archivo_pesos: str = ARCHIVO_PESOS) -> bool:
    """Entrena el modelo ACT (CVAE+Transformer) a partir de los episodios grabados."""
    rutas = _listar_episodios(episodios_dir)
    if not rutas:
        print(f"❌ No hay episodios grabados en '{episodios_dir}/'. Pulsa [G] primero.")
        return False

    print(f"\n🧠 ENTRENANDO ACT (CVAE + Transformer) con {len(rutas)} episodio(s)... "
          f"Por favor, espera.")

    d = preparar_datos(rutas)
    if d is None:
        return False
    x_arr, y_arr, dt_objetivo = d["x"], d["y"], d["dt_objetivo"]
    episodio_id_data, episodios_usados = d["episodio"], d["n_episodios"]
    formato, col_estado, col_accion = d["formato"], d["col_estado"], d["col_accion"]

    UMBRAL_MUESTRAS_MINIMO = 300
    if len(x_arr) < UMBRAL_MUESTRAS_MINIMO:
        print(f"⚠️  AVISO: solo {len(x_arr)} muestras de entrenamiento de "
              f"{episodios_usados} episodio(s). Recomendado: más episodios / más "
              f"repeticiones por gesto. Continuando de todas formas...")

    # — Normalización por articulación (compartida en qpos y en ctrl) —
    # (5-nov) con un mínimo de 0,01 rad: una articulación que apenas se mueve en los episodios
    # (p. ej. el brazo quieto en una tarea de dedos) no debe amplificar ×10⁶ cualquier
    # diferencia al reproducir.
    mu_x, sigma_x = x_arr.mean(axis=(0, 1)), np.maximum(x_arr.std(axis=(0, 1)), 1e-2)
    mu_y, sigma_y = y_arr.mean(axis=(0, 1)), np.maximum(y_arr.std(axis=(0, 1)), 1e-2)
    x_norm = (x_arr - mu_x) / sigma_x
    y_norm = (y_arr - mu_y) / sigma_y

    n = len(x_norm)
    episodio_arr = np.array(episodio_id_data)
    ids_episodios = np.unique(episodio_arr)
    n_ep = len(ids_episodios)

    if n_ep > 1:
        ids_perm  = np.random.permutation(ids_episodios)
        n_val_ep  = max(1, int(n_ep * 0.2))
        ids_val   = set(ids_perm[:n_val_ep].tolist())
        idx_val   = np.where(np.isin(episodio_arr, list(ids_val)))[0]
        idx_train = np.where(~np.isin(episodio_arr, list(ids_val)))[0]
        print(f"📊 Split por episodio: {n_ep - n_val_ep} episodio(s) train, "
              f"{n_val_ep} episodio(s) val (de {n_ep} totales).")
    else:
        # Solo hay 1 episodio válido: no existe forma honesta de aislar la validación.
        print("⚠️  Solo hay 1 episodio con datos suficientes: la validación no es "
              "independiente (ventanas del mismo episodio en train y val). "
              "Grabad más episodios para una validación fiable.")
        idx = np.random.permutation(n)
        n_val = max(1, int(n * 0.2))
        idx_val, idx_train = idx[:n_val], idx[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_ia   = ACTModel(n_joints=x_arr.shape[2], dim_accion=y_arr.shape[2]).to(device)
    optimizador = optim.Adam(modelo_ia.parameters(), lr=1e-4)

    x_train = torch.tensor(x_norm[idx_train], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_norm[idx_train], dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_norm[idx_val],   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_norm[idx_val],   dtype=torch.float32, device=device)

    MAX_EPOCAS    = IA_MAX_EPOCAS  # (8000) los Transformers necesitan bastantes más iteraciones que
                          # un MLP para converger — LeRobot recomienda del orden de
                          # ~100.000 pasos de entrenamiento para ACT en su casos de uso
    TAM_LOTE      = 64
    PACIENCIA     = IA_PACIENCIA   # (200) mucho más margen que antes — un Transformer puede tardar
                          # bastante en "arrancar" antes de que la validación mejore
                          # de forma consistente
    RUIDO_ENTRADA = 0.08
    mejor_val, mejor_estado, sin_mejora = float("inf"), None, 0
    n_train = x_train.shape[0]

    def _perdida(pred, mu, logvar, objetivo):
        recon = nn.functional.mse_loss(pred, objetivo)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + BETA_KL * kl, recon, kl

    for epoch in range(MAX_EPOCAS):
        modelo_ia.train()
        perm = torch.randperm(n_train, device=device)
        for ini in range(0, n_train, TAM_LOTE):
            lote = perm[ini:ini + TAM_LOTE]
            x_ruidoso = x_train[lote] + RUIDO_ENTRADA * torch.randn_like(x_train[lote])
            optimizador.zero_grad()
            pred, mu, logvar = modelo_ia(x_ruidoso, y_train[lote])
            loss, _, _ = _perdida(pred, mu, logvar, y_train[lote])
            loss.backward()
            optimizador.step()

        modelo_ia.eval()
        with torch.no_grad():
            pred_val, mu_val, logvar_val = modelo_ia(x_val, y_val)
            val_loss, val_recon, val_kl = _perdida(pred_val, mu_val, logvar_val, y_val)
            val_loss = val_loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch [{epoch+1}/{MAX_EPOCAS}]  Val total={val_loss:.5f}  "
                  f"(recon={val_recon.item():.5f}  kl={val_kl.item():.5f})")

        if val_loss < mejor_val - 1e-5:
            mejor_val = val_loss
            mejor_estado = {k: v.clone() for k, v in modelo_ia.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print(f"   ⏹  Sin mejora en validación durante {PACIENCIA} epochs. "
                      f"Paramos en epoch {epoch+1}.")
                break

    if mejor_estado is not None:
        modelo_ia.load_state_dict(mejor_estado)

    torch.save({
        "state_dict": modelo_ia.state_dict(),
        "mu_x": mu_x, "sigma_x": sigma_x,
        "mu_y": mu_y, "sigma_y": sigma_y,
        "ventana": VENTANA_IA,
        "chunk": CHUNK_IA,
        "dt_objetivo": dt_objetivo,
        "formato": formato,                 # 5-nov: 2 = brazo + mano
        "col_estado": col_estado,           # p. ej. ["q:joint1", …, "q:right_wrist", …]
        "col_accion": col_accion,           # p. ej. ["a:joint1", …]
        "dim_estado": int(x_arr.shape[2]),
        "dim_accion": int(y_arr.shape[2]),
    }, archivo_pesos)

    print(f"✅ ENTRENAMIENTO COMPLETADO (mejor Val={mejor_val:.5f}, "
          f"{episodios_usados} episodios, {len(x_arr)} muestras). "
          f"Pesos guardados en '{archivo_pesos}'.\n")
    return True


def entrenar_todas_las_tareas(episodios_dir: str = EPISODIOS_DIR,
                               modelos_dir: str = MODELOS_DIR):
    """Recorre cada subcarpeta de episodios/ (una por tarea) y entrena un modelo
    independiente para cada una, guardado en modelos/<tarea>.pth."""
    tareas = _listar_tareas_grabadas(episodios_dir)
    if not tareas:
        print(f"❌ No hay ninguna tarea grabada en '{episodios_dir}/'. Pulsa [G] primero.")
        return

    os.makedirs(modelos_dir, exist_ok=True)
    print(f"\n📋 Se van a entrenar {len(tareas)} tarea(s): {', '.join(tareas)}\n")

    for tarea in tareas:
        print(f"\n{'='*70}\n▶ Entrenando tarea: '{tarea}'\n{'='*70}")
        ruta_episodios = os.path.join(episodios_dir, tarea)
        ruta_pesos = os.path.join(modelos_dir, f"{tarea}.pth")
        entrenar_ia(ruta_episodios, ruta_pesos)

    print(f"\n🏁 Entrenamiento de todas las tareas completado. "
          f"Modelos disponibles en '{modelos_dir}/'.\n")


def cargar_ia(archivo_pesos: str = ARCHIVO_PESOS):
    """Carga pesos + estadísticas de normalización del modelo ACT. Devuelve un dict
    listo para inferencia, o None."""
    if not os.path.exists(archivo_pesos):
        print("❌ No hay IA entrenada.")
        return None
    try:
        chk = torch.load(archivo_pesos, map_location="cpu", weights_only=False)
        ventana = int(chk.get("ventana", VENTANA_IA))
        chunk   = int(chk.get("chunk", CHUNK_IA))
        dt_objetivo = float(chk.get("dt_objetivo", 1.0 / 15.0))
        modelo_ia = ACTModel(n_joints=int(chk.get("dim_estado", N_ACT)), ventana=ventana, chunk=chunk,
                             dim_accion=int(chk.get("dim_accion", N_ACT)))
        modelo_ia.load_state_dict(chk["state_dict"])
        modelo_ia.eval()
        print(f"✅ IA (ACT) cargada desde '{archivo_pesos}' "
              f"(ventana={ventana}, bloque={chunk}, ritmo={dt_objetivo*1000:.0f} ms/fotograma).")
        return {
            "modelo":  modelo_ia,
            "mu_x":    torch.as_tensor(chk["mu_x"],    dtype=torch.float32),
            "sigma_x": torch.as_tensor(chk["sigma_x"], dtype=torch.float32),
            "mu_y":    torch.as_tensor(chk["mu_y"],    dtype=torch.float32),
            "sigma_y": torch.as_tensor(chk["sigma_y"], dtype=torch.float32),
            "ventana": ventana,
            "chunk":   chunk,
            "dt_objetivo": dt_objetivo,
            "formato": int(chk.get("formato", 1)),
            "col_estado": chk.get("col_estado"),
            "col_accion": chk.get("col_accion"),
        }
    except Exception as e:
        print(f"❌ Error al cargar la IA: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCCIÓN: historial + predicción + ensamblado temporal (5-nov, fuera del bucle)
# ══════════════════════════════════════════════════════════════════════════════

class PoliticaACT:
    """Envuelve el modelo cargado (cargar_ia): con cada estado nuevo predice un bloque y
    devuelve la acción del instante actual combinando los bloques anteriores (temporal
    ensembling, pesos exp(−peso_exp·k), más peso a lo más reciente)."""

    def __init__(self, red, peso_exp=0.15):
        self.red = red
        self.peso_exp = float(peso_exp)
        self.historial = None
        self.bloques = deque(maxlen=int(red["chunk"]))

    def reiniciar(self, estado0):
        self.historial = deque([list(map(float, estado0))] * int(self.red["ventana"]),
                               maxlen=int(self.red["ventana"]))
        self.bloques.clear()

    def paso(self, estado=None):
        """estado: postura actual (se añade al historial antes de predecir; None = no
        añadir). Devuelve la acción (np.ndarray) del instante actual."""
        if self.historial is None:
            self.reiniciar(estado)
        elif estado is not None:
            self.historial.append(list(map(float, estado)))
        r = self.red
        x = (torch.tensor([list(self.historial)], dtype=torch.float32) - r["mu_x"]) / r["sigma_x"]
        with torch.no_grad():
            pred, _, _ = r["modelo"](x)
        self.bloques.append((pred[0] * r["sigma_y"] + r["mu_y"]).numpy())
        preds = [(k, b[k]) for k, b in enumerate(reversed(self.bloques)) if k < b.shape[0]]
        w = np.array([math.exp(-self.peso_exp * k) for k, _ in preds])
        w /= w.sum()
        return sum(wi * p for wi, (_, p) in zip(w, preds))


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Entrenar ACT con los episodios grabados ([G])")
    ap.add_argument("tarea", nargs="?", help="nombre de la tarea (carpeta de episodios/)")
    ap.add_argument("--todas", action="store_true", help="entrenar todas las tareas")
    a = ap.parse_args(argv)
    if a.todas:
        entrenar_todas_las_tareas()
    elif a.tarea:
        os.makedirs(MODELOS_DIR, exist_ok=True)
        ok = entrenar_ia(os.path.join(EPISODIOS_DIR, a.tarea), os.path.join(MODELOS_DIR, f"{a.tarea}.pth"))
        sys.exit(0 if ok else 1)
    else:
        print("Tareas grabadas:", ", ".join(_listar_tareas_grabadas()) or "(ninguna)")
        ap.print_help()


if __name__ == "__main__":
    _main()
