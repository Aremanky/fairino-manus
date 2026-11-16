"""
Cerebro IA: modelo ACT (Action Chunking Transformer), entrenamiento y carga.

(Extraído de Skill_Shield_Nivel_6.py.) 5-nov: brazo + mano (formato 2), 23 dimensiones.

16-nov — entrenamiento listo para agarrar (lo que cambia):
  · VE EL MUNDO (formato 3, ver episodios.py): el objeto visto desde la mano, los táctiles y la
    postura REAL de la mano y del brazo. Antes era «ciega» (solo historial de posturas).
  · Acción "mano_en_objeto": dónde poner la mano en el marco del objeto (la IK la pasa a joints),
    así el gesto se traslada con el objeto. "joints" (como antes) si no hay objeto.
  · Una muestra por CADA fotograma: el historial se rellena repitiendo el primero (como hace la
    reproducción al arrancar) y el bloque repitiendo la última acción (aprende a quedarse quieta
    al terminar). Además aprende el PROGRESO (0 → 1): la reproducción sabe cuándo ha terminado.
  · Historial de 1 fotograma si ve el mundo (con más, la red tiende a «seguir como iba» en vez de
    mirar); 8 si es ciega, como antes.
  · Pérdida L1 y KL sumado sobre la latente (como el ACT original/LeRobot), AdamW, gradiente
    recortado, presupuesto en pasos con validación periódica.
  · MULTITAREA: un modelo para varias tareas (p. ej. coger cada herramienta) con un token de tarea.
  · Checkpoints elegidos con el ENSAYO en lazo cerrado de los episodios de validación
    (evaluar.py), no solo con la pérdida de validación.
  · Episodios marcados como fallidos con [M] no se usan. Datos fuera del repo (episodios.directorios).
  · Los .pth antiguos (formatos 1 y 2) se siguen cargando y reproduciendo.

Entrenar FUERA de la app ([E] lo lanza así, en otro proceso):
    python -m skill_shield.ia.act <tarea>                     # una tarea → modelos/<tarea>.pth
    python -m skill_shield.ia.act --todas                     # cada tarea, su modelo
    python -m skill_shield.ia.act coger_pelota coger_destornillador --modelo agarre   # multitarea
Opciones: --accion auto|joints|mano_en_objeto · --ventana N · --chunk N · --pasos N · --lote N ·
          --incluir-fallidos · --sin-tactil · --sin-objeto · --cpu · --semilla N
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from config import *    # noqa: F401,F403  (N_ACT, ARM_JOINTS, ARCHIVO_PESOS, RUTA_XML…)
from skill_shield.ia.episodios import (Conjunto, compatibles, describir, directorios, disposicion, exito,
                                       leer_episodio, listar_episodios, listar_modelos, listar_tareas,
                                       nombre_seguro, receta_para)


# ══════════════════════════════════════════════════════════════════════════════
# CEREBRO IA  —  ACT (Action Chunking Transformer)
# ══════════════════════════════════════════════════════════════════════════════
# Basado en ACT (Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost
# Hardware"), la familia de ALOHA y LeRobot:
# 1. CVAE con variable latente de "estilo" (z): un Transformer codificador, que SOLO se usa al
#    entrenar, resume el bloque de acciones real en z; así el decodificador no promedia entre
#    gestos distintos. En inferencia z = 0 (la media del prior).
# 2. El decodificador usa "action queries" aprendidas (una por paso futuro) que atienden al
#    contexto (historial de estado + z + entorno + tarea) y dan el bloque de acciones de golpe.

VENTANA_IA     = 8     # historial de una red CIEGA (con el mundo, la receta pone 1)
CHUNK_IA       = 30    # acciones futuras que predice de una vez (1 s a 30 Hz)
DIM_LATENTE    = 32
DIM_MODELO     = 128
N_CABEZAS      = 4
N_CAPAS_ENC    = 4     # codificador de estilo
N_CAPAS_CTX    = 4     # contexto
N_CAPAS_DEC    = 1     # LeRobot usa 1 a propósito (en el ACT original solo se usaba la 1ª de 7)
BETA_KL        = 10.0  # peso del KL (estándar en ACT: 10)

IA_PASOS            = 20000   # pasos de entrenamiento (lotes)
IA_LOTE             = 64
IA_LR               = 1e-4
IA_WD               = 1e-4    # weight decay (AdamW)
IA_EVAL_CADA        = 250     # validar cada estos pasos
IA_PACIENCIA_EVALS  = 30      # validaciones seguidas sin mejorar → parar
IA_N_CHECKPOINTS    = 6       # checkpoints guardados a lo largo del entrenamiento (+ el mejor en val)
RUIDO_ENTRADA       = 0.08    # ruido en la entrada normalizada (robustez)
P_SIN_TACTIL        = 0.15    # a veces se le quitan los táctiles: sigue funcionando sin ellos
PESO_RELLENO        = 1.0     # peso de las acciones repetidas tras el final (quedarse quieta)
PESO_PROGRESO       = 1.0

EPISODIOS_DIR, MODELOS_DIR = directorios()    # cada TAREA: episodios/<tarea>/episodio_XXXX.csv

# nombres de antes (Skill_Shield_Nivel_6.py los importa)
_nombre_seguro = nombre_seguro
_listar_episodios = listar_episodios


def _listar_tareas_grabadas(episodios_dir=None):
    return listar_tareas(episodios_dir or EPISODIOS_DIR)


def _listar_modelos_entrenados(modelos_dir=None):
    return listar_modelos(modelos_dir or MODELOS_DIR)


class ACTModel(nn.Module):
    """CVAE + Transformer. dim_entorno > 0 → un token más con el objeto; n_tareas > 0 → un token de
    tarea; con_progreso → otra salida con el progreso. Sin esas tres, es exactamente el modelo de
    antes (los .pth antiguos cargan tal cual)."""

    def __init__(self, n_joints: int = N_ACT, ventana: int = VENTANA_IA, chunk: int = CHUNK_IA,
                 dim: int = DIM_MODELO, n_heads: int = N_CABEZAS, n_capas_enc: int = N_CAPAS_ENC,
                 n_capas_ctx: int = N_CAPAS_CTX, n_capas_dec: int = N_CAPAS_DEC, dim_z: int = DIM_LATENTE,
                 dim_accion: int = None, dim_entorno: int = 0, n_tareas: int = 0, con_progreso: bool = False):
        super().__init__()
        self.ventana, self.chunk, self.dim_z = ventana, chunk, dim_z
        dim_accion = n_joints if dim_accion is None else dim_accion

        # — Embeddings de entrada —
        self.embed_qpos   = nn.Linear(n_joints, dim)            # (el «estado»; nombre de antes)
        self.embed_action = nn.Linear(dim_accion, dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, 2 + chunk, dim))  # CLS + estado + chunk

        # — Codificador de estilo (solo con demostración real, al entrenar) —
        capa_enc = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                                              batch_first=True)
        self.style_encoder = nn.TransformerEncoder(capa_enc, num_layers=n_capas_enc)
        self.to_mu_logvar  = nn.Linear(dim, dim_z * 2)

        # — Política: contexto (historial + z [+ entorno] [+ tarea]) + action queries —
        self.latent_proj = nn.Linear(dim_z, dim)
        self.embed_entorno = nn.Linear(dim_entorno, dim) if dim_entorno else None
        self.embed_tarea = nn.Embedding(n_tareas, dim) if n_tareas else None
        n_extra = (1 if dim_entorno else 0) + (1 if n_tareas else 0)
        self.pos_embed_pol = nn.Parameter(torch.zeros(1, ventana + 1 + n_extra, dim))
        capa_pol_enc = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                                                  batch_first=True)
        self.policy_encoder = nn.TransformerEncoder(capa_pol_enc, num_layers=n_capas_ctx)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk, dim))
        capa_pol_dec = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                                                  batch_first=True)
        self.policy_decoder = nn.TransformerDecoder(capa_pol_dec, num_layers=n_capas_dec)
        self.cabeza_salida = nn.Linear(dim, dim_accion)
        self.cabeza_progreso = nn.Linear(dim, 1) if con_progreso else None

    def _codificar_estilo(self, estado_actual, action_chunk):
        b = action_chunk.shape[0]
        seq = torch.cat([self.cls_token.expand(b, -1, -1), self.embed_qpos(estado_actual).unsqueeze(1),
                         self.embed_action(action_chunk)], dim=1) + self.pos_embed_enc
        mu, logvar = self.to_mu_logvar(self.style_encoder(seq)[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def _decodificar(self, hist, z, entorno=None, tarea=None):
        b = hist.shape[0]
        tokens = [self.embed_qpos(hist), self.latent_proj(z).unsqueeze(1)]
        if self.embed_entorno is not None:
            tokens.append(self.embed_entorno(entorno).unsqueeze(1))
        if self.embed_tarea is not None:
            if tarea is None:
                tarea = torch.zeros(b, dtype=torch.long, device=hist.device)
            tokens.append(self.embed_tarea(tarea).unsqueeze(1))
        memoria = self.policy_encoder(torch.cat(tokens, dim=1) + self.pos_embed_pol)
        salida = self.policy_decoder(self.action_queries.expand(b, -1, -1), memoria)
        prog = self.cabeza_progreso(salida).squeeze(-1) if self.cabeza_progreso is not None else None
        return self.cabeza_salida(salida), prog

    def forward(self, hist, action_chunk=None, entorno=None, tarea=None):
        """Con action_chunk (entrenamiento): z del codificador de estilo. Sin él (inferencia): z = 0.
        → (acciones (B, chunk, Da), mu, logvar, progreso (B, chunk) en logits o None)."""
        b = hist.shape[0]
        if action_chunk is not None:
            mu, logvar = self._codificar_estilo(hist[:, -1, :], action_chunk)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(b, self.dim_z, device=hist.device)
        pred, prog = self._decodificar(hist, z, entorno, tarea)
        return pred, mu, logvar, prog


def _modelo_desde_arq(arq):
    return ACTModel(n_joints=int(arq["dim_estado"]), ventana=int(arq["ventana"]), chunk=int(arq["chunk"]),
                    dim=int(arq.get("dim", DIM_MODELO)), n_heads=int(arq.get("n_cabezas", N_CABEZAS)),
                    n_capas_enc=int(arq.get("n_capas_enc", N_CAPAS_ENC)),
                    n_capas_ctx=int(arq.get("n_capas_ctx", N_CAPAS_CTX)),
                    n_capas_dec=int(arq.get("n_capas_dec", N_CAPAS_DEC)), dim_z=int(arq.get("dim_z", DIM_LATENTE)),
                    dim_accion=int(arq["dim_accion"]), dim_entorno=int(arq.get("dim_entorno", 0)),
                    n_tareas=int(arq.get("n_tareas", 0)), con_progreso=bool(arq.get("con_progreso", False)))


# ══════════════════════════════════════════════════════════════════════════════
# CARGA (modelos nuevos y antiguos)
# ══════════════════════════════════════════════════════════════════════════════

def _receta_legado(chk, nombre):
    """Receta equivalente a un .pth de antes del 16-nov (ciego: historial de posturas → comandos)."""
    ventana, chunk = int(chk.get("ventana", VENTANA_IA)), int(chk.get("chunk", CHUNK_IA))
    if int(chk.get("formato", 1)) == 2 and chk.get("col_estado"):
        nombres = [c[2:] for c in chk["col_estado"]]
        brazo = [n for n in nombres if n in ARM_JOINTS]
        mano = [n for n in nombres if n not in ARM_JOINTS]
        if nombres != brazo + mano or [c[2:] for c in chk.get("col_accion") or []] != nombres:
            raise ValueError("modelo antiguo con un orden de columnas inesperado")
    else:
        brazo, mano = [], None
    n_m = len(mano) if mano else int(chk.get("dim_estado", N_ACT))
    estado = ([["q_brazo", len(brazo)]] if brazo else []) + [["q_mano", n_m]]
    return dict(version=0, legado=True, formato=int(chk.get("formato", 1)), nombres_brazo=brazo,
                nombres_mano=mano, n_brazo=len(brazo), n_mano=n_m, accion="joints", estado=estado, entorno=[],
                clases=[], tareas=[nombre], ventana=ventana, chunk=chunk,
                fps=1.0 / float(chk.get("dt_objetivo", 1.0 / 15.0)), usar_mano_real=False, usar_brazo_real=False)


def red_desde_paquete(chk, dispositivo="cpu", nombre="modelo"):
    """Un .pth ya leído (o un paquete recién entrenado) → `red` para politica.Politica: dict con
    modelo, receta, norm, predecir(hist, entorno, tarea) → (acciones (C, Da), progreso (C,) o None)."""
    dev = torch.device(dispositivo)
    if "receta" in chk:                                              # 16-nov en adelante
        receta, norm, arq = chk["receta"], chk["norm"], chk["arquitectura"]
        modelo = _modelo_desde_arq(arq)
        legado = False
    else:                                                            # formatos 1 y 2 (antes)
        receta = _receta_legado(chk, nombre)
        modelo = ACTModel(n_joints=int(chk.get("dim_estado", N_ACT)), ventana=receta["ventana"],
                          chunk=receta["chunk"], dim_accion=int(chk.get("dim_accion", N_ACT)))
        norm = dict(estado=(chk["mu_x"], chk["sigma_x"]), entorno=None, accion=(chk["mu_y"], chk["sigma_y"]))
        legado = True
    modelo.load_state_dict(chk["state_dict"])
    modelo.to(dev).eval()

    def _t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)

    mu_s, sd_s = _t(norm["estado"][0]), _t(norm["estado"][1])
    mu_a, sd_a = _t(norm["accion"][0]), _t(norm["accion"][1])
    ne = norm.get("entorno")
    mu_e, sd_e = (_t(ne[0]), _t(ne[1])) if ne is not None else (None, None)

    def predecir(hist, entorno=None, tarea=0):
        with torch.no_grad():
            x = (torch.as_tensor(np.asarray(hist, np.float32), device=dev)[None] - mu_s) / sd_s
            e = None
            if modelo.embed_entorno is not None:
                if entorno is None:
                    raise ValueError("este modelo necesita el objeto (entorno)")
                e = (torch.as_tensor(np.asarray(entorno, np.float32), device=dev)[None] - mu_e) / sd_e
            t = (torch.tensor([int(tarea)], dtype=torch.long, device=dev)
                 if modelo.embed_tarea is not None else None)
            pred, _, _, prog = modelo(x, entorno=e, tarea=t)
            acc = (pred[0] * sd_a + mu_a).cpu().numpy().astype(float)
            pr = torch.sigmoid(prog[0]).cpu().numpy().astype(float) if prog is not None else None
        return acc, pr

    return dict(modelo=modelo, receta=receta, norm=norm, predecir=predecir, legado=legado, nombre=nombre,
                dt_objetivo=1.0 / float(receta.get("fps") or 15.0), ventana=receta["ventana"], chunk=receta["chunk"],
                tareas=list(receta.get("tareas") or [nombre]), info=chk.get("info", {}))


def cargar_ia(archivo_pesos: str = None, dispositivo="cpu", silencioso=False):
    """Carga un .pth (nuevo o antiguo) → `red` (ver red_desde_paquete), o None."""
    archivo_pesos = archivo_pesos or ARCHIVO_PESOS
    if not os.path.exists(archivo_pesos):
        print(f"❌ No hay IA entrenada en '{archivo_pesos}'.")
        return None
    try:
        chk = torch.load(archivo_pesos, map_location="cpu", weights_only=False)
        nombre = os.path.splitext(os.path.basename(archivo_pesos))[0]
        red = red_desde_paquete(chk, dispositivo, nombre=nombre)
        if not silencioso:
            print(f"✅ IA (ACT) cargada desde '{archivo_pesos}'"
                  + (" (modelo antiguo, ciego)" if red["legado"] else "") + f": {describir(red['receta'])}")
        return red
    except Exception as e:
        print(f"❌ Error al cargar la IA ({archivo_pesos}): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def _columnas(grupos, nombres):
    """Índices de las columnas de los grupos 'nombres'."""
    return [i for g, i0, i1 in disposicion(grupos) if g in nombres for i in range(i0, i1)]


def entrenar(tareas, modelo=None, episodios_dir=None, modelos_dir=None, accion="auto", ventana=None,
             chunk=CHUNK_IA, pasos=IA_PASOS, lote=IA_LOTE, lr=IA_LR, incluir_fallidos=False, con_tactil=None,
             con_objeto=None, semilla=0, cpu=False, eval_cada=IA_EVAL_CADA, paciencia=IA_PACIENCIA_EVALS,
             n_checkpoints=IA_N_CHECKPOINTS, ensayar=True):
    """Entrena UN modelo con los episodios de una o varias tareas. Devuelve la ruta del .pth o None."""
    episodios_dir = episodios_dir or EPISODIOS_DIR
    modelos_dir = modelos_dir or MODELOS_DIR
    tareas = list(tareas)
    nombre = modelo or (tareas[0] if len(tareas) == 1 else "_".join(tareas))
    t_ini = time.time()
    print(f"\n🧠 ENTRENANDO '{nombre}' con {len(tareas)} tarea(s): {', '.join(tareas)}")

    # ── 1. episodios ──
    eps, tarea_de, n_fallidos = [], [], 0
    for ti, t in enumerate(tareas):
        rutas = listar_episodios(os.path.join(episodios_dir, t))
        if not rutas:
            print(f"❌ No hay episodios en '{os.path.join(episodios_dir, t)}'. Graba con [G] primero.")
            return None
        for r in rutas:
            ep = leer_episodio(r)
            if ep is None:
                print(f"⚠️  No se puede leer {r}: se ignora.")
                continue
            if not incluir_fallidos and not exito(ep):
                n_fallidos += 1
                continue
            ep["tarea"] = t
            eps.append(ep)
            tarea_de.append(ti)
    if n_fallidos:
        print(f"ℹ️  {n_fallidos} episodio(s) marcado(s) como fallidos ([M]): no se usan.")
    usados, avisos = compatibles(eps)
    for a in avisos:
        print("⚠️  " + a)
    quedan = {id(e) for e in usados}
    tarea_de = [t for e, t in zip(eps, tarea_de) if id(e) in quedan]
    eps = [e for e in eps if id(e) in quedan]
    if not eps:
        print("❌ No queda ningún episodio con el que entrenar.")
        return None

    # ── 2. qué ve y qué hace la red ──
    try:
        receta = receta_para(eps, tareas, accion=accion, ventana=ventana, chunk=chunk,
                             con_tactil=con_tactil, con_objeto=con_objeto)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    print(f"📋 {describir(receta)}")
    print(f"   postura de la mano: {'REAL' if receta['usar_mano_real'] else 'simulada'} · "
          f"brazo: {'REAL' if receta['usar_brazo_real'] else 'simulado/gemelo'} · {receta['fps']:.0f} Hz")
    cadena = None
    if receta["entorno"] or receta["accion"] == "mano_en_objeto":
        from skill_shield.teleop_fr5.cinematica import CadenaFR5
        cadena = CadenaFR5.desde_mjcf(RUTA_XML)
    try:
        conj = Conjunto(receta, eps, tarea_de, cadena)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    for r, m in conj.saltados:
        print(f"⚠️  {os.path.basename(os.path.dirname(r))}/{os.path.basename(r)}: {m} → no se usa")
    idx_tr, idx_va, eps_va = conj.dividir(0.2, semilla)
    print(f"📊 {conj.n_episodios} episodios, {conj.n} muestras · validación: {len(eps_va)} episodio(s) enteros")
    if not len(idx_va):
        print("⚠️  Una sola demo por tarea: la validación NO es independiente. Grabad más episodios.")
    if conj.n_episodios < 10:
        print("⚠️  Pocas demos: para que generalice, del orden de 50 por tarea (posiciones y "
              "direcciones de llegada distintas).")
    norm = conj.normalizacion(idx_tr)

    # ── 3. tensores en el dispositivo ──
    torch.manual_seed(semilla)
    dev = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    f32 = dict(dtype=torch.float32, device=dev)
    S = torch.as_tensor((conj.S - norm["estado"][0]) / norm["estado"][1], **f32)
    En = torch.as_tensor((conj.E - norm["entorno"][0]) / norm["entorno"][1], **f32) if conj.E is not None else None
    A = torch.as_tensor((conj.A - norm["accion"][0]) / norm["accion"][1], **f32)
    OI = torch.as_tensor(conj.obs_idx, device=dev)
    AI = torch.as_tensor(conj.act_idx, device=dev)
    PAD = torch.as_tensor(conj.pad, device=dev)
    PROG = torch.as_tensor(conj.prog, **f32)
    TAR = torch.as_tensor(conj.tarea, device=dev)
    tr = torch.as_tensor(idx_tr, device=dev)
    va = torch.as_tensor(idx_va if len(idx_va) else idx_tr[:2048], device=dev)
    # ruido solo en lo continuo (no en banderas ni one-hot) · columnas de los táctiles (para quitarlos)
    def _vector(n, columnas, valor, dtype=torch.float32):
        v = torch.zeros(n, dtype=dtype, device=dev)
        if columnas:
            v[torch.as_tensor(columnas, dtype=torch.long, device=dev)] = valor
        return v

    ruido_s = _vector(S.shape[1], _columnas(receta["estado"], ("q_brazo", "q_mano", "tactil")), RUIDO_ENTRADA)
    ruido_e = (_vector(En.shape[1], _columnas(receta["entorno"], ("obj_pos", "obj_rot", "obj_tam")), RUIDO_ENTRADA)
               if En is not None else None)
    mascara_tac = _vector(S.shape[1], _columnas(receta["estado"], ("tactil", "tactil_ok")), True, torch.bool)
    hay_tac = bool(mascara_tac.any())

    def reunir(idx):
        return (S[OI[idx]], None if En is None else En[idx], A[AI[idx]], PAD[idx], PROG[idx], TAR[idx])

    # ── 4. modelo ──
    arq = dict(dim_estado=int(S.shape[1]), dim_accion=int(A.shape[1]), dim_entorno=0 if En is None else int(En.shape[1]),
               n_tareas=len(tareas), ventana=int(receta["ventana"]), chunk=int(receta["chunk"]), con_progreso=True,
               dim=DIM_MODELO, n_cabezas=N_CABEZAS, n_capas_enc=N_CAPAS_ENC, n_capas_ctx=N_CAPAS_CTX,
               n_capas_dec=N_CAPAS_DEC, dim_z=DIM_LATENTE)
    red_nn = _modelo_desde_arq(arq).to(dev)
    opt = torch.optim.AdamW(red_nn.parameters(), lr=lr, weight_decay=IA_WD)
    print(f"🖥️  {dev} · {sum(p.numel() for p in red_nn.parameters()) / 1e6:.2f} M parámetros · "
          f"hasta {pasos} pasos de {lote}")

    def perdidas(pred, mu, logvar, prog_pred, y, pad, prog):
        w = (1.0 + (PESO_RELLENO - 1.0) * pad.float()).unsqueeze(-1)
        l1 = (torch.abs(pred - y) * w).sum() / (w.sum() * y.shape[-1])
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean() if mu is not None else pred.new_zeros(())
        lp = torch.abs(torch.sigmoid(prog_pred) - prog).mean() if prog_pred is not None else pred.new_zeros(())
        return l1 + BETA_KL * kl + PESO_PROGRESO * lp, l1, kl, lp

    @torch.no_grad()
    def validar():
        """L1 en validación con z = 0 (como al reproducir) y error del progreso."""
        red_nn.eval()
        tot, totp, n = 0.0, 0.0, 0
        for i in range(0, len(va), 512):
            x, e, y, _, prog, tar = reunir(va[i:i + 512])
            pred, _, _, pp = red_nn(x, entorno=e, tarea=tar)
            tot += torch.abs(pred - y).mean(dim=(1, 2)).sum().item()
            totp += torch.abs(torch.sigmoid(pp) - prog).mean(dim=1).sum().item()
            n += len(x)
        red_nn.train()
        return tot / max(n, 1), totp / max(n, 1)

    def paquete(estado, metricas):
        return dict(version_ia=3, state_dict=estado, arquitectura=arq, receta=receta,
                    norm={k: (None if v is None else (np.asarray(v[0]), np.asarray(v[1]))) for k, v in norm.items()},
                    dt_objetivo=1.0 / float(receta["fps"]), metricas=metricas,
                    info=dict(nombre=nombre, fecha=time.strftime("%Y-%m-%d %H:%M:%S"),
                              episodios=[f"{os.path.basename(os.path.dirname(r))}/{os.path.basename(r)}" for r in conj.rutas],
                              episodios_val=[f"{os.path.basename(os.path.dirname(conj.rutas[i]))}/"
                                             f"{os.path.basename(conj.rutas[i])}" for i in eps_va],
                              semilla=semilla))

    def copia_estado():
        return {k: v.detach().to("cpu").clone() for k, v in red_nn.state_dict().items()}

    # ── 5. bucle ──
    dir_ckpt = os.path.join(modelos_dir, nombre)
    os.makedirs(dir_ckpt, exist_ok=True)
    for f in os.listdir(dir_ckpt):                 # los de un entrenamiento anterior de este modelo
        if f.endswith(".pth"):
            os.remove(os.path.join(dir_ckpt, f))
    ckpt_cada = max(eval_cada, (pasos // max(n_checkpoints, 1)) // eval_cada * eval_cada)
    candidatos = []                                 # (nombre, paquete)
    mejor_val, mejor_estado, mejor_paso, sin_mejora = float("inf"), None, 0, 0
    perm, pos, n_tr = torch.randperm(len(tr), device=dev), 0, len(tr)
    red_nn.train()
    paso = 0
    for paso in range(1, pasos + 1):
        if pos + lote > n_tr:
            perm, pos = torch.randperm(n_tr, device=dev), 0
        b = tr[perm[pos:pos + lote]]
        pos += lote
        x, e, y, pad, prog, tar = reunir(b)
        x = x + ruido_s * torch.randn_like(x)
        if hay_tac and P_SIN_TACTIL > 0:          # sin táctiles: valores a 0 y bandera a 0
            quitar = (torch.rand(len(b), device=dev) < P_SIN_TACTIL)[:, None, None] & mascara_tac[None, None, :]
            x = torch.where(quitar, torch.zeros_like(x), x)
        if e is not None:
            e = e + ruido_e * torch.randn_like(e)
        pred, mu, logvar, pp = red_nn(x, action_chunk=y, entorno=e, tarea=tar)
        loss, l1, kl, lp = perdidas(pred, mu, logvar, pp, y, pad, prog)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(red_nn.parameters(), 10.0)
        opt.step()
        if paso % eval_cada == 0 or paso == pasos:
            v, vp = validar()
            marca = ""
            if v < mejor_val - 1e-6:
                mejor_val, mejor_estado, mejor_paso, sin_mejora, marca = v, copia_estado(), paso, 0, " ⭐"
            else:
                sin_mejora += 1
            print(f"   paso {paso:6d}  train L1={l1.item():.4f} KL={kl.item():.3f} prog={lp.item():.3f}  "
                  f"val L1={v:.4f} prog={vp:.3f}{marca}", flush=True)
            if paso % ckpt_cada == 0:
                candidatos.append((f"paso_{paso:06d}", paquete(copia_estado(), dict(paso=paso, val_l1=v))))
            if sin_mejora >= paciencia:
                print(f"   ⏹  {paciencia} validaciones sin mejorar: paro en el paso {paso}.")
                break
    if not candidatos or candidatos[-1][0] != f"paso_{paso:06d}":
        v, _ = validar()
        candidatos.append((f"paso_{paso:06d}", paquete(copia_estado(), dict(paso=paso, val_l1=v))))
    if mejor_estado is not None:
        candidatos.append(("mejor_val", paquete(mejor_estado, dict(paso=mejor_paso, val_l1=mejor_val))))
    for nom, paq in candidatos:
        torch.save(paq, os.path.join(dir_ckpt, f"{nom}.pth"))

    # ── 6. elegir: ensayo en lazo cerrado con los episodios de validación ──
    elegido = candidatos[-1]                       # el mejor en validación (o el último, si no lo hay)
    informe = []
    if ensayar and eps_va:
        from skill_shield.ia.evaluar import evaluar, resumen
        eps_val = [conj.episodios[i] for i in eps_va]
        tar_val = [conj.tarea_ep[i] for i in eps_va]
        print(f"🧪 Ensayo en lazo cerrado de {len(candidatos)} checkpoint(s) con {len(eps_val)} episodio(s) de validación:")
        mejor_p = float("inf")
        for nom, paq in candidatos:
            try:
                p, res = evaluar(red_desde_paquete(paq, "cpu", nombre), eps_val, tar_val, cadena=cadena,
                                 v_max_brazo=float(globals().get("IA_V_MAX_BRAZO", 1.0)),
                                 lim_brazo=LIMITES_CTRL_BRAZO, lim_mano=LIMITES_CTRL)
            except Exception as ex:                     # la evaluación nunca debe perder el modelo
                print(f"   {nom:14s} no se pudo ensayar: {ex}")
                continue
            paq["metricas"]["ensayo"] = p
            informe.append(dict(checkpoint=nom, puntuacion=p, val_l1=paq["metricas"]["val_l1"], resultados=res))
            print(f"   {nom:14s} puntuación {p:7.2f} (val L1 {paq['metricas']['val_l1']:.4f}) · {resumen(res)}")
            if p <= mejor_p:                            # empate → el más entrenado (o el mejor en val)
                mejor_p, elegido = p, (nom, paq)
    ruta = os.path.join(modelos_dir, f"{nombre}.pth")
    torch.save(elegido[1], ruta)
    with open(os.path.join(dir_ckpt, "informe.json"), "w", encoding="utf-8") as f:
        json.dump(dict(elegido=elegido[0], receta=receta, checkpoints=informe,
                       episodios=elegido[1]["info"]["episodios"], episodios_val=elegido[1]["info"]["episodios_val"]),
                  f, indent=1, ensure_ascii=False, default=float)
    print(f"✅ ENTRENAMIENTO COMPLETADO en {(time.time() - t_ini) / 60:.1f} min · elegido '{elegido[0]}' → {ruta}\n"
          f"   (checkpoints e informe en {dir_ckpt}/)\n")
    return ruta


def entrenar_ia(episodios_dir=None, archivo_pesos=None, **kw) -> bool:
    """Como antes: entrena la tarea de la carpeta 'episodios_dir' y guarda en 'archivo_pesos'."""
    episodios_dir = episodios_dir or EPISODIOS_DIR
    tarea = os.path.basename(os.path.normpath(episodios_dir))
    modelos_dir = os.path.dirname(os.path.abspath(archivo_pesos)) if archivo_pesos else MODELOS_DIR
    nombre = os.path.splitext(os.path.basename(archivo_pesos))[0] if archivo_pesos else tarea
    return entrenar([tarea], modelo=nombre, episodios_dir=os.path.dirname(os.path.normpath(episodios_dir)),
                    modelos_dir=modelos_dir, **kw) is not None


def entrenar_todas_las_tareas(episodios_dir=None, modelos_dir=None, **kw):
    """Cada subcarpeta de episodios/ (una por tarea) → su propio modelo, modelos/<tarea>.pth."""
    episodios_dir, modelos_dir = episodios_dir or EPISODIOS_DIR, modelos_dir or MODELOS_DIR
    tareas = listar_tareas(episodios_dir)
    if not tareas:
        print(f"❌ No hay ninguna tarea grabada en '{episodios_dir}/'. Pulsa [G] primero.")
        return
    os.makedirs(modelos_dir, exist_ok=True)
    print(f"\n📋 Se van a entrenar {len(tareas)} tarea(s): {', '.join(tareas)}\n")
    for t in tareas:
        print(f"\n{'=' * 70}\n▶ Entrenando tarea: '{t}'\n{'=' * 70}")
        entrenar([t], episodios_dir=episodios_dir, modelos_dir=modelos_dir, **kw)
    print(f"\n🏁 Entrenamiento de todas las tareas completado. Modelos en '{modelos_dir}/'.\n")


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Entrenar ACT con los episodios grabados ([G])")
    ap.add_argument("tareas", nargs="*", help="tarea(s) (carpetas de episodios/)")
    ap.add_argument("--todas", action="store_true", help="cada tarea grabada, su propio modelo")
    ap.add_argument("--modelo", help="nombre del modelo (obligatorio con varias tareas juntas)")
    ap.add_argument("--accion", default="auto", choices=("auto", "joints", "mano_en_objeto"))
    ap.add_argument("--ventana", type=int, default=None, help="historial (por defecto 1 con el mundo, 8 ciega)")
    ap.add_argument("--chunk", type=int, default=CHUNK_IA)
    ap.add_argument("--pasos", type=int, default=IA_PASOS)
    ap.add_argument("--lote", type=int, default=IA_LOTE)
    ap.add_argument("--incluir-fallidos", action="store_true", help="usar también los marcados con [M]")
    ap.add_argument("--sin-tactil", action="store_true")
    ap.add_argument("--sin-objeto", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="entrenar en la CPU aunque haya GPU")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args(argv)
    kw = dict(accion=a.accion, ventana=a.ventana, chunk=a.chunk, pasos=a.pasos, lote=a.lote,
              incluir_fallidos=a.incluir_fallidos, con_tactil=False if a.sin_tactil else None,
              con_objeto=False if a.sin_objeto else None, cpu=a.cpu, semilla=a.semilla)
    if a.todas:
        entrenar_todas_las_tareas(**kw)
        return 0
    if not a.tareas:
        print("Tareas grabadas:", ", ".join(_listar_tareas_grabadas()) or "(ninguna)")
        ap.print_help()
        return 0
    if len(a.tareas) > 1 and not a.modelo:
        print("❌ Con varias tareas en un mismo modelo, dale nombre: --modelo <nombre>")
        return 2
    os.makedirs(MODELOS_DIR, exist_ok=True)
    return 0 if entrenar(a.tareas, modelo=a.modelo, **kw) else 1


if __name__ == "__main__":
    sys.exit(_main())
