# YOLO de herramientas de obra y su mango — paso a paso

Un único modelo de **segmentación** (YOLO26-seg) que, en cada imagen de la cámara de cabeza, devuelve la
máscara de cada herramienta **y** la de su mango como una instancia aparte de la clase `mango`. Las clases
están en `clases.yaml` (el orden es el id que ve YOLO: no lo cambiéis una vez entrenado; las nuevas, al final).

Solo **herramientas de obra e instalaciones**, 25 más el mango:

| Grupo | Clases |
|---|---|
| De mano | martillo, alicates, destornillador, llave, carraca, llave_allen, mazo, cincel, palanca, lima |
| Eléctricas | taladro (también atornillador y percutor), amoladora |
| Corte | sierra, cuter, tijera_chapa |
| Medir | flexometro, nivel, escuadra |
| Albañilería | paleta, pala, espatula |
| Montaje | sargento, pistola_silicona, grapadora (de obra), remachadora |

Lo demás que sale en los datasets (tijeras, cuchillos de cocina, oficina, jardín, EPI…) no se aprende:
queda como fondo. Cada fuente convertida guarda en su `info.json` la lista de clases con la que se hizo, y
`unir_datasets.py` deja fuera la que no coincida con la actual: si cambiáis `clases.yaml`, hay que volver a
convertir (lo pasado por SAM no se repite).

Todo se lanza **desde la raíz del repo**. Los datasets van FUERA del repo, en `$HERRAMIENTAS_DATOS`
(por defecto `~/datasets_herramientas`): ponedlo en un disco con sitio.

```bash
echo 'export HERRAMIENTAS_DATOS=$HOME/datasets_herramientas' >> ~/.bashrc    # una vez (u otra carpeta)
source ~/.bashrc
```
Exportadla siempre, aunque sea la de por defecto: los comandos de este LEEME la usan. (`revisar.py` y los
conversores entienden también `/convertidos/paco` con la variable vacía, o solo el nombre: `revisar.py paco`.)

---
## 1. Instalar (una vez)

```bash
./yolo_herramientas/instalar.sh          # crea venv_yolo: PyTorch para la RTX 5080 + Ultralytics + lo demás
source venv_yolo/bin/activate            # en cada terminal donde se vaya a trabajar con esto
```
Es un entorno **aparte** de `venv_nivel6`: Ultralytics instala `opencv-python` y en `venv_nivel6` eso
rompe `cv2.aruco`. Al final dice si PyTorch ve la GPU; si no la ve, no sigas (entrenaría en la CPU).

---
## Camino rápido: solo datasets públicos (`publicos.sh`)

Sin fotos propias, en dos fases, con lo público que hay de herramientas de obra:

| Fuente | Qué aporta | Mango |
|---|---|---|
| HANDAL (Google Drive) | martillos, alicates, destornilladores, llaves, carracas y taladros; decenas de miles de imágenes. Se baja solo, sin las categorías de cocina | sí |
| PACO | las mismas, pocas (unas 150 imágenes) | sí |
| Small Size Construction Tools (Zenodo) | 25 084 fotos de obra de interior, la cuarta parte en obras reales: taladro, amoladora, martillo, sierra, cúter, llaves, paleta, pala, grapadora de obra | no (lo pone la fase 1) |
| Open Images V7 | martillo, destornillador, taladro, llave, cincel (unas 400 imágenes) | no |
| Roboflow Universe | herramientas de mano y de obra en fotos variadas (7 proyectos) | no |

1. **FASE 1 «mango»:** se entrena con HANDAL + PACO, que son las únicas fuentes con el mango marcado.
2. **COMPLETAR:** ese modelo marca el mango dentro de cada herramienta de las demás fuentes.
3. **FASE 2 «obra»:** se entrena con todo junto. El resultado es el modelo final, `herramientas_obra`.

```bash
source venv_yolo/bin/activate
export ROBOFLOW_API_KEY=xxxxxxxx      # opcional: la «Private API Key» (app.roboflow.com → Settings → API Keys)
./yolo_herramientas/publicos.sh descargar           # PACO, Open Images, Roboflow, HANDAL y el de obra
./yolo_herramientas/publicos.sh resumen             # qué está ya y qué falta
python yolo_herramientas/convertir_handal.py --inspeccionar  # comprobar dónde está el mango (ver 2b)
python yolo_herramientas/convertir_obra.py --inspeccionar    # comprobar el de obra (ver 2e)
./yolo_herramientas/publicos.sh convertir           # todo a nuestras clases; las cajas, con SAM 2 (horas)
#    → mira los mosaicos de $HERRAMIENTAS_DATOS/convertidos/*/revision/
./yolo_herramientas/publicos.sh fase1               # entrena mango_v1
./yolo_herramientas/publicos.sh completar           # → mira convertidos/*_mango/revision/
./yolo_herramientas/publicos.sh fase2               # entrena herramientas_obra (el modelo final)
```
HANDAL y el de obra son lo largo (decenas de GB cada uno). Se pueden bajar a la vez en dos terminales
(`./yolo_herramientas/descargar.sh handal` y `./yolo_herramientas/descargar.sh obra`); si se cortan, se
relanzan y siguen.

Cada paso se puede repetir: lo ya bajado y lo ya pasado por SAM no se rehace. Hay que mirar los mosaicos
entre pasos. Las máscaras que salen de cajas con SAM y los mangos que pone la fase 1 son automáticos, y
un error ahí solo se ve a ojo.

- **Más herramientas:** añadid su nombre en `clases` (al final) y en `nombres` de `clases.yaml`. Después se
  añaden más fuentes: `open_images.clases`, `dataset_ninja` o más proyectos en `roboflow`.
- **Fondo:** en las fuentes de cajas, las fotos sin ninguna herramienta de obra se quedan como mucho en un
  10 % de las que sí tienen (`convertir_yolo_seg.py --max-fondo`), y SAM solo procesa las cajas de clases
  que están en `nombres` (`cajas_a_mascaras.py --todas` para todas).
- **Equilibrio:** HANDAL tiene muchas más imágenes que el resto. En la fase 2 se limita a `MAX_HANDAL=15000`
  para que no se coma a las demás.
- **Modelo final:** `runs/segment/herramientas_obra/weights/best.pt`. Para probarlo en vivo, el paso 6.

Los pasos de abajo son lo mismo, una fuente cada vez, por si hay que repetir o revisar una sola.

---
## 2. Datasets públicos → nuestro formato

Cada `convertir_*.py` deja su fuente en `$HERRAMIENTAS_DATOS/convertidos/<fuente>/` ya con nuestras clases.
**Después de cada conversión, mírala** con `revisar.py` (herramienta en color, mango en verde con borde
blanco): un error de conversión no lo detecta el entrenamiento, solo lo empeora.

### 2a. PACO (pocas herramientas de obra, pero con el mango marcado)
```bash
./yolo_herramientas/descargar.sh paco
python yolo_herramientas/convertir_paco.py --listar     # cuántas hay de cada herramienta y de cada parte
python yolo_herramientas/convertir_paco.py              # baja solo las imágenes de COCO que hacen falta
python yolo_herramientas/revisar.py paco
```

### 2b. HANDAL (la fuente principal del mango)
1. Baja **solo** las categorías de herramientas (martillos, alicates, destornilladores, llaves, carracas y
   taladros) desde su carpeta de Google Drive, con `gdown`. Cada zip se descomprime sin las carpetas
   `depth/` (no se usan) y después se borra:
   ```bash
   python yolo_herramientas/descargar_handal.py --listar    # qué hay en la carpeta y qué se bajaría
   ./yolo_herramientas/descargar.sh handal                  # bajar (se puede cortar y relanzar)
   ```
   Si Google corta («demasiadas descargas de este archivo»), relánzalo más tarde, o baja ese zip a mano
   desde el navegador y déjalo en `$HERRAMIENTAS_DATOS/handal/zips/`: al relanzar se descomprime igual.
2. **Inspecciona** antes de convertir:
   ```bash
   python yolo_herramientas/convertir_handal.py --inspeccionar
   ```
   La última línea dice de dónde va a sacar el mango. Si dice «NO se ha encontrado el mango», no conviertas
   todavía: pasa la salida completa de `--inspeccionar` para ajustar el conversor.
3. Convierte (1 imagen de cada 10 de cada vídeo; los fotogramas seguidos son casi iguales) y revisa:
   ```bash
   python yolo_herramientas/convertir_handal.py --cada 10
   python yolo_herramientas/revisar.py handal --n 48
   ```

### 2c. Roboflow Universe (opcional; sin mango)
```bash
export ROBOFLOW_API_KEY=xxxxxxxx         # la «Private API Key» (la que empieza por rf_ es la publicable: no sirve)
./yolo_herramientas/descargar.sh roboflow
```
Los proyectos están en `clases.yaml` → `roboflow`, como `workspace/proyecto` de su URL. Se baja la última
versión de cada uno (o la de `version: N` si se pone); uno que no exista se salta con un aviso. Las cajas
pasan por SAM y los polígonos se quedan como vienen. No traen el mango: lo pone la fase 1.

### 2d. Datasets solo con cajas (Open Images, Dataset Ninja…)
Open Images baja primero unos 3 GB de índices; si se corta (o se pulsa Ctrl+C), al relanzar sigue por donde
iba. La primera vez se queda unos minutos leyendo un índice de 2 GB sin escribir nada: no está colgado.
METU-ALET (Dataset Ninja) está fuera de `clases.yaml`: en noviembre de 2026 su descarga está rota (la
automática devuelve una página web y la del navegador llega cortada). Si vuelve a funcionar, basta con
`dataset_ninja: [METU-ALET]`; un `.tar` cortado se aparta como `.incompleto` sin dejar nada a medias.

`convertir_supervisely.py` (los de Dataset Ninja) o `descargar_open_images.py` los dejan en formato YOLO de
detección; `cajas_a_mascaras.py` les saca la máscara con SAM 2 y `convertir_yolo_seg.py` los pasa a nuestras
clases. Tampoco traen mango (ver paso 7). `publicos.sh convertir` hace todo esto solo.

### 2e. Small Size Construction Tools (Zenodo, la fuente principal de obra)
Lee, Jeon y Shin (2023), CC BY 4.0: 25 084 fotos de 12 herramientas de obra de interior, con cajas.
Son cuatro zip, unos 110 GB en total; se leen sin descomprimir y las imágenes se guardan a 1280 px.
Zenodo da más o menos 1 MB/s por conexión, así que los zip que falten se bajan **a la vez** (una línea
resume cómo va cada uno y la velocidad total; el detalle, en `obra/DATAn.log`).
```bash
./yolo_herramientas/descargar.sh obra                      # se puede cortar: sigue por donde iba
OBRA_PARTES="1" ./yolo_herramientas/descargar.sh obra      # solo el primer zip (18,9 GB)
OBRA_PARALELO=1 ./yolo_herramientas/descargar.sh obra      # de uno en uno, si se quiere
python yolo_herramientas/convertir_obra.py --inspeccionar  # cómo vienen y un mosaico con los nombres
python yolo_herramientas/convertir_obra.py                 # → yolo_crudo/obra (luego SAM y nuestras clases)
```
La web da sus 12 clases en orden alfabético pero no dice qué id lleva cada una: `clases.yaml → obra.origen`
asume ese orden. **Antes de convertir, mira `$HERRAMIENTAS_DATOS/obra/revision_obra.jpg`**: el nombre de
cada caja tiene que ser la herramienta que encierra. Si los ids empiezan en 1, `--inspeccionar` lo avisa
(entonces `--base1`, o `OBRA_BASE1=1` con `publicos.sh`). `bucket` (cubo) no se usa; `knife` y `cutter`
son el cúter; `spanner` y `wrench`, llaves; `tacker`, la grapadora de obra.

Los 110 GB son fotos a resolución completa (unos 4 MB cada una) que se guardan a 1280 px: una vez convertido
(`yolo_crudo/obra`), los `DATA*.zip` se pueden borrar. Dejad los `DATA*.zip.ok`, que son los que dicen a
`descargar.sh` que ese zip ya se bajó.

---
## 3. Fotos propias (más adelante)

Lo público enseña qué es un martillo; lo que más mejora el resultado **en la sala y con nuestra cámara** son
fotos tomadas con la propia OAK-D de la cabeza. Cuando las haya, entran como una fuente más.

```bash
./venv_nivel6/bin/python yolo_herramientas/capturar_oak.py --sesion mesa_luz_dia
```
ESPACIO = una foto · R = ráfaga (una cada 0,7 s mientras mueves robot o herramientas) · Q = salir.
En cada sesión cambia fondo, luz, distancia (0,3-1,3 m), ángulo, herramientas juntas y tapándose, la mano
del robot cerca… y haz también fotos sin herramientas. Para empezar, 50-100 fotos por herramienta.

**Etiquetar en CVAT** (app.cvat.ai o instalado en local):
1. Proyecto con las etiquetas **con los mismos nombres** que `clases.yaml` (`mango`, `martillo`, `alicates`…).
2. Tarea con las fotos de una sesión. Con «AI Tools → Segment Anything» basta con un clic por objeto.
3. Por cada herramienta, **dos polígonos**: la herramienta entera (incluido el mango) y el mango aparte con
   la etiqueta `mango`.
4. Exportar: «Ultralytics YOLO Segmentation 1.0» (con imágenes), descomprimir y convertir:
   ```bash
   python yolo_herramientas/convertir_yolo_seg.py ~/Descargas/mesa_luz_dia --fuente propias_mesa_luz_dia --repartir
   ```

Guarda una sesión entera (otro día, otra luz) **sin entrenar con ella**: es la que dirá de verdad si el
modelo funciona en la sala (paso 6).

---
## 4. Unir las fuentes

```bash
python yolo_herramientas/unir_datasets.py --nombre herramientas_v1 --repetir propias_mesa_luz_dia=5
python yolo_herramientas/revisar.py $HERRAMIENTAS_DATOS/yolo/herramientas_v1
```
Por defecto entran todas las fuentes de `convertidos/` **que tengan mango** (las que no, se dejan fuera y se
avisa: sus mangos sin marcar le enseñarían a YOLO que un mango es «fondo»). `--repetir` hace que vuestras
fotos pesen más frente a las decenas de miles de HANDAL. Al final escribe una tabla con cuántos ejemplos hay
de cada clase.

---
## 5. Entrenar

```bash
./yolo_herramientas/entrenar.sh herramientas_v1
```
YOLO26s-seg, 150 épocas, imágenes de 960 px (los mangos son finos y a 1 m ocupan pocos píxeles).
Variables: `MODELO=yolo26m-seg.pt` (más preciso y más lento), `EPOCAS`, `IMGSZ`, `BATCH=8` (si se queda sin
memoria la GPU). Se corta con Ctrl+C y se sigue con `yolo segment train resume model=runs/segment/herramientas_v1/weights/last.pt`.

Se entrena con `overlap_mask=False`, y es imprescindible: el mango está DENTRO de la herramienta y, con la
opción de serie, YOLO junta todas las máscaras en una y la herramienta se quedaría con un agujero en el mango.

Resultado: `runs/segment/herramientas_v1/weights/best.pt`, y en la misma carpeta `results.png` (curvas),
la matriz de confusión y ejemplos de validación pintados.

---
## 6. Evaluar

```bash
yolo segment val model=runs/segment/herramientas_v1/weights/best.pt \
     data=$HERRAMIENTAS_DATOS/yolo/herramientas_v1/herramientas_v1.yaml imgsz=960
```
Mira sobre todo la fila **mango** y la columna de máscaras (`Mask mAP50-95`). Para medir en la sala, con la
sesión que guardaste aparte:
```bash
python yolo_herramientas/convertir_yolo_seg.py ~/Descargas/sesion_test --fuente test_sala --repartir --frac-val 1
python yolo_herramientas/unir_datasets.py --nombre test_sala --fuentes test_sala
yolo segment val model=runs/segment/herramientas_v1/weights/best.pt data=$HERRAMIENTAS_DATOS/yolo/test_sala/test_sala.yaml imgsz=960
```
Y en vivo, con la cámara de cabeza (ESPACIO guarda las imágenes en las que falle):
```bash
python yolo_herramientas/probar_oak.py --modelo runs/segment/herramientas_v1/weights/best.pt
```

---
## 7. Mejorar (cada vuelta, mejor)

- **Más fotos propias, más rápido:** el modelo las preetiqueta y en CVAT solo se corrige
  («Upload annotations» → «Ultralytics YOLO Segmentation 1.0» → el zip):
  ```bash
  python yolo_herramientas/preetiquetar.py --modelo runs/segment/herramientas_v1/weights/best.pt \
         --imagenes $HERRAMIENTAS_DATOS/propias/nueva_sesion/images
  ```
- **Aprovechar las fuentes sin mango:** el modelo les pone el mango dentro de cada herramienta etiquetada
  (crea `<fuente>_mango`, que ya entra al unir). Revísalo antes.
  ```bash
  python yolo_herramientas/preetiquetar.py --modelo runs/segment/herramientas_v1/weights/best.pt \
         --completar-mango rf_tools_segmentation_0jsgu
  ```
- Volver a unir con otro nombre y entrenar partiendo del modelo anterior:
  ```bash
  python yolo_herramientas/unir_datasets.py --nombre herramientas_v2 --repetir propias_mesa_luz_dia=5
  MODELO=runs/segment/herramientas_v1/weights/best.pt ./yolo_herramientas/entrenar.sh herramientas_v2
  ```

---
## 8. Para la integración (fases 5-8)

En la GPU del PC, exportado a TensorRT va bastante más rápido que en PyTorch:
```bash
yolo export model=runs/segment/herramientas_v2/weights/best.pt format=engine half=True imgsz=960
```
(en la propia OAK-D Pro no compensa: su procesador va justo con segmentación a esta resolución).

---
## Archivos

| Archivo | Qué hace |
|---|---|
| `clases.yaml` | clases y cómo se traduce cada dataset a ellas |
| `publicos.sh` | el camino solo con datasets públicos: descargar → convertir → fase 1 → completar → fase 2 |
| `instalar.sh` | crea `venv_yolo` |
| `descargar.sh` | PACO, HANDAL, Dataset Ninja, Open Images, Roboflow, el de obra de Zenodo, COCO entero (opcional) |
| `descargar_open_images.py` · `descargar_roboflow.py` | los usa `descargar.sh` |
| `descargar_handal.py` | HANDAL desde Google Drive: solo las categorías de herramientas, sin `depth/` |
| `descargar_ninja.py` | Baja y descomprime los de Dataset Ninja; si la descarga sale vacía, explica cómo bajarlo a mano |
| `convertir_supervisely.py` | Dataset Ninja (METU-ALET…) → YOLO con sus nombres |
| `convertir_obra.py` | Small Size Construction Tools (sus zip, sin descomprimir) → YOLO con sus nombres |
| `convertir_paco.py` · `convertir_handal.py` · `convertir_yolo_seg.py` | cada dataset → `convertidos/<fuente>/` |
| `cajas_a_mascaras.py` | dataset de cajas → máscaras con SAM 2 |
| `capturar_oak.py` | fotos con la cámara de cabeza (con `venv_nivel6`) |
| `unir_datasets.py` | fuentes → dataset de entrenamiento + `data.yaml` |
| `revisar.py` | mosaicos con las máscaras pintadas para comprobar |
| `entrenar.sh` | entrenamiento en la RTX 5080 |
| `preetiquetar.py` | el modelo etiqueta fotos nuevas (para CVAT) o completa el mango |
| `probar_oak.py` | el modelo en vivo con la cámara de cabeza |
| `comun.py` | utilidades compartidas |
