package main

// Render del gemelo para la web: lo mismo que skill_shield/autonomo/render_gemelo.py, con la
// librería C de MuJoCo en vez de mujoco.Renderer de Python.
//
//   · carga RUTA_XML (fairino_description/mjcf/fairino_mjcf.xml), sin reflejo en los materiales
//     y con el búfer fuera de pantalla de al menos 1352×1014;
//   · el entorno del taller [3] y el del taladro [4] escondidos (EntornoTaller / EntornoTaladro
//     al arrancar) y el brazo en POSICION_STANDBY_BRAZO (sin FR5 no hay pose real: así lo deja
//     modo.py);
//   · «sim»: vista libre 960×540 que se gira desde la web; «simcam»: 640×480 desde la cámara de
//     cabeza calibrada (camara_robot.json), como la que se superpone a la imagen real.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// ── valores de config.py ──
const (
	rutaXML        = "fairino_description/mjcf/fairino_mjcf.xml" // RUTA_XML
	simAncho       = 960                                         // AUTONOMO_SIM_ANCHO
	simAlto        = 540                                         // AUTONOMO_SIM_ALTO
	simFPS         = 20                                          // AUTONOMO_SIM_FPS
	jpegCalidad    = 80                                          // AUTONOMO_JPEG_CALIDAD
	camAncho       = 640                                         // ANCHO_CAM
	camAlto        = 480                                         // ALTO_CAM
	archivoCalib   = "camara_robot.json"                         // CALIB_ROBOT_ARCHIVO
	mjOBJ_BODY     = 1
	mjOBJ_JOINT    = 3
	mjSTATE_QPOS   = 1 << 1
	mjSTATE_MOCAP  = 1 << 9
	mjFB_OFFSCREEN = 1
	mjCAT_ALL      = 7
	mjFONTSCALE150 = 150
	mjCAMERA_FREE  = 0
)

var (
	articulacionesBrazo = []string{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"} // ARM_JOINTS
	posicionReposo      = []float64{0.0, -1.5708, 2.0, 0.0, 1.5708, 0.0}                       // POSICION_STANDBY_BRAZO
	objOculto           = [3]float64{0, 0, -10}                                                // OBJ_OCULTO
	tallerObjetos       = []string{"taller_alicates", "taller_destornillador", "taller_bombilla", "taller_fusible"}
)

type Fotograma struct {
	jpg []byte
	t   time.Time
}

type Gemelo struct {
	mu        sync.Mutex
	sim       Fotograma
	simcam    Fotograma
	vista     func() Vista
	estado    string // texto para la consola si algo falla
	listo     bool
	nuevo     chan struct{}
	simSeq    uint64
	simcamSeq uint64
}

// Arranca el hilo de render (bloqueado a un hilo del sistema: el contexto de OpenGL es suyo).
func arrancarGemelo(dirExe, dirModelo string, vista func() Vista) *Gemelo {
	g := &Gemelo{vista: vista, nuevo: make(chan struct{})}
	listo := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		err := g.bucle(dirExe, dirModelo, listo)
		if err != nil {
			select {
			case listo <- err:
			default:
			}
		}
	}()
	select {
	case err := <-listo:
		if err != nil {
			g.estado = err.Error()
		} else {
			g.listo = true
		}
	case <-time.After(60 * time.Second):
		g.estado = "el gemelo tarda demasiado en arrancar"
	}
	return g
}

func (g *Gemelo) publicar(cual string, jpg []byte) {
	g.mu.Lock()
	f := Fotograma{jpg, time.Now()}
	if cual == "sim" {
		g.sim = f
		g.simSeq++
	} else {
		g.simcam = f
		g.simcamSeq++
	}
	viejo := g.nuevo
	g.nuevo = make(chan struct{})
	g.mu.Unlock()
	close(viejo)
}

// Último fotograma, su número y un canal que se cierra cuando haya otro.
func (g *Gemelo) leer(cual string) ([]byte, uint64, <-chan struct{}) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if cual == "sim" {
		return g.sim.jpg, g.simSeq, g.nuevo
	}
	return g.simcam.jpg, g.simcamSeq, g.nuevo
}

func (g *Gemelo) edad(cual string) any {
	g.mu.Lock()
	defer g.mu.Unlock()
	f := g.sim
	if cual == "simcam" {
		f = g.simcam
	}
	if f.jpg == nil {
		return nil
	}
	return math.Round(time.Since(f.t).Seconds()*100) / 100
}

func (g *Gemelo) bucle(dirExe, dirModelo string, listo chan<- error) error {
	mj, err := cargarMuJoCo(filepath.Join(dirExe, nombreLibreria))
	if err != nil {
		msg := fmt.Sprintf("no se pudo cargar %s: %v", nombreLibreria, err)
		if runtime.GOOS == "windows" {
			msg += "\n     Si falta el runtime de Visual C++, instala «Microsoft Visual C++ Redistributable (x64)»:" +
				"\n     https://aka.ms/vs/17/release/vc_redist.x64.exe"
		}
		return fmt.Errorf("%s", msg)
	}
	info, err := crearContextoGL()
	if err != nil {
		return fmt.Errorf("no se pudo crear el contexto de OpenGL: %v", err)
	}
	fmt.Println("  OpenGL:      ", info)
	// MuJoCo necesita OpenGL ≥ 1.5 con framebuffers; sin driver de la gráfica Windows da el 1.1
	// de «GDI Generic» y mjr_makeContext cerraría el programa: mejor seguir sin gemelo.
	if i := strings.Index(info, "OpenGL "); i >= 0 && strings.HasPrefix(info[i+7:], "1.") {
		return fmt.Errorf("este PC solo tiene OpenGL %s (¿falta el driver de la tarjeta gráfica?)", info[i+7:])
	}

	// ── modelo (rutas relativas a la carpeta «modelo», así no importan tildes en la ruta) ──
	if err := os.Chdir(dirModelo); err != nil {
		return fmt.Errorf("no encuentro la carpeta del modelo (%s)", dirModelo)
	}
	errBuf := mj.reservar(1000)
	m := mj.c("mj_loadXML", mj.cadena(rutaXML), 0, errBuf, 1000)
	if m == 0 {
		return fmt.Errorf("MuJoCo no pudo cargar %s: %s", rutaXML, cadenaC(errBuf, 1000))
	}
	nmat := int(leerI32(m, mNmat))
	for i := 0; i < nmat; i++ { // sin reflejo del suelo (como el visor)
		*(*float32)(ptr(leerPtr(m, mMatReflectance) + uintptr(4*i))) = 0
	}
	escribirI32(m, mOffwidth, max32(leerI32(m, mOffwidth), simAncho, 1352))
	escribirI32(m, mOffheight, max32(leerI32(m, mOffheight), simAlto, 1014))
	d := mj.c("mj_makeData", m)

	// ── estado inicial: taller y taladro escondidos, brazo en reposo ──
	nq := int(leerI32(m, mNq))
	nmocap := int(leerI32(m, mNmocap))
	qpos := mj.leerEstado(m, d, mjSTATE_QPOS, nq)
	mocap := mj.leerEstado(m, d, mjSTATE_MOCAP, 3*nmocap)
	id := func(tipo int, nombre string) int {
		return int(int32(mj.c("mj_name2id", m, uintptr(tipo), mj.cadena(nombre))))
	}
	ponerMocap := func(cuerpo string, p [3]float64) {
		if b := id(mjOBJ_BODY, cuerpo); b >= 0 {
			if k := int(mj.intModelo(m, mBodyMocapid, b)); k >= 0 {
				copy(mocap[3*k:3*k+3], p[:])
			}
		}
	}
	ponerLibre := func(qa int, p [3]float64) {
		copy(qpos[qa:qa+3], p[:])
		copy(qpos[qa+3:qa+7], []float64{1, 0, 0, 0})
	}
	// EntornoTaller.ocultar()
	ponerMocap("taller_mesa", objOculto)
	for k, nombre := range tallerObjetos {
		if b := id(mjOBJ_BODY, nombre); b >= 0 && mj.intModelo(m, mBodyJntnum, b) == 1 {
			j := int(mj.intModelo(m, mBodyJntadr, b))
			if mj.intModelo(m, mJntType, j) == 0 { // mjJNT_FREE
				ponerLibre(int(mj.intModelo(m, mJntQposadr, j)), [3]float64{objOculto[0] + 0.5*float64(k+1), objOculto[1], objOculto[2]})
			}
		}
	}
	// EntornoTaladro.ocultar()
	for k := 0; k < 32; k++ {
		ponerMocap(fmt.Sprintf("taladro_agujero_%d", k), [3]float64{objOculto[0] + 3.0 + 0.1*float64(k), objOculto[1], objOculto[2]})
	}
	ponerMocap("taladro_techo", objOculto)
	ponerMocap("taladro_soporte", [3]float64{objOculto[0] + 1.0, objOculto[1], objOculto[2]})
	if j := id(mjOBJ_JOINT, "taladro_libre"); j >= 0 {
		ponerLibre(int(mj.intModelo(m, mJntQposadr, j)), [3]float64{objOculto[0] + 2.0, objOculto[1], objOculto[2]})
	}
	// sincronizar_mujoco(reposo)
	for k, nombre := range articulacionesBrazo {
		if j := id(mjOBJ_JOINT, nombre); j >= 0 {
			qpos[mj.intModelo(m, mJntQposadr, j)] = posicionReposo[k]
		}
	}
	mj.escribirEstado(m, d, mjSTATE_QPOS, qpos)
	mj.escribirEstado(m, d, mjSTATE_MOCAP, mocap)
	mj.c("mj_forward", m, d)

	// ── escena y contexto (lo que hace mujoco.Renderer) ──
	opt := mj.reservar(tamOpcion)
	mj.c("mjv_defaultOption", opt)
	cam := mj.reservar(tamCamara)
	mj.c("mjv_defaultCamera", cam)
	escribirI32(cam, cType, mjCAMERA_FREE)
	scn := mj.reservar(tamEscena)
	mj.c("mjv_defaultScene", scn)
	mj.c("mjv_makeScene", m, scn, 10000)
	con := mj.reservar(tamContexto)
	mj.c("mjr_defaultContext", con)
	mj.c("mjr_makeContext", m, con, mjFONTSCALE150)
	mj.c("mjr_setBuffer", mjFB_OFFSCREEN, con)
	rgb := mj.reservar(3 * 1352 * 1014)

	calib := cargarCalibracion(filepath.Join(dirModelo, archivoCalib))
	fmt.Printf("  Gemelo:       MuJoCo %d · render %dx%d a %d fps", mjVersionCabecera, simAncho, simAlto, simFPS)
	if calib != nil {
		fmt.Print(" · vista desde la cámara calibrada")
	}
	fmt.Println()
	listo <- nil

	render := func(w, h int) []byte {
		args := append(mj.argsRect(0, 0, w, h), scn, con)
		mj.c("mjr_render", args...)
		args = append([]uintptr{rgb, 0}, mj.argsRect(0, 0, w, h)...)
		mj.c("mjr_readPixels", append(args, con)...)
		return codificarJPEG(rgb, w, h)
	}
	periodo := time.Second / simFPS
	var tCam time.Time
	for {
		t0 := time.Now()
		v := g.vista()
		escribirF64(cam, cAzimuth, v.Azimut)
		escribirF64(cam, cElevation, v.Elevacion)
		escribirF64(cam, cDistance, v.Distancia)
		for i := 0; i < 3; i++ {
			escribirF64(cam, cLookat+uintptr(8*i), v.Mirar[i])
		}
		mj.c("mjv_updateScene", m, d, opt, 0, cam, mjCAT_ALL, scn)
		g.publicar("sim", render(simAncho, simAlto))

		if calib != nil && t0.Sub(tCam) >= 100*time.Millisecond { // a 10 fps, como el original
			tCam = t0
			mj.c("mjv_updateScene", m, d, opt, 0, cam, mjCAT_ALL, scn)
			cerca := float64(leerF32(scn, sCamera+gFrustumNear))
			cg := calib.camaraGL(camAncho, camAlto, cerca)
			for k := uintptr(0); k < 2; k++ {
				c := scn + sCamera + k*tamGLCamara
				for i := uintptr(0); i < 3; i++ {
					escribirF32(c, gPos+4*i, f32(cg.pos[i]))
					escribirF32(c, gForward+4*i, f32(cg.forward[i]))
					escribirF32(c, gUp+4*i, f32(cg.up[i]))
				}
				escribirF32(c, gFrustumCenter, f32(cg.centro))
				escribirF32(c, gFrustumTop, f32(cg.arriba))
				escribirF32(c, gFrustumBottom, f32(cg.abajo))
			}
			g.publicar("simcam", render(camAncho, camAlto))
		}
		if dt := periodo - time.Since(t0); dt > 0 {
			time.Sleep(dt)
		}
	}
}

func max32(vs ...int32) int32 {
	m := vs[0]
	for _, v := range vs[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func (mj *MJ) leerEstado(m, d uintptr, spec uint32, n int) []float64 {
	if tam := int(int32(mj.c("mj_stateSize", m, uintptr(spec)))); tam != n {
		panic(fmt.Sprintf("mj_stateSize(%d) = %d, esperaba %d", spec, tam, n))
	}
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	b := mj.reservar(8 * n)
	defer mj.liberar(b)
	mj.c("mj_getState", m, d, b, uintptr(spec))
	for i := range out {
		out[i] = *(*float64)(ptr(b + uintptr(8*i)))
	}
	return out
}

func (mj *MJ) escribirEstado(m, d uintptr, spec uint32, v []float64) {
	if len(v) == 0 {
		return
	}
	b := mj.reservar(8 * len(v))
	defer mj.liberar(b)
	for i, x := range v {
		*(*float64)(ptr(b + uintptr(8*i))) = x
	}
	mj.c("mj_setState", m, d, b, uintptr(spec))
}

// RGB de OpenGL (de abajo arriba) → JPEG
func codificarJPEG(rgb uintptr, w, h int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		fila := (*[1 << 30]byte)(ptr(rgb + uintptr(3*w*(h-1-y))))[: 3*w : 3*w]
		o := img.PixOffset(0, y)
		for x := 0; x < w; x++ {
			img.Pix[o], img.Pix[o+1], img.Pix[o+2], img.Pix[o+3] = fila[3*x], fila[3*x+1], fila[3*x+2], 255
			o += 4
		}
	}
	var b bytes.Buffer
	jpeg.Encode(&b, img, &jpeg.Options{Quality: jpegCalidad})
	return b.Bytes()
}

// ── calibración cámara→robot (calibracion.cargar + camara_gl, sin corrección guardada) ──

type Calibracion struct {
	T           [4][4]float64
	K           [3][3]float64
	ancho, alto float64
}

type camGL struct {
	pos, forward, up      [3]float64
	centro, arriba, abajo float64
}

func cargarCalibracion(ruta string) *Calibracion {
	b, err := os.ReadFile(ruta)
	if err != nil {
		return nil
	}
	var d struct {
		T   [4][4]float64 `json:"T_mundo_cam"`
		Int struct {
			K     [3][3]float64 `json:"K"`
			Ancho float64       `json:"ancho"`
			Alto  float64       `json:"alto"`
		} `json:"intrinsecos"`
	}
	if json.Unmarshal(b, &d) != nil || d.Int.Ancho == 0 {
		return nil
	}
	return &Calibracion{d.T, d.Int.K, d.Int.Ancho, d.Int.Alto}
}

func (c *Calibracion) camaraGL(w, h int, cerca float64) camGL {
	fx := c.K[0][0] * float64(w) / c.ancho
	cx := c.K[0][2] * float64(w) / c.ancho
	fy := c.K[1][1] * float64(h) / c.alto
	cy := c.K[1][2] * float64(h) / c.alto
	var g camGL
	for i := 0; i < 3; i++ {
		g.pos[i] = c.T[i][3]
		g.forward[i] = c.T[i][2]
		g.up[i] = -c.T[i][1]
	}
	g.arriba = cerca * cy / fy
	g.abajo = -cerca * (float64(h) - cy) / fy
	g.centro = cerca * (float64(w)/2.0 - cx) / fx
	return g
}
