// Skill Shield · web del funcionamiento autónomo en Windows, SIN robots ni cámara.
//
// Sirve la página del proyecto (web/index.html, la de skill_shield/autonomo/estatico) con las
// mismas rutas que skill_shield/autonomo/web.py, y el estado es el que publica modo.py cuando
// no hay nada conectado: FR5 sin conexión, OrcaHand sin USB, táctiles sin sensores y cámara a
// la espera. El gemelo digital lo dibuja MuJoCo (mujoco.dll) con el modelo del proyecto, como
// render_gemelo.py.
package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	cookieSesion = "ss_sesion"
	latidoPedal  = 350 * time.Millisecond // AUTONOMO_PEDAL_LATIDO_S
	sesionS      = 12 * 3600
)

type Vista struct {
	Azimut    float64    `json:"azimut"`
	Elevacion float64    `json:"elevacion"`
	Distancia float64    `json:"distancia"`
	Mirar     [3]float64 `json:"mirar"`
}

var vistaInicial = Vista{90, -20, 1.1, [3]float64{0.15, 0.0, 0.25}} // VISOR_VISTA_INICIAL

// ══════════════════════════════════════════════════════════════════════════════
// ESTADO (web.py ServicioWeb + lo que publica modo.py sin hardware)
// ══════════════════════════════════════════════════════════════════════════════

type Servicio struct {
	mu          sync.Mutex
	sesiones    map[string]time.Time
	pulsando    map[string]time.Time
	obligatorio bool
	verObst     bool
	objetivo    *int
	vista       Vista
	tEstado     string
	tTexto      string
	calib       map[string]any
	gemelo      *Gemelo
}

func nuevoServicio(calib map[string]any) *Servicio {
	return &Servicio{sesiones: map[string]time.Time{}, pulsando: map[string]time.Time{}, obligatorio: true,
		verObst: true, vista: vistaInicial, tEstado: "libre", tTexto: "Elige la herramienta y pulsa «Planificar».",
		calib: calib}
}

func (s *Servicio) leerVista() Vista {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vista
}

func (s *Servicio) pedalPisado() bool {
	for _, t := range s.pulsando {
		if time.Since(t) < latidoPedal {
			return true
		}
	}
	return false
}

func (s *Servicio) autorizado(tok string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.sesiones[tok]
	return ok && time.Now().Before(c)
}

func (s *Servicio) estado(tok string) map[string]any {
	aut := s.autorizado(tok)
	s.mu.Lock()
	defer s.mu.Unlock()
	pisado := s.pedalPisado()
	var objetivo any
	if s.objetivo != nil {
		objetivo = *s.objetivo
	}
	puedePlan := s.tEstado == "libre" || s.tEstado == "lista" || s.tEstado == "terminada" || s.tEstado == "error" || s.tEstado == "pausada"
	var edadSim, edadSimcam any
	if s.gemelo != nil {
		edadSim, edadSimcam = s.gemelo.edad("sim"), s.gemelo.edad("simcam")
	}
	return map[string]any{
		// ── de modo.py (se publica cada 0,1 s) ──
		"modo": "AUTÓNOMO · FR5 en simulacro",
		"fr5": map[string]any{"texto": "sin conexión (reintentando solo)", "en_linea": false, "en_error": false,
			"colision": false, "codigo": []int{0, 0}, "simulacro": true, "q_real_deg": nil},
		"mano": map[string]any{"estado": "sin_usb", "texto": "sin USB (/dev/ttyUSB0 no existe)", "motores_ok": 0,
			"motores_total": 17, "puerto": "/dev/ttyUSB0"},
		"tactil":      map[string]any{"estado": "error", "texto": "sin sensores (¿USB de los táctiles?)"},
		"gemelo":      "sin pose real",
		"calibracion": s.calib,
		"tarea": map[string]any{"estado": s.tEstado, "texto": s.tTexto, "simulacro": true, "fase": nil,
			"i": nil, "n": nil, "progreso": 0.0, "vel_deg_s": 15.0,
			"puede": map[string]any{"planificar": puedePlan, "ejecutar": false, "cancelar": false, "volver": false}},
		"edad_s": 0.05,
		// ── lo que añade web.py ──
		"camara": map[string]any{},
		"mapa":   map[string]any{"ver": s.verObst},
		"herramientas": map[string]any{"lista": []any{}, "calibrada": false, "objetivo": objetivo,
			"activo": false},
		"control": map[string]any{"pedal_obligatorio_robot": s.obligatorio, "pedal_pisado": pisado,
			"movimiento_permitido": false, "motivo": "FR5 sin conexión o con error", "web_viva": true,
			"paso_mueve": false, "pedal_obligatorio": s.obligatorio, "pedal_pisado_web": pisado,
			"permitir_sin_pedal": true},
		"sesion":     map[string]any{"autorizado": aut, "requiere_clave": true},
		"gemelo_web": map[string]any{"edad_s": edadSim, "cam_edad_s": edadSimcam},
	}
}

// estado_calibracion() de modo.py
func estadoCalibracion(ruta string) map[string]any {
	b, err := os.ReadFile(ruta)
	if err != nil {
		return map[string]any{"estado": "sin", "texto": "sin calibrar"}
	}
	var d map[string]any
	if err := json.Unmarshal(b, &d); err != nil {
		return map[string]any{"estado": "error", "texto": "archivo ilegible: " + err.Error()}
	}
	n := 0
	if u, ok := d["muestras_usadas"].([]any); ok {
		n = len(u)
	}
	e, _ := d["error_medio_mm"].(float64)
	ok := d["error_medio_mm"] != nil && e <= 5.0 // CALIB_ROBOT_ERROR_MAX_MM
	out := map[string]any{"fecha": d["fecha"], "error_medio_mm": d["error_medio_mm"], "n": n}
	if ok {
		out["estado"], out["texto"] = "ok", fmt.Sprintf("%v mm", d["error_medio_mm"])
	} else {
		out["estado"], out["texto"] = "aviso", fmt.Sprintf("%v mm (revisar)", d["error_medio_mm"])
	}
	return out
}

// ══════════════════════════════════════════════════════════════════════════════
// HTTP (las rutas de web.py)
// ══════════════════════════════════════════════════════════════════════════════

func responder(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func cuerpo(r *http.Request) map[string]any {
	m := map[string]any{}
	json.NewDecoder(r.Body).Decode(&m)
	return m
}

func token(r *http.Request) string {
	if c, err := r.Cookie(cookieSesion); err == nil {
		return c.Value
	}
	return ""
}

func num(v any, def float64) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return def
}

func limitar(v, a, b float64) float64 { return math.Max(a, math.Min(b, v)) }

func (s *Servicio) rutas(mux *http.ServeMux, web fs.FS) {
	noAut := func(w http.ResponseWriter) {
		responder(w, 401, map[string]any{"ok": false, "msg": "Desbloquea los mandos con la contraseña"})
	}
	archivo := func(nombre, tipo string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			b, err := fs.ReadFile(web, nombre)
			if err != nil {
				http.Error(w, "No encuentro web/"+nombre, 404)
				return
			}
			w.Header().Set("Content-Type", tipo)
			w.Header().Set("Cache-Control", "no-store")
			w.Write(b)
		}
	}
	mux.HandleFunc("GET /{$}", archivo("index.html", "text/html; charset=utf-8"))
	mux.HandleFunc("GET /logo.png", archivo("logo_idi.png", "image/png"))

	mux.HandleFunc("GET /api/estado", func(w http.ResponseWriter, r *http.Request) {
		responder(w, 200, s.estado(token(r)))
	})

	// cámara: no hay proceso de cámara → sin imagen (la página enseña «Esperando a la cámara…»)
	mux.HandleFunc("GET /foto.jpg", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(503)
		w.Write([]byte("Sin imagen"))
	})
	mux.HandleFunc("GET /video.mjpg", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(200)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done() // como web.py: la conexión queda abierta esperando fotogramas
	})
	mux.HandleFunc("GET /sim.mjpg", func(w http.ResponseWriter, r *http.Request) { s.mjpeg(w, r, "sim") })
	mux.HandleFunc("GET /simcam.mjpg", func(w http.ResponseWriter, r *http.Request) { s.mjpeg(w, r, "simcam") })

	mux.HandleFunc("POST /api/vista", func(w http.ResponseWriter, r *http.Request) {
		m := cuerpo(r)
		s.mu.Lock()
		v := &s.vista
		switch m["accion"] {
		case "rotar":
			v.Azimut = math.Mod(math.Mod(v.Azimut-0.35*num(m["dx"], 0), 360)+360, 360)
			v.Elevacion = limitar(v.Elevacion-0.35*num(m["dy"], 0), -89, -2)
		case "zoom":
			v.Distancia = limitar(v.Distancia*limitar(num(m["factor"], 1), 0.5, 2), 0.25, 4)
		case "mover":
			az := v.Azimut * math.Pi / 180
			k := 0.0015 * v.Distancia
			dx, dy := num(m["dx"], 0), num(m["dy"], 0)
			v.Mirar[0] += k * dx * math.Sin(az)
			v.Mirar[1] += k * -dx * math.Cos(az)
			v.Mirar[2] += k * dy
		case "preset":
			p := map[string][2]float64{"frente": {90, -8}, "lado": {0, -8}, "arriba": {90, -88}}
			if n, _ := m["nombre"].(string); n == "inicial" {
				*v = vistaInicial
			} else if ae, ok := p[n]; ok {
				*v = vistaInicial
				v.Azimut, v.Elevacion = ae[0], ae[1]
			}
		}
		out := *v
		s.mu.Unlock()
		responder(w, 200, map[string]any{"ok": true, "vista": out})
	})

	mux.HandleFunc("POST /api/login", func(w http.ResponseWriter, r *http.Request) {
		cuerpo(r) // sin robots no hay nada que proteger: vale cualquier contraseña
		b := make([]byte, 24)
		rand.Read(b)
		tok := hex.EncodeToString(b)
		s.mu.Lock()
		s.sesiones[tok] = time.Now().Add(sesionS * time.Second)
		s.mu.Unlock()
		http.SetCookie(w, &http.Cookie{Name: cookieSesion, Value: tok, Path: "/", HttpOnly: true,
			MaxAge: sesionS, SameSite: http.SameSiteStrictMode})
		responder(w, 200, map[string]any{"ok": true})
	})
	mux.HandleFunc("POST /api/logout", func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		delete(s.sesiones, token(r))
		delete(s.pulsando, token(r))
		s.mu.Unlock()
		http.SetCookie(w, &http.Cookie{Name: cookieSesion, Value: "", Path: "/", MaxAge: -1})
		responder(w, 200, map[string]any{"ok": true})
	})

	mux.HandleFunc("POST /api/pedal", func(w http.ResponseWriter, r *http.Request) {
		tok := token(r)
		pulsado, _ := cuerpo(r)["pulsado"].(bool)
		if pulsado && !s.autorizado(tok) {
			noAut(w)
			return
		}
		s.mu.Lock()
		if pulsado {
			s.pulsando[tok] = time.Now()
		} else {
			delete(s.pulsando, tok)
		}
		p := s.pedalPisado()
		s.mu.Unlock()
		responder(w, 200, map[string]any{"ok": true, "pisado": p})
	})
	mux.HandleFunc("POST /api/pedal/modo", func(w http.ResponseWriter, r *http.Request) {
		if !s.autorizado(token(r)) {
			noAut(w)
			return
		}
		ob, ok := cuerpo(r)["obligatorio"].(bool)
		if !ok {
			ob = true
		}
		s.mu.Lock()
		s.obligatorio = ob
		s.mu.Unlock()
		msg := "Pedal obligatorio"
		if !ob {
			msg = "Pedal desactivado: el robot puede moverse sin pedal"
		}
		responder(w, 200, map[string]any{"ok": true, "msg": msg})
	})
	mux.HandleFunc("POST /api/objetivo", func(w http.ResponseWriter, r *http.Request) {
		m := cuerpo(r)
		s.mu.Lock()
		if f, ok := m["id"].(float64); ok {
			i := int(f)
			s.objetivo = &i
		} else {
			s.objetivo = nil
		}
		var o any
		if s.objetivo != nil {
			o = *s.objetivo
		}
		s.mu.Unlock()
		responder(w, 200, map[string]any{"ok": true, "objetivo": o})
	})
	mux.HandleFunc("POST /api/mapa", func(w http.ResponseWriter, r *http.Request) {
		acc, _ := cuerpo(r)["accion"].(string)
		if acc == "borrar" && !s.autorizado(token(r)) {
			noAut(w)
			return
		}
		msgs := map[string]string{"ver": "Obstáculos visibles", "ocultar": "Obstáculos ocultos",
			"borrar": "Mapa borrado: se vuelve a llenar en un momento"}
		msg, ok := msgs[acc]
		if !ok {
			responder(w, 400, map[string]any{"ok": false, "msg": "Acción desconocida"})
			return
		}
		s.mu.Lock()
		if acc != "borrar" {
			s.verObst = acc == "ver"
		}
		s.mu.Unlock()
		responder(w, 200, map[string]any{"ok": true, "msg": msg})
	})
	mux.HandleFunc("POST /api/tarea", func(w http.ResponseWriter, r *http.Request) {
		if !s.autorizado(token(r)) {
			noAut(w)
			return
		}
		acc, _ := cuerpo(r)["accion"].(string)
		msgs := map[string]string{"planificar": "Planificando…", "ejecutar": "Pisa el pedal para que se mueva",
			"cancelar": "Tarea cancelada", "volver": "Planificando la vuelta al inicio…"}
		msg, ok := msgs[acc]
		if !ok {
			responder(w, 400, map[string]any{"ok": false, "msg": "Acción desconocida"})
			return
		}
		// lo que hace Tarea.orden() sin pose real del FR5
		s.mu.Lock()
		puede := s.tEstado == "libre" || s.tEstado == "lista" || s.tEstado == "terminada" || s.tEstado == "error" || s.tEstado == "pausada"
		switch acc {
		case "planificar", "volver":
			if puede {
				s.tEstado, s.tTexto = "error", "No hay pose real del FR5: hace falta para planificar."
			}
		case "cancelar":
			s.tEstado, s.tTexto = "libre", "Cancelada: el robot se queda donde está."
		}
		s.mu.Unlock()
		responder(w, 200, map[string]any{"ok": true, "msg": msg})
	})
	mux.HandleFunc("POST /api/reset", func(w http.ResponseWriter, r *http.Request) {
		if !s.autorizado(token(r)) {
			noAut(w)
			return
		}
		fmt.Println("⚠️  Web: reinicio pedido, pero no hay conexión con el FR5.")
		responder(w, 200, map[string]any{"ok": true, "msg": "Reinicio pedido: borrar error → deshabilitar → habilitar"})
	})
}

func (s *Servicio) mjpeg(w http.ResponseWriter, r *http.Request, cual string) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	w.Header().Set("Cache-Control", "no-store")
	fl, _ := w.(http.Flusher)
	var leido uint64
	for {
		if s.gemelo == nil {
			<-r.Context().Done()
			return
		}
		jpg, seq, nuevo := s.gemelo.leer(cual)
		if jpg != nil && seq != leido {
			leido = seq
			if _, err := fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(jpg)); err != nil {
				return
			}
			if _, err := w.Write(jpg); err != nil {
				return
			}
			if _, err := w.Write([]byte("\r\n")); err != nil {
				return
			}
			if fl != nil {
				fl.Flush()
			}
		}
		select {
		case <-r.Context().Done():
			return
		case <-nuevo:
		case <-time.After(time.Second):
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// ARRANQUE
// ══════════════════════════════════════════════════════════════════════════════

func abrirNavegador(url string) {
	switch runtime.GOOS {
	case "windows":
		exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	default:
		exec.Command("xdg-open", url).Start()
	}
}

func esperarEnter() {
	if runtime.GOOS == "windows" && os.Getenv("SS_NO_ESPERAR") == "" {
		fmt.Println("\nPulsa Enter para cerrar.")
		fmt.Scanln()
	}
}

func main() {
	exe, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	dirExe := filepath.Dir(exe)
	if d := os.Getenv("SS_DIR"); d != "" { // para probar fuera de la carpeta del programa
		dirExe = d
	}
	dirWeb := filepath.Join(dirExe, "web")
	dirModelo := filepath.Join(dirExe, "modelo")
	if _, err := os.Stat(filepath.Join(dirWeb, "index.html")); err != nil {
		fmt.Println("✗ No encuentro web\\index.html junto al programa. ¿Has descomprimido la carpeta entera?")
		esperarEnter()
		os.Exit(1)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:8080") // AUTONOMO_WEB_PUERTO (solo este PC)
	if err != nil {
		ln, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			log.Fatal(err)
		}
	}
	url := fmt.Sprintf("http://localhost:%d/", ln.Addr().(*net.TCPAddr).Port)

	linea := strings.Repeat("═", 66)
	fmt.Println(linea)
	fmt.Println("  Skill Shield · Funcionamiento autónomo — web SIN robots ni cámara")
	fmt.Println(linea)

	s := nuevoServicio(estadoCalibracion(filepath.Join(dirModelo, archivoCalib)))
	fmt.Println("  Cargando el gemelo digital (MuJoCo)…")
	g := arrancarGemelo(dirExe, dirModelo, s.leerVista)
	if g.listo {
		s.gemelo = g
	} else {
		fmt.Println("  ⚠️  Sin gemelo digital:", g.estado)
		fmt.Println("     La página funciona igual; el panel del gemelo se quedará «Esperando al gemelo…».")
	}

	mux := http.NewServeMux()
	s.rutas(mux, os.DirFS(dirWeb))
	fmt.Println()
	fmt.Println("  Página:      ", url)
	fmt.Println("  Archivos:    ", dirWeb)
	fmt.Println("                (edita index.html y pulsa F5 en el navegador)")
	fmt.Println("  Contraseña:   cualquiera (desbloquea pedal, tarea…)")
	fmt.Println()
	fmt.Println("  Cierra esta ventana para pararlo.")
	fmt.Println(linea)

	if os.Getenv("SS_NO_NAVEGADOR") == "" {
		go func() { time.Sleep(300 * time.Millisecond); abrirNavegador(url) }()
	}
	log.Fatal(http.Serve(ln, mux))
}
