//go:build linux

package main

// Linux (solo para PROBAR el programa aquí con la misma versión de MuJoCo): libmujoco.so con
// dlopen y contexto de OpenGL con EGL sin pantalla. En Windows esto lo hace mujoco_windows.go.

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
typedef uintptr_t U;
static U llamar_c(void* f, int n, U* a) {
  switch (n) {
  case 0: return ((U(*)(void))f)();
  case 1: return ((U(*)(U))f)(a[0]);
  case 2: return ((U(*)(U,U))f)(a[0],a[1]);
  case 3: return ((U(*)(U,U,U))f)(a[0],a[1],a[2]);
  case 4: return ((U(*)(U,U,U,U))f)(a[0],a[1],a[2],a[3]);
  case 5: return ((U(*)(U,U,U,U,U))f)(a[0],a[1],a[2],a[3],a[4]);
  case 6: return ((U(*)(U,U,U,U,U,U))f)(a[0],a[1],a[2],a[3],a[4],a[5]);
  case 7: return ((U(*)(U,U,U,U,U,U,U))f)(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);
  case 8: return ((U(*)(U,U,U,U,U,U,U,U))f)(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
  }
  return 0;
}
static void* abrir_c(const char* p) { return dlopen(p, RTLD_NOW | RTLD_GLOBAL); }
static void* simbolo_c(void* h, const char* n) { return dlsym(h, n); }
static const char* error_c(void) { return dlerror(); }

// EGL sin pantalla (como MUJOCO_GL=egl en Python)
typedef void* (*fGetDisplay)(void*);
typedef void* (*fGetPlatformDisplay)(unsigned, void*, const intptr_t*);
typedef unsigned (*fInitialize)(void*, int*, int*);
typedef unsigned (*fChooseConfig)(void*, const int*, void**, int, int*);
typedef unsigned (*fBindAPI)(unsigned);
typedef void* (*fCreateContext)(void*, void*, void*, const int*);
typedef unsigned (*fMakeCurrent)(void*, void*, void*, void*);
typedef const unsigned char* (*fGetString)(unsigned);
static const char* crear_egl(char* info, int n) {
  void* h = dlopen("libEGL.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!h) return "no hay libEGL.so.1";
  void* hgl = dlopen("libGL.so.1", RTLD_NOW | RTLD_GLOBAL);
  fGetPlatformDisplay gpd = (fGetPlatformDisplay)dlsym(h, "eglGetPlatformDisplay");
  fGetDisplay gd = (fGetDisplay)dlsym(h, "eglGetDisplay");
  void* dpy = gpd ? gpd(0x31DD, NULL, NULL) : NULL;   // EGL_PLATFORM_SURFACELESS_MESA
  if (!dpy) dpy = gd(NULL);
  int ma, mi;
  if (!dpy || !((fInitialize)dlsym(h, "eglInitialize"))(dpy, &ma, &mi)) return "eglInitialize";
  int att[] = {0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3021, 8, 0x3025, 24, 0x3026, 8,
               0x3040, 0x0008, 0x3033, 0x0001, 0x3038};   // RGBA8 depth24 stencil8, OPENGL_BIT, PBUFFER_BIT
  void* cfg; int ncfg = 0;
  if (!((fChooseConfig)dlsym(h, "eglChooseConfig"))(dpy, att, &cfg, 1, &ncfg) || ncfg < 1) return "eglChooseConfig";
  ((fBindAPI)dlsym(h, "eglBindAPI"))(0x30A2);   // EGL_OPENGL_API
  void* ctx = ((fCreateContext)dlsym(h, "eglCreateContext"))(dpy, cfg, NULL, NULL);
  if (!ctx) return "eglCreateContext";
  if (!((fMakeCurrent)dlsym(h, "eglMakeCurrent"))(dpy, NULL, NULL, ctx)) return "eglMakeCurrent";
  fGetString gs = (fGetString)dlsym(hgl ? hgl : h, "glGetString");
  if (gs) { const unsigned char* r = gs(0x1F01); const unsigned char* v = gs(0x1F02);
    snprintf(info, n, "%s · OpenGL %s", r ? (const char*)r : "?", v ? (const char*)v : "?"); }
  return NULL;
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

const nombreLibreria = "libmujoco.so.3.3.0"

type libreria struct{ h unsafe.Pointer }

func abrirLibreria(ruta string) (*libreria, error) {
	cs := C.CString(ruta)
	defer C.free(unsafe.Pointer(cs))
	h := C.abrir_c(cs)
	if h == nil {
		return nil, errors.New(C.GoString(C.error_c()))
	}
	return &libreria{h}, nil
}

func (l *libreria) simbolo(nombre string) uintptr {
	cs := C.CString(nombre)
	defer C.free(unsafe.Pointer(cs))
	return uintptr(C.simbolo_c(l.h, cs))
}

func llamar(f uintptr, args ...uintptr) uintptr {
	var a [8]C.U
	for i, v := range args {
		a[i] = C.U(v)
	}
	return uintptr(C.llamar_c(unsafe.Pointer(f), C.int(len(args)), &a[0])) //nolint:govet
}

// System V x86-64: un mjrRect (4 int) por valor va en DOS registros enteros.
func (mj *MJ) argsRect(x, y, w, h int) []uintptr {
	return []uintptr{uintptr(uint32(x)) | uintptr(uint32(y))<<32, uintptr(uint32(w)) | uintptr(uint32(h))<<32}
}

func crearContextoGL() (string, error) {
	buf := make([]byte, 512)
	if e := C.crear_egl((*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))); e != nil {
		return "", errors.New(C.GoString(e))
	}
	return C.GoString((*C.char)(unsafe.Pointer(&buf[0]))), nil
}
