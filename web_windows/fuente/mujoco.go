package main

// Acceso a la librería C de MuJoCo (mujoco.dll en Windows) SIN Python y SIN cgo en Windows:
// se cargan las funciones por nombre y se llaman con argumentos enteros/punteros. Las
// estructuras de MuJoCo se reservan con mju_malloc y se leen/escriben con los offsets de
// offsets_mujoco.go (sacados de las cabeceras de la misma versión).
//
// La parte que depende del sistema (cargar la librería, llamar, pasar un mjrRect y crear el
// contexto de OpenGL) está en mujoco_windows.go / mujoco_linux.go.

import (
	"fmt"
	"unsafe"
)

type MJ struct {
	lib  *libreria
	fns  map[string]uintptr
	rect uintptr // mjrRect reutilizable (solo Windows lo pasa por puntero)
}

func cargarMuJoCo(ruta string) (*MJ, error) {
	l, err := abrirLibreria(ruta)
	if err != nil {
		return nil, err
	}
	mj := &MJ{lib: l, fns: map[string]uintptr{}}
	v := int32(mj.c("mj_version"))
	if v != mjVersionCabecera {
		return nil, fmt.Errorf("la librería es MuJoCo %d y este programa está hecho para la %d", v, mjVersionCabecera)
	}
	mj.rect = mj.reservar(16)
	return mj, nil
}

func (mj *MJ) fn(nombre string) uintptr {
	if f, ok := mj.fns[nombre]; ok {
		return f
	}
	f := mj.lib.simbolo(nombre)
	if f == 0 {
		panic("MuJoCo: no encuentro la función " + nombre)
	}
	mj.fns[nombre] = f
	return f
}

// c llama a una función de MuJoCo por su nombre.
func (mj *MJ) c(nombre string, args ...uintptr) uintptr { return llamar(mj.fn(nombre), args...) }

// ── memoria de C ──

func (mj *MJ) reservar(n int) uintptr {
	p := mj.c("mju_malloc", uintptr(n))
	if p == 0 {
		panic("MuJoCo: sin memoria")
	}
	b := unsafe.Slice((*byte)(ptr(p)), n)
	for i := range b {
		b[i] = 0
	}
	return p
}

func (mj *MJ) liberar(p uintptr) { mj.c("mju_free", p) }

func (mj *MJ) cadena(s string) uintptr {
	p := mj.reservar(len(s) + 1)
	copy(unsafe.Slice((*byte)(ptr(p)), len(s)), s)
	return p
}

//go:nocheckptr
func ptr(p uintptr) unsafe.Pointer { return unsafe.Pointer(p) } //nolint:govet

func leerI32(p uintptr, off uintptr) int32        { return *(*int32)(ptr(p + off)) }
func escribirI32(p uintptr, off uintptr, v int32) { *(*int32)(ptr(p + off)) = v }
func leerPtr(p uintptr, off uintptr) uintptr      { return *(*uintptr)(ptr(p + off)) }
func escribirF64(p uintptr, off uintptr, v float64) {
	*(*float64)(ptr(p + off)) = v
}
func escribirF32(p uintptr, off uintptr, v float32) { *(*float32)(ptr(p + off)) = v }
func leerF32(p uintptr, off uintptr) float32        { return *(*float32)(ptr(p + off)) }

// arrays de mjModel (int* / float*) por índice
func (mj *MJ) intModelo(m uintptr, campo uintptr, i int) int32 {
	return *(*int32)(ptr(leerPtr(m, campo) + uintptr(4*i)))
}

func cadenaC(p uintptr, max int) string {
	if p == 0 {
		return ""
	}
	b := unsafe.Slice((*byte)(ptr(p)), max)
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

func f32(v float64) float32 { return float32(v) }
