//go:build windows

package main

// Windows: mujoco.dll con syscall (sin cgo) y contexto de OpenGL con WGL en una ventana oculta.

import (
	"fmt"
	"syscall"
	"unsafe"
)

const nombreLibreria = "mujoco.dll"

type libreria struct{ dll *syscall.DLL }

func abrirLibreria(ruta string) (*libreria, error) {
	d, err := syscall.LoadDLL(ruta)
	if err != nil {
		return nil, err
	}
	return &libreria{d}, nil
}

func (l *libreria) simbolo(nombre string) uintptr {
	p, err := l.dll.FindProc(nombre)
	if err != nil {
		return 0
	}
	return p.Addr()
}

func llamar(f uintptr, args ...uintptr) uintptr {
	r, _, _ := syscall.SyscallN(f, args...)
	return r
}

// En Windows x64 una estructura de 16 bytes pasada por valor va como puntero a una copia.
func (mj *MJ) argsRect(x, y, w, h int) []uintptr {
	escribirI32(mj.rect, 0, int32(x))
	escribirI32(mj.rect, 4, int32(y))
	escribirI32(mj.rect, 8, int32(w))
	escribirI32(mj.rect, 12, int32(h))
	return []uintptr{mj.rect}
}

// ── OpenGL: ventana oculta + wglCreateContext (hay que llamarlo desde el hilo que renderiza) ──

type wndclassex struct {
	cbSize        uint32
	style         uint32
	lpfnWndProc   uintptr
	cbClsExtra    int32
	cbWndExtra    int32
	hInstance     uintptr
	hIcon         uintptr
	hCursor       uintptr
	hbrBackground uintptr
	lpszMenuName  *uint16
	lpszClassName *uint16
	hIconSm       uintptr
}

type pixelformatdescriptor struct {
	nSize, nVersion                                    uint16
	dwFlags                                            uint32
	iPixelType, cColorBits, cRedBits, cRedShift        uint8
	cGreenBits, cGreenShift, cBlueBits, cBlueShift     uint8
	cAlphaBits, cAlphaShift, cAccumBits, cAccumRedBits uint8
	cAccumGreenBits, cAccumBlueBits, cAccumAlphaBits   uint8
	cDepthBits, cStencilBits, cAuxBuffers, iLayerType  uint8
	bReserved                                          uint8
	dwLayerMask, dwVisibleMask, dwDamageMask           uint32
}

func crearContextoGL() (string, error) {
	user32 := syscall.NewLazyDLL("user32.dll")
	gdi32 := syscall.NewLazyDLL("gdi32.dll")
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	opengl32 := syscall.NewLazyDLL("opengl32.dll")
	if err := opengl32.Load(); err != nil {
		return "", fmt.Errorf("no hay opengl32.dll: %v", err)
	}

	hinst, _, _ := kernel32.NewProc("GetModuleHandleW").Call(0)
	clase, _ := syscall.UTF16PtrFromString("SkillShieldGemeloGL")
	wc := wndclassex{style: 0x0020 /* CS_OWNDC */, lpfnWndProc: user32.NewProc("DefWindowProcW").Addr(),
		hInstance: hinst, lpszClassName: clase}
	wc.cbSize = uint32(unsafe.Sizeof(wc))
	user32.NewProc("RegisterClassExW").Call(uintptr(unsafe.Pointer(&wc)))
	titulo, _ := syscall.UTF16PtrFromString("gemelo")
	hwnd, _, e := user32.NewProc("CreateWindowExW").Call(0, uintptr(unsafe.Pointer(clase)),
		uintptr(unsafe.Pointer(titulo)), 0x00CF0000 /* WS_OVERLAPPEDWINDOW, sin mostrar */, 0, 0, 64, 64, 0, 0, hinst, 0)
	if hwnd == 0 {
		return "", fmt.Errorf("CreateWindowEx: %v", e)
	}
	hdc, _, _ := user32.NewProc("GetDC").Call(hwnd)
	pfd := pixelformatdescriptor{nVersion: 1, dwFlags: 0x4 | 0x20 | 0x1, /* DRAW_TO_WINDOW|SUPPORT_OPENGL|DOUBLEBUFFER */
		cColorBits: 24, cAlphaBits: 8, cDepthBits: 24, cStencilBits: 8}
	pfd.nSize = uint16(unsafe.Sizeof(pfd))
	formato, _, e := gdi32.NewProc("ChoosePixelFormat").Call(hdc, uintptr(unsafe.Pointer(&pfd)))
	if formato == 0 {
		return "", fmt.Errorf("ChoosePixelFormat: %v", e)
	}
	if r, _, e := gdi32.NewProc("SetPixelFormat").Call(hdc, formato, uintptr(unsafe.Pointer(&pfd))); r == 0 {
		return "", fmt.Errorf("SetPixelFormat: %v", e)
	}
	ctx, _, e := opengl32.NewProc("wglCreateContext").Call(hdc)
	if ctx == 0 {
		return "", fmt.Errorf("wglCreateContext: %v", e)
	}
	if r, _, e := opengl32.NewProc("wglMakeCurrent").Call(hdc, ctx); r == 0 {
		return "", fmt.Errorf("wglMakeCurrent: %v", e)
	}
	getString := opengl32.NewProc("glGetString")
	ver, _, _ := getString.Call(0x1F02)
	ren, _, _ := getString.Call(0x1F01)
	return cadenaC(ren, 256) + " · OpenGL " + cadenaC(ver, 256), nil
}
