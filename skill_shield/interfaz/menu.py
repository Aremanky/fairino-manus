"""
Menú gráfico (Tkinter) de selección de modo.

NOTA: actualmente main() usa zenity; esta función se conserva tal cual estaba.

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""


def mostrar_menu_grafico(cola_retorno):
    import tkinter as tk
    ventana = tk.Tk()
    ventana.title("Skill Shield Nivel 6")
    ventana.geometry("380x150")

    # Forzar que aparezca visible y encima de todo
    ventana.lift()
    ventana.attributes('-topmost', True)
    ventana.focus_force()
    ventana.update()  # <- clave: fuerza el refresco antes del mainloop
    try:
        ventana.eval('tk::PlaceWindow . center')
    except:
        pass

    modo_elegido = "SIMULACION"

    def set_simulacion():
        nonlocal modo_elegido
        modo_elegido = "SIMULACION"
        ventana.quit()

    def set_hardware():
        nonlocal modo_elegido
        modo_elegido = "HARDWARE_REAL"
        ventana.quit()

    def set_sliders():
        nonlocal modo_elegido
        modo_elegido = "GEMELO_SLIDERS"
        ventana.quit()

    ventana.geometry("420x190")
    tk.Label(ventana, text="¿Qué modo de operación deseas iniciar?", font=("Arial", 11, "bold")).pack(pady=15)
    tk.Button(ventana, text="💻 Solo Simulación Virtual (MuJoCo)", command=set_simulacion, width=40, bg="#e0f7fa").pack(pady=5)
    tk.Button(ventana, text="🤖 Mano Física (USB) + Gemelo Digital", command=set_hardware, width=40, bg="#e8f5e9").pack(pady=5)
    tk.Button(ventana, text="🎚️ Gemelo Digital (sliders): brazo con sliders", command=set_sliders, width=40, bg="#fff3e0").pack(pady=5)
    
    ventana.mainloop()
    ventana.destroy()
    cola_retorno.put(modo_elegido)
