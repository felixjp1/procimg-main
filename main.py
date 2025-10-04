import tkinter as tk
from tkinter import messagebox
from image_handler import load_image, display_image
from filters import *

img_cv = None
processed_img_cv = None
slider_window_ref = None
filters_menu = None  
filters_items_indices = [] 

def set_filters_state(state: str):
    if filters_menu is None:
        return
    for i in filters_items_indices:
        try:
            filters_menu.entryconfig(i, state=state)
        except Exception:
            pass

def set_img_cv(img):
    global img_cv, processed_img_cv
    img_cv = img
    processed_img_cv = None
    if img_cv is not None:
        display_image(img_cv, original_image_canvas, original=True)
        set_filters_state("normal")
    else:
        set_filters_state("disabled")

def _ensure_image_loaded() -> bool:
    if img_cv is None:
        messagebox.showwarning("Atenção", "Carregue uma imagem primeiro (Arquivo → Carregar imagem).")
        return False
    return True

def apply_filter(filter_function, canvas, *args, **kwargs):
    global processed_img_cv
    if not _ensure_image_loaded():
        return
    base = img_cv.copy()
    processed_img_cv = filter_function(base, *args, **kwargs)
    if processed_img_cv is None:
        messagebox.showerror("Erro", "Falha ao aplicar o filtro.")
        return
    display_image(processed_img_cv, canvas, original=False)

def slider_window(root, canvas, filter_function, label, param_name, param_range, default_value):
    global slider_window_ref
    if not _ensure_image_loaded():
        return
    if slider_window_ref is not None:
        try:
            slider_window_ref.destroy()
        except Exception:
            pass
    slider_window_ref = tk.Toplevel(root)
    slider_window_ref.title(f"{label}")
    def on_slider_change(value):
        val = int(float(value))
        apply_filter(filter_function, canvas, **{param_name: val})
    slider = tk.Scale(
        slider_window_ref,
        from_=param_range[0],
        to=param_range[1],
        orient=tk.HORIZONTAL,
        label=label,
        command=on_slider_change,
    )
    slider.pack(padx=20, pady=20)
    slider.set(default_value)


root = tk.Tk()
root.title("Processamento de Imagens — Etapa 1 (Clássica)")
root.geometry("1085x600")
root.config(bg="#2e2e2e")

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Arquivo", menu=file_menu)
file_menu.add_command(label="Carregar imagem", command=lambda: set_img_cv(load_image(original_image_canvas, edited_image_canvas, slider_window_ref)))
file_menu.add_separator()
file_menu.add_command(label="Sair", command=root.quit)

# menu Filtros
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filtros (Clássico)", menu=filters_menu)

def add_filter_item(texto, cmd):
    filters_menu.add_command(label=texto, command=cmd)
    filters_items_indices.append(filters_menu.index("end"))

def add_separator():
    filters_menu.add_separator()


add_filter_item("Passa-baixa (Gaussiano padrão)", lambda: apply_filter(low_pass, edited_image_canvas))
add_filter_item("Passa-baixa (Gaussiano σ)", lambda: slider_window(root, edited_image_canvas, low_pass_gaussian, "Sigma", "sigma", (1, 10), 3))
add_filter_item("Passa-baixa (Média)", lambda: slider_window(root, edited_image_canvas, low_pass_media, "Tamanho do kernel", "kernel_size", (3, 21), 9))
add_separator()
add_filter_item("Passa-alta (Laplaciano)", lambda: apply_filter(high_pass, edited_image_canvas))
add_filter_item("Passa-alta (Laplaciano custom)", lambda: slider_window(root, edited_image_canvas, high_pass_laplacian, "Tamanho do kernel", "kernel_value", (1, 15), 3))
add_filter_item("Passa-alta (Sobel X)", lambda: apply_filter(high_pass_sobel, edited_image_canvas, direction='x'))
add_filter_item("Passa-alta (Sobel Y)", lambda: apply_filter(high_pass_sobel, edited_image_canvas, direction='y'))
add_separator()
add_filter_item("Limiarização (valor fixo)", lambda: slider_window(root, edited_image_canvas, thresholding_segmentation, "Limiar", "threshold_value", (0, 255), 90))
add_filter_item("Limiarização (Otsu)", lambda: apply_filter(otsu_segmentation, edited_image_canvas))
add_separator()
add_filter_item("Erosão", lambda: slider_window(root, edited_image_canvas, erosion, "Tamanho do kernel", "kernel_size", (1, 15), 5))
add_filter_item("Dilatação", lambda: slider_window(root, edited_image_canvas, dilatation, "Tamanho do kernel", "kernel_size", (1, 15), 5))
add_filter_item("Abertura (Opening)", lambda: slider_window(root, edited_image_canvas, open, "Tamanho do kernel", "kernel_size", (1, 15), 5))
add_filter_item("Fechamento (Closing)", lambda: slider_window(root, edited_image_canvas, close, "Tamanho do kernel", "kernel_size", (1, 15), 5))

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

footer = tk.Label(root, text="João Pedro Felix & Douglas Costa", fg="#cccccc", bg="#2e2e2e")
footer.grid(row=1, column=0, columnspan=2, pady=(0, 10))

set_filters_state("disabled")

root.mainloop()
