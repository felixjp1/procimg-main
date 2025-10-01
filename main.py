import tkinter as tk
from image_handler import load_image, display_image
from filters import *



img_cv = None
processed_img_cv = None
slider_window_ref = None   

def set_img_cv(img):
    global img_cv, processed_img_cv
    img_cv = img
    processed_img_cv = None
    if img_cv is not None:
        display_image(img_cv, original_image_canvas, original=True)

def apply_filter(filter_function, canvas, *args, **kwargs):
    global processed_img_cv
    if img_cv is None:
        return

    base = img_cv.copy()
    processed_img_cv = filter_function(base, *args, **kwargs)
    display_image(processed_img_cv, canvas, original=False)

def slider_window(root, canvas, filter_function, label, param_name, param_range, default_value):
    global slider_window_ref


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
root.title("Image Processing - Stage 1 (Classic)")
root.geometry("1085x550")
root.config(bg="#2e2e2e")

# barra de menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=lambda: set_img_cv(load_image(original_image_canvas, edited_image_canvas, slider_window_ref)))
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters (Classic)", menu=filters_menu)



filters_menu.add_command(label="Low Pass (Gaussian default)", command=lambda: apply_filter(low_pass, edited_image_canvas))
filters_menu.add_command(label="Low Pass (Gaussian σ)", command=lambda: slider_window(root, edited_image_canvas, low_pass_gaussian, "Sigma", "sigma", (1, 10), 3))
filters_menu.add_command(label="Low Pass (Mean)", command=lambda: slider_window(root, edited_image_canvas, low_pass_media, "Kernel Size", "kernel_size", (3, 21), 9))
filters_menu.add_separator()
filters_menu.add_command(label="High Pass (Laplacian)", command=lambda: apply_filter(high_pass, edited_image_canvas))
filters_menu.add_command(label="High Pass (Laplacian custom)", command=lambda: slider_window(root, edited_image_canvas, high_pass_laplacian, "Kernel Size", "kernel_value", (1, 15), 3))
filters_menu.add_command(label="High Pass (Sobel X)", command=lambda: apply_filter(high_pass_sobel, edited_image_canvas, direction='x'))
filters_menu.add_command(label="High Pass (Sobel Y)", command=lambda: apply_filter(high_pass_sobel, edited_image_canvas, direction='y'))
filters_menu.add_separator()
filters_menu.add_command(label="Thresholding (fixed)", command=lambda: slider_window(root, edited_image_canvas, thresholding_segmentation, "Threshold", "threshold_value", (0, 255), 90))
filters_menu.add_command(label="Thresholding (Otsu)", command=lambda: apply_filter(otsu_segmentation, edited_image_canvas))
filters_menu.add_separator()
filters_menu.add_command(label="Erosion", command=lambda: slider_window(root, edited_image_canvas, erosion, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Dilation", command=lambda: slider_window(root, edited_image_canvas, dilatation, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Opening", command=lambda: slider_window(root, edited_image_canvas, open, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Closing", command=lambda: slider_window(root, edited_image_canvas, close, "Kernel Size", "kernel_size", (1, 15), 5))

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
