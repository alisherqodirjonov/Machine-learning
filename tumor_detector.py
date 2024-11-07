import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# Oynani yaratish
root = tk.Tk()
root.title("Tumor detector Dasturi")
root.geometry("900x600")

# Kirish yozuvi
greeting_label = tk.Label(root, text="Tumor detector Dasturiga Xush Kelibsiz!", font=("Times new Roman", 14))
greeting_label.pack(pady=(10,50))

instruction_label = tk.Label(root, text="Boshlash uchun 'START' tugmasini bosing va rasmni tanlang.", font=("Times new Roman", 20))
instruction_label.pack(pady=(10,300))

# Rasm tahlili funksiyasi
def analyze_image(filepath):
    image = cv2.imread(filepath, 0)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center) - 20 

    brain_mask = np.zeros_like(image)
    cv2.circle(brain_mask, center, radius, (255, 255, 255), -1)
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=brain_mask)

    block_size = 30
    max_percentage = 0
    max_block = None

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = masked_thresh[i:i+block_size, j:j+block_size]
            block_area = block_size * block_size
            white_area = np.sum(block == 255)
            percentage = (white_area / block_area) * 100
            
            if percentage > max_percentage:
                max_percentage = percentage
                max_block = (i, j)

    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if max_block:
        i, j = max_block
        cv2.rectangle(contour_image, (j, i), (j + block_size, i + block_size), (0, 255, 0), 2)
        cv2.putText(contour_image, f"Max: {max_percentage:.2f}%", (j, i - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original rasm')
    plt.subplot(1, 2, 2), plt.imshow(contour_image), plt.title('Bo\'laklarda qidirilgan')
    plt.show()

    if max_percentage > 50:
        messagebox.showinfo("Natija", "O\'simta aniqlandi!")
    else:
        messagebox.showinfo("Natija", "O\'simta aniqlanmadi!")

# Rasmni tanlash funksiyasi
def select_image():
    filepath = filedialog.askopenfilename(
        title="Rasmni tanlang",
        filetypes=[("Rasm fayllari", "*.png *.jpg *.jpeg *.bmp"), ("Boshqa fayllar", "*.*")]
    )
    if filepath:
        analyze_image(filepath)

# Boshlash tugmasi
start_button = tk.Button(root, text="START", font=("Times new Roman", 14), command=select_image)
start_button.pack(pady=20)

# Oynani ishga tushirish
root.mainloop()
