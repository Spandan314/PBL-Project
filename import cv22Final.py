import cv2
import pytesseract
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageEnhance
import pyperclip
import numpy as np
import subprocess
from collections import Counter
import re

# ===================== TESSERACT SETUP VALIDATION =====================
def validate_tesseract():
    try:
        subprocess.run(['tesseract', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def install_tesseract_instructions():
    message = """
Tesseract OCR is not installed or not in your system PATH.

REQUIRED INSTALLATION:
1. Download installer from:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer and CHECK:
   [✓] "Add Tesseract to your system PATH"
3. Default install location:
   C:\\Program Files\\Tesseract-OCR
"""
    messagebox.showerror("Tesseract Not Found", message)
    return False

# ===================== OCR PROCESSING =====================
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)

        return processed
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

def extract_text(image_path, lang='eng', handwriting=False):
    try:
        processed_img = preprocess_image(image_path)
        config = r'--oem 1 --psm 6' if handwriting else r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, lang=lang, config=config)
        return text.strip() or "No text detected"
    except Exception as e:
        raise ValueError(f"OCR failed: {str(e)}")

# ===================== ENHANCED GUI APPLICATION =====================
class TextExtractorApp:
    def __init__(self, root):
        self.root = root
        self.current_image = None
        self.batch_mode = False
        self.handwriting_mode = False
        self.setup_ui()

        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
        self.configure_tesseract()

    def configure_tesseract(self):
        try:
            if os.path.exists(self.tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                os.environ['TESSDATA_PREFIX'] = self.tessdata_path
            elif not validate_tesseract():
                install_tesseract_instructions()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Tesseract setup failed:\n{str(e)}")

    def setup_ui(self):
        self.root.title("Professional Text Extractor")
        self.root.geometry("1000x800")
        self.root.minsize(800, 700)
        self.root.configure(bg='#ecf0f1')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'), foreground='#ffffff', background='#2c3e50')
        style.map('Primary.TButton', background=[('active', '#34495e')])
        style.configure('TFrame', background='#ecf0f1')
        style.configure('TLabelFrame', background='#ecf0f1', font=('Segoe UI', 10, 'bold'))
        style.configure('TCombobox', font=('Segoe UI', 10))
        style.configure('TLabel', background='#ecf0f1')

        # Main container frame
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Language and options frame
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(pady=5, fill=tk.X)

        ttk.Label(lang_frame, text="📚 OCR Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value='eng')
        self.lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                     values=['eng', 'hin', 'fra', 'spa', 'deu'], width=8, state='readonly')
        self.lang_combo.pack(side=tk.LEFT, padx=10)

        self.handwriting_var = tk.BooleanVar()
        self.batch_var = tk.BooleanVar()
        ttk.Checkbutton(lang_frame, text="✍ Handwriting", variable=self.handwriting_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(lang_frame, text="🗂 Batch Mode", variable=self.batch_var).pack(side=tk.LEFT)

        # Image control buttons
        img_control_frame = ttk.Frame(main_frame)
        img_control_frame.pack(pady=10, fill=tk.X)

        ttk.Button(img_control_frame, text="📁 Open Image(s)", style='Primary.TButton', 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_control_frame, text="🔄 Re-process", style='Primary.TButton', 
                  command=self.reprocess_current_image).pack(side=tk.LEFT, padx=5)

        # Image preview frame
        self.img_frame = ttk.LabelFrame(main_frame, text="🖼️ Image Preview", padding=10)
        self.img_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Canvas for image with scrollbars
        self.img_canvas = tk.Canvas(self.img_frame, bg='white')
        self.img_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Text output frame
        text_frame = ttk.LabelFrame(main_frame, text="📝 Extracted Text", padding=10)
        text_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        self.text_output = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            font=('Segoe UI', 10), 
            padx=10, 
            pady=10,
            bg='#ffffff',
            width=40,
            height=10
        )
        self.text_output.pack(fill=tk.BOTH, expand=True)

        # Statistics label
        self.stats_label = ttk.Label(main_frame, text="📊 Stats: Words: 0 | Characters: 0 | Most Frequent: N/A")
        self.stats_label.pack(pady=5)

        # Action buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10, fill=tk.X)

        ttk.Button(btn_frame, text="📋 Copy Text", style='Primary.TButton', 
                  command=self.copy_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Save As...", style='Primary.TButton', 
                  command=self.save_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="❌ Clear All", style='Primary.TButton', 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=('Segoe UI', 9), 
            background='#34495e', 
            foreground='#ffffff'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        self.status_var.set(message)
        self.root.after(5000, lambda: self.status_var.set("Ready"))

    def update_stats(self, text):
        words = re.findall(r'\w+', text)
        char_count = len(text)
        word_count = len(words)
        freq = Counter(words)
        most_common = freq.most_common(1)
        top_word = most_common[0][0] if most_common else 'N/A'
        self.stats_label.config(text=f"📊 Stats: Words: {word_count} | Characters: {char_count} | Most Frequent: {top_word}")

    def load_image(self):
        filetypes = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        if self.batch_var.get():
            filepaths = filedialog.askopenfilenames(filetypes=filetypes)
        else:
            filepath = filedialog.askopenfilename(filetypes=filetypes)
            filepaths = [filepath] if filepath else []

        if not filepaths:
            return

        self.text_output.delete(1.0, tk.END)
        full_text = ""

        for filepath in filepaths:
            if filepath:
                try:
                    self.current_image = filepath
                    text = extract_text(filepath, self.lang_var.get(), self.handwriting_var.get())
                    full_text += f"\n\n==== {os.path.basename(filepath)} ====\n{text}"
                except Exception as e:
                    full_text += f"\n\n==== {os.path.basename(filepath)} ====\nError: {str(e)}"

        self.text_output.insert(tk.END, full_text.strip())
        self.update_stats(full_text)
        self.show_preview(self.current_image)
        self.update_status(f"Processed {len(filepaths)} image(s)")

    def show_preview(self, image_path):
        if not image_path:
            return
            
        try:
            img = Image.open(image_path)
            img = ImageEnhance.Contrast(img).enhance(1.5)
            
            # Calculate aspect ratio preserving dimensions
            max_size = (450, 450)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            
            # Clear previous image
            self.img_canvas.delete("all")
            
            # Create image on canvas
            self.img_canvas.config(width=img.width, height=img.height)
            self.img_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            
            # Keep reference to prevent garbage collection
            self.img_canvas.image = img_tk
        except Exception as e:
            self.update_status(f"Error displaying image: {str(e)}")

    def reprocess_current_image(self):
        if not self.current_image:
            self.update_status("No image loaded to reprocess!")
            return
            
        try:
            text = extract_text(self.current_image, self.lang_var.get(), self.handwriting_var.get())
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, f"==== {os.path.basename(self.current_image)} ====\n{text}")
            self.update_stats(text)
            self.update_status(f"Re-processed: {os.path.basename(self.current_image)}")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to reprocess image:\n{str(e)}")

    def copy_text(self):
        text = self.text_output.get(1.0, tk.END).strip()
        if text:
            pyperclip.copy(text)
            self.update_status("Text copied to clipboard!")
        else:
            self.update_status("Warning: No text to copy!")

    def save_text(self):
        text = self.text_output.get(1.0, tk.END).strip()
        if not text:
            self.update_status("Warning: No text to save!")
            return

        filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=filetypes,
            initialfile="extracted_text"
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.update_status(f"Text saved to: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")

    def clear_all(self):
        self.current_image = None
        self.text_output.delete(1.0, tk.END)
        self.img_canvas.delete("all")
        self.img_canvas.image = None
        self.stats_label.config(text="📊 Stats: Words: 0 | Characters: 0 | Most Frequent: N/A")
        self.update_status("All content cleared")

# ===================== RUN APPLICATION =====================
if __name__ == "__main__":
    root = tk.Tk()
    app = TextExtractorApp(root)
    root.mainloop()