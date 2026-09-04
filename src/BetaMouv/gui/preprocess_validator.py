from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk



class Controller:
    def __init__(self, model, view):

        self.model = model
        self.view = view
        self.i = 0

        view.set_callbacks(
            self.previous,
            self.keep_raw,
            self.keep_interpolate,
            self.reject
        )

        self.update()

    def update(self):

        if self.i >= len(self.model):
            self.view.destroy()
            return

        path = self.model.get_path(self.i)
        self.view.show_image(path, self.i, len(self.model))
        self.view.update_progress(self.i, len(self.model))

    def keep_raw(self):
        self.model.validation[self.model.get_pathname(self.i)] = "raw"
        self.i += 1
        self.update()

    def keep_interpolate(self):
        self.model.validation[self.model.get_pathname(self.i)] = "interpolate"
        self.i += 1
        self.update()

    def reject(self):
        self.model.validation[self.model.get_pathname(self.i)] = "rejected"
        self.i += 1
        self.update()

    def previous(self):
        if self.i > 0:
            self.i -= 1
            self.update()



class View(tk.Tk):
    def __init__(self):
        super().__init__()
        self.stop_requested = False
        self.title("Preprocessing validator")

        # layout
        self.info = ttk.Label(self, text="")
        self.info.pack(pady=5)

        self.progress = ttk.Progressbar(
            self,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(pady=5)

        self.image_label = ttk.Label(self)
        self.image_label.pack(expand=True)

        self.buttons = ttk.Frame(self)
        self.buttons.pack(pady=10)

        self.btn_prev = ttk.Button(self.buttons, text="Previous")
        self.btn_keepRaw = ttk.Button(self.buttons, text="Keep raw")
        self.btn_keepInterpolate = ttk.Button(self.buttons, text="Keep interpolate")
        self.btn_reject = ttk.Button(self.buttons, text="Reject both")
        self.btn_stop = tk.Button(self.buttons, text="Stop and save", command=self.stop, bg="red")

        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_keepRaw.grid(row=0, column=1, padx=5)
        self.btn_keepInterpolate.grid(row=0, column=2, padx=5)
        self.btn_reject.grid(row=0, column=3, padx=5)
        self.btn_stop.grid(row=0, column=4, padx=5)

        self.image = None

    def set_callbacks(self, prev, raw, interpolate, reject):
        self.btn_prev.config(command=prev)
        self.btn_keepRaw.config(command=raw)
        self.btn_keepInterpolate.config(command=interpolate)
        self.btn_reject.config(command=reject)

    def show_image(self, path, index, total):

        name = path.stem.replace("_interpolation", "")
        img = Image.open(path)
        self.image = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.image)
        self.info.config(text=f"{name}\n{index+1}/{total}")

    def update_progress(self, index, total):

        self.progress["maximum"] = total
        self.progress["value"] = index + 1


    def stop(self) : 
        self.stop_requested = False
        self.quit()
        self.destroy()
        return


    # -------------- key binds shortcuts -----------------

    def bind_keys(self, controller):

        self.bind("<Right>", lambda e: controller.reject())
        self.bind("<Down>", lambda e: controller.keep_raw())
        self.bind("<Up>", lambda e: controller.keep_interpolate())
        self.bind("<Left>", lambda e: controller.previous())
            


class Model:
    def __init__(self, trajfig_dir: Path):
        self.paths = sorted(trajfig_dir.rglob("*_interpolation.png"))
        self.validation = {}

    def get_path(self, i) -> Path:
        return self.paths[i]
    
    def get_pathname(self, i):
        full_name = self.get_path(i)
        return full_name.stem.replace("_interpolation", "")

    def __len__(self):
        return len(self.paths)





def load_preprocess_validator(dir) : 

    model = Model(dir) 
    view = View()
    controller = Controller(model, view)
    view.bind_keys(controller)
    view.mainloop()

    if view.stop_requested : 
        return

    return model.validation





if __name__ == "__main__" : 
    
    print("No main")