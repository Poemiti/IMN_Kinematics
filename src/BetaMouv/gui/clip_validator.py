from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2



class Controller:
    def __init__(self, model, view):

        self.model = model
        self.view = view
        self.i = 0

        view.set_callbacks(
            self.previous,
            self.keep_all,
            self.keep_laser,
            self.reject
        )

        self.update()

    def update(self):

        if self.i >= len(self.model):
            self.view.destroy()
            return

        path = self.model.get_path(self.i)
        self.view.show_video(path, self.i, len(self.model))
        self.view.update_progress(self.i, len(self.model))

    def keep_all(self):
        self.model.validation[self.model.get_pathname(self.i)] = "keep_all"
        self.i += 1
        self.update()

    def keep_laser(self):
        self.model.validation[self.model.get_pathname(self.i)] = "keep_laser"
        self.i += 1
        self.update()

    def reject(self):
        self.model.validation[self.model.get_pathname(self.i)] = "crossed_paw"
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
        self.title("Clip validator")

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

        self.video_label = ttk.Label(self)
        self.video_label.pack(expand=True)

        self.buttons = ttk.Frame(self)
        self.buttons.pack(pady=10)

        self.btn_prev = ttk.Button(self.buttons, text="Previous")
        self.btn_keep_laser = ttk.Button(self.buttons, text="Keep ONLY laser")
        self.btn_keep_all = ttk.Button(self.buttons, text="Keep ALL traj")
        self.btn_reject = ttk.Button(self.buttons, text="Reject")
        self.btn_stop = tk.Button(self.buttons, text="Stop and save", command=self.stop, bg="red")

        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_keep_all.grid(row=0, column=1, padx=5)
        self.btn_keep_laser.grid(row=0, column=2, padx=5)
        self.btn_reject.grid(row=0, column=3, padx=5)
        self.btn_stop.grid(row=0, column=4, padx=5)

        self.image = None

    def set_callbacks(self, prev, keep_all, keep_laser, reject):
        self.btn_prev.config(command=prev)
        self.btn_keep_all.config(command=keep_all)
        self.btn_keep_laser.config(command=keep_laser)
        self.btn_reject.config(command=reject)


    def show_video(self, path, index, total):
        name = path.stem.replace("_annotated", "")
        
        # Stop previous loop
        if hasattr(self, "after_id"):
            try:
                self.video_label.after_cancel(self.after_id)
            except Exception:
                pass

        # Release previous video
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        name = path.stem.replace("_interpolation", "")
        self.cap = cv2.VideoCapture(str(path))

        #  unique token to invalidate old loops
        self._video_token = object()
        token = self._video_token

        def update_frame():
            #  Stop if this is an old loop
            if token is not self._video_token:
                return

            ret, frame = self.cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.image = ImageTk.PhotoImage(img)
                self.video_label.config(image=self.image)

                self.after_id = self.video_label.after(30, update_frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.after_id = self.video_label.after(30, update_frame)

        update_frame()

        update_frame()
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
        self.bind("<Down>", lambda e: controller.keep_laser())
        self.bind("<Up>", lambda e: controller.keep_all())
        self.bind("<Left>", lambda e: controller.previous())
            


class Model:
    def __init__(self, trajfig_dir: Path):
        self.paths = sorted(trajfig_dir.rglob("*_annotated.mp4"))
        self.validation = {}

    def get_path(self, i) -> Path:
        return self.paths[i]
    
    def get_pathname(self, i):
        full_name = self.get_path(i)
        return full_name.stem.replace("_annotated", "")

    def __len__(self):
        return len(self.paths)





def load_clip_validator(dir) : 

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