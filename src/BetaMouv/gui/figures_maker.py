from pathlib import Path
import tkinter as tk
from tkinter import ttk
import inspect

import rats_kinematics_utils.analysis.plot as p
import rats_kinematics_utils.analysis.plot_comparative as pc
import rats_kinematics_utils.analysis.behavior_plot as bp
import rats_kinematics_utils.analysis.inter_rat as ir

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.selected_files = None
        self.selected_functions = {}

        # populate view
        for path in model.metrics_paths:
            view.add_file_checkbox(path)

        for name in model.available_functions:
            view.add_function_checkbox(name)

        view.add_controls(self.on_run, self.on_reset)
        view.add_select_all_files_button(self.on_select_all_files)

    def on_select_all_files(self):
        self.view.select_all_files()

    def on_reset(self):
        self.view.reset_all()

    def on_run(self):
        self.selected_files = [
            path for path, var in self.view.file_vars.items()
            if var.get()
        ]

        self.selected_functions = {
            name: var.get()
            for name, var in self.view.function_vars.items()
        }

        self.view.quit()     # stop mainloop
        self.view.destroy()  # close window




class View(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Figure selector")

        # ---- frames
        self.file_frame = ttk.LabelFrame(self, text=" Datasets ")
        self.file_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.plot_frame = ttk.LabelFrame(self, text=" Plot functions ")
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.control_frame = ttk.Frame(self.plot_frame)
        self.control_frame.pack(side="bottom", fill="x", expand=False, padx=5, pady=5)

        # ---- variables
        self.file_vars = {}       # path -> BooleanVar
        self.function_vars = {}   # name -> BooleanVar

    # -------- buttons creation --------

    def add_file_checkbox(self, path: Path):
        var = tk.BooleanVar()
        chk = tk.Checkbutton(self.file_frame, text=path.stem, variable=var)
        chk.pack(anchor="w", padx=5, pady=5)
        self.file_vars[path] = var

    def add_function_checkbox(self, name: str):
        var = tk.BooleanVar()
        chk = tk.Checkbutton(self.plot_frame, text=name, variable=var)
        chk.pack(anchor="w", padx=5, pady=5)
        self.function_vars[name] = var

    def add_controls(self, on_run, on_reset):
        ttk.Button(self.control_frame, text="Run", command=on_run).pack(side="right", padx=5)
        ttk.Button(self.control_frame, text="Reset", command=on_reset).pack(side="right", padx=5)

    def add_select_all_files_button(self, callback):
        ttk.Button(
            self.file_frame,
            text="Select all files",
            command=callback
        ).pack(anchor="e", side="bottom", padx=5, pady=5)

    # -------- utilities --------

    def select_all_files(self):
        for var in self.file_vars.values():
            var.set(True)

    def reset_all(self):
        for var in self.file_vars.values():
            var.set(False)
        for var in self.function_vars.values():
            var.set(False)



class Model:
    def __init__(self, filenames: list[Path], kind: str = "single"):
        """
        Initiate app model
        filenames: list[Path]: list of all .joblib file to use
        kind: str: select with ploting script to use: 
            - single
            - comparative
            - inter_rat
        """
        self.metrics_paths = filenames

        if kind == "single" : 
            self.available_functions = {
                name: func
                for name, func in inspect.getmembers(p, inspect.isfunction)
                if name.startswith("plot")}
            
        elif kind == "comparative" : 
            self.available_functions = {
                name: func
                for name, func in inspect.getmembers(pc, inspect.isfunction)
                if name.startswith("plot")}
            
        elif kind == "inter_rat": 
            self.available_functions = {
                name: func
                for name, func in inspect.getmembers(ir, inspect.isfunction)
                if name.startswith("plot")}
            
        elif kind == "behavior": 
            self.available_functions = {
                name: func
                for name, func in inspect.getmembers(bp, inspect.isfunction)
                if name.startswith("plot")}
            


def load_figure_maker(filenames, kind: str) : 
    """
    filenames: list[Path]: list of all .joblib file to use
    kind: str: select with ploting script to use: 
        - single
        - comparative
        - inter_rat
    """
    model = Model(filenames, kind)
    view = View()
    controller = Controller(model, view)
    view.mainloop()

    filenames : list[Path] = controller.selected_files
    plot_choice : list[bool] = controller.selected_functions

    return filenames, plot_choice
