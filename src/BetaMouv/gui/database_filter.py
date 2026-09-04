from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import ttk
import sys



class Controller:           # where we controle what does each button do 
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.filtered_dataset = None
        self.dataset_name = view.dataset_name_input()

        # display existing database
        for database_path in sorted(model.existing_database) : 
            view.database_buttons(database_path.stem,
                                  lambda p=database_path: self.on_database_selected(p))

        # display filter buttons
        for column in model.filters.keys():
            values = model.get_unique_values(column)
            view.filter_buttons(column,
                                values,
                                self.on_filter_selected)

        view.save_button(self.on_save)
        view.reset_button(self.on_reset)
        view.show_results(model.get_filtered())

    def on_filter_selected(self, column, value):
        self.model.set_filter(column, value)
        self.view.highlight_selection(column, value)
        self.view.show_results(self.model.get_filtered())

    def on_save(self) : 
        self.filtered_dataset = self.model.get_filtered()
        self.view.destroy()

    def on_reset(self) : 
        self.model.reset_filters()
        self.view.clear_all_selections()
        self.view.show_results(self.model.get_filtered())

    def on_database_selected(self, name) : 
        self.filtered_dataset = self.model.get_existing_database(name)
        self.view.destroy()




class View(tk.Tk):          # does all the graphical aspect
    def __init__(self):
        super().__init__()
        self.title("Video File Filter")
        self.geometry("1900x1000")
        self.frames = {}
        self.buttons = {}
        self.result_box = None
        
        # ---------- TOP FRAMES ----------
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # ---------- BOTTOM CONTAINER ----------
        self.bottom_frame = ttk.Frame(self)
        self.bottom_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.filter_frame = ttk.Frame(self.bottom_frame)
        self.filter_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.database_frame = ttk.Frame(self.bottom_frame)
        self.database_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.save_reset_frame = ttk.Frame(self.filter_frame)
        self.save_reset_frame.pack(side="bottom", fill="x", expand=False, padx=5, pady=5)

        # ---------- labels ----------
        self.n_file_label = ttk.Label(self.info_frame, text="Files number = 0")
        self.n_file_label.pack(anchor="w")

        self.database_label = ttk.Label(
            self.database_frame,
            text="Already existing database:"
        )
        self.database_label.pack(anchor="w")

    # ---------------------------- buttons -------------------------------

    def save_button(self, callback) :
        btn = ttk.Button(
            self.save_reset_frame,
            text="save",
            command=callback
        )
        btn.pack(side="right", padx=5, pady=2)
        self.buttons["save"] = btn


    def reset_button(self, callback) :
        btn = ttk.Button(
            self.save_reset_frame,
            text="reset",
            command=callback
        )
        btn.pack(side="right", padx=5, pady=2)
        self.buttons["reset"] = btn


    def filter_buttons(self, name, values, callback):
        frame = ttk.LabelFrame(self.filter_frame, text=name)
        frame.pack(fill="both", padx=5, pady=5)

        self.buttons[name] = {}

        for val in values:
            btn = ttk.Button(
                frame,
                text=str(val),
                command=lambda v=val: callback(name, v)
            )
            btn.pack(side="left", padx=2, pady=3)
            self.buttons[name][val] = btn

        self.frames[name] = frame


    def database_buttons(self, name, callback) : 
        btn = ttk.Button(self.database_frame,
                         text=str(name),
                         command=callback
        )
        btn.pack(fill="both", padx=5, pady=5)

    # --------------------------- Entry (user input) --------------------------------
    
    # input string var to rename the filtered dataset
    def dataset_name_input(self):
        frame = ttk.LabelFrame(self.filter_frame, text="Enter dataset name (for saving) :")
        frame.pack(fill="x", padx=5, pady=5)

        self.dataset_name = tk.StringVar()

        entry = ttk.Entry(frame, textvariable=self.dataset_name, width=40)
        entry.pack(side="left", padx=5, pady=5)

        return self.dataset_name
    
    # ----------------------------- utilities ------------------------------------------
    
    def clear_all_selections(self):
        for group in self.buttons:
            if group in ("save", "reset"):
                continue
            for btn in self.buttons[group].values():
                btn.state(["!pressed"])


    def highlight_selection(self, group, selected_value):
        for val, btn in self.buttons[group].items():
            btn.state(["!pressed"])
            if val == selected_value:
                btn.state(["pressed"])

    def show_results(self, dataframe) :
        # Update file count
        self.n_file_label.config(text=f"Files number = {len(dataframe)}")

        # Replace result box only
        if self.result_box:
            self.result_box.destroy()

        self.result_box = tk.Text(self.result_frame, height=10)
        self.result_box.pack(fill="both", expand=True)

        self.result_box.insert("end", dataframe.to_string())
    


class Model:                # focus on filtration manipulation
    def __init__(self, dataframe: pd.DataFrame, database_dir : Path):
        self._df = dataframe
        self.filters = {column: None for column in dataframe.columns[1:]} # add filter for every columns of the dataframe
        self.existing_database = [path for path in database_dir.glob("*.csv")]

    def set_filter(self, key, value):
        self.filters[key] = value

    def reset_filters(self) : 
        for key in self.filters : 
            self.filters[key] = None

    def get_unique_values(self, column):
        return sorted(self._df[column].unique())

    def get_filtered(self) -> pd.DataFrame : 
        df = self._df
        for key, value in self.filters.items():
            if value is not None:
                df = df[df[key] == value]
        return df
    
    def get_existing_database(self, csv_path : Path) -> pd.DataFrame : 
        return pd.read_csv(csv_path, index_col=0)





def load_database(files_path, database_path, source: str): 

    if source == "video" :
        raw_database = make_database(files_path, is_video)
    elif source == "csv" : 
        raw_database = make_database(files_path, is_csv)
    else : 
        raise ValueError("Invalid source. Available source are :\n\tcsv \n\tvideo \nChange source type in 'config.yaml'")    

    model = Model(raw_database, database_path)
    view = View()
    controller = Controller(model, view)
    view.mainloop()

    if controller.filtered_dataset is None: 
        print("No database seletected, stop !")
        sys.exit()

    database = controller.filtered_dataset.reset_index(drop=True)
    print(database)

    # save database
    if controller.dataset_name.get() :
        dataset_name = f"{controller.dataset_name.get().strip()}.csv"
        database.to_csv(database_path / dataset_name)
        print(f"\nFiltered dataset saved as : {database_path / dataset_name}")

    print(f"\nNumber of files in database : {len(database)}")

    return database



def make_database(root_dir : Path, satisfy_condition):
    """
    Build a database of video metadata from a directory tree.
    The directory is recursively scanned, and files satisfying
    a given condition are classified and stored in a DataFrame.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory to scan.
    satisfy_condition : callable
        Function that takes a filename and returns True if it
        should be included.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing extracted metadata for all matching files.
    """
    from src.BetaMouv.File import File

    sorted_videos = []
    for file_path in root_dir.rglob("*"):

        file = File(file_path)

        if satisfy_condition(file.name):
            file.classify_file(file_path, sorted_videos)
    return pd.DataFrame(sorted_videos)




def is_video(filename : str) -> bool : 
    """
    Check if a file is a video file.
    Supported video formats are ``.avi`` and ``.mp4``.
    """
    video_extensions = (".avi", ".mp4")
    return filename.lower().endswith(video_extensions)


def is_csv(filename : str) -> bool : 
    """
    Check if a file is CSV
    """
    return filename.lower().endswith(".csv")



if __name__ == "__main__" : 

    print("No main")