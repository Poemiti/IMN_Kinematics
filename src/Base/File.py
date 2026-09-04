# src/Base/File.py

from pathlib import Path

class File: 

    def __init__(self, path: str):

        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"File does not exist: {self.path}")
        
        self.name = self.path.stem