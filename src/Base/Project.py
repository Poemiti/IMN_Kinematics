# src/Base/Project.py

from pathlib import Path


class Project:

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        desc = f"""
        ================= Project info =================
        Project name: {self.name}
        Config directory: config/{self.name}/
        ================================================\n"""
        return desc
