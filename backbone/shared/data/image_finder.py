"""Resolve image paths under a folder (inference; CL-agnostic)."""
import os

from tqdm import tqdm


class ImageFinder:
    """Recursive image lookup by file name or relative path."""

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self._index = None

    def build_index(self):
        index = {}
        if not self.image_folder:
            self._index = index
            return
        from backbone.shared.runtime_logging import is_debug, log_infer

        log_infer(f"Building image index for folder: {self.image_folder}")
        walker = os.walk(self.image_folder)
        if is_debug():
            walker = tqdm(walker, dynamic_ncols=True, leave=False)

        for dirpath, _, filenames in walker:
            for filename in filenames:
                lower = filename.lower()
                if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                index.setdefault(filename, os.path.join(dirpath, filename))
        self._index = index

    def find(self, image_file):
        if image_file is None:
            raise FileNotFoundError(f"Image not found: {image_file} under image_folder={self.image_folder}")

        direct_path = os.path.join(self.image_folder, image_file)
        if os.path.isfile(direct_path):
            return direct_path

        base_name = os.path.basename(image_file)
        root, ext = os.path.splitext(base_name)
        if ext == "":
            for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                p = os.path.join(self.image_folder, base_name + e)
                if os.path.isfile(p):
                    return p

        if self._index is None:
            self.build_index()
        candidate = self._index.get(base_name)
        if candidate and os.path.isfile(candidate):
            return candidate

        if ext == "":
            for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidate = self._index.get(base_name + e)
                if candidate and os.path.isfile(candidate):
                    return candidate

        raise FileNotFoundError(f"Image not found: {image_file} under image_folder={self.image_folder}")
