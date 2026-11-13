# src/BetaMouv/Video.py


from src.Base.Video import Video as BaseVideo
from .File import File
import cv2, yaml
from pathlib import Path
import pandas as pd
import numpy as np


class Video(BaseVideo): 

    file_cls = File 

    def compute_camera_shift(self, view: str, ref_path: str, img_path: str, save_as = None):
        from skimage.io import imread
        import numpy as np
        import pandas as pd
        from skimage.color import rgb2gray
        from dipy.align.transforms import TranslationTransform2D
        from dipy.align.imaffine import AffineRegistration
        from skimage import img_as_ubyte
        import matplotlib.pyplot as plt

        def load_image(path: str):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"'{path}' does not exist, can't open")
            return imread(str(path))

        frame_width = 512
        y_top_lim = 400
        y_bottom_lim = frame_width

        ref_img = load_image(ref_path)
        img = load_image(img_path)

        is_flip = view == "right"
        if is_flip :
            img = img[:, ::-1]                 # flip x axis
            print("flip")

        ref_img_cropped = ref_img[y_top_lim : y_bottom_lim]   # crop ref
        img_cropped = img[y_top_lim : y_bottom_lim]   

        # shift calculation 
        
        ref_gray = rgb2gray(ref_img_cropped)
        img_gray = rgb2gray(img_cropped)

        affreg = AffineRegistration()
        transform = TranslationTransform2D()
        tx = affreg.optimize(ref_gray, img_gray, transform, params0=None)

        shift_matrix = tx.affine
        dx, dy = shift_matrix[0, 2], shift_matrix[1, 2]

        if save_as: 
            from matplotlib.patches import Rectangle
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))

            axes[0, 0].imshow(ref_img, cmap="gray")
            axes[0, 0].set_title("ref img")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(img, cmap="gray")
            axes[0, 1].set_title("img")
            axes[0, 1].axis("off")

            # Rectangle: (x, y), width, height
            rect = Rectangle((0, y_top_lim), frame_width, y_bottom_lim,
                linewidth=5, edgecolor="red", facecolor="none")

            axes[0, 1].add_patch(rect)

            # Red = reference
            # Green = current frame
            stereo = np.zeros((frame_width, frame_width, 3), dtype=np.uint8)
            stereo[..., 0] = ref_img[..., 0]
            stereo[..., 1] = img[..., 1]

            axes[1, 0].imshow(stereo)
            axes[1, 0].set_title("superimposed")
            axes[1, 0].axis("off")

            img_gray = rgb2gray(img)
            shifted_img = tx.transform(img_gray, sampling_grid_shape=img_gray.shape[:2])

            stereo2 = np.zeros((frame_width, frame_width, 3), dtype=np.uint8)
            stereo2[..., 0] = ref_img[..., 0]
            stereo2[..., 1] = img_as_ubyte(np.clip(shifted_img, 0, 1))
    
            axes[1, 1].imshow(stereo2)
            axes[1, 1].set_title(f"shifted (dx={dx:.1f}, dy={dy:.1f})\nflip={is_flip}")
            axes[1, 1].axis("off")
    
            fig.tight_layout()
            fig.savefig(save_as)
            plt.close(fig)

        return dx, dy

