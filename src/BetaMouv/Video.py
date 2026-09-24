# src/BetaMouv/Video.py


from src.Base.Video import Video as BaseVideo
from .File import File
import cv2, yaml
from pathlib import Path

from skimage.io import imread
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from dipy.align.transforms import TranslationTransform2D
from dipy.align.imaffine import AffineRegistration
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

class Video(BaseVideo): 

    file_cls = File 


    def extract_frames(self, frame_range:list[int], output_base: Path, video_path: Path=None):
        if video_path is None:
            video_path = self.path

        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count-1): 
            _, frame = cap.read()

            if i in frame_range: 
                cv2.imwrite(f"{output_base}_{i}.png", frame)

        cap.release()


    def compute_camera_shift(self, ref_path, img_path = None, save_as = None):
        if img_path is None: 
            img_path = self.path
         
        ref_img = imread(str(ref_path))
        ref_img = ref_img[450:, :, :]

        img = imread(str(img_path))
        img = img[450:, :, :]

        # shift calculation 
        
        ref_gray = rgb2gray(ref_img)
        img_gray = rgb2gray(img)

        affreg = AffineRegistration()
        transform = TranslationTransform2D()
        tx = affreg.optimize(ref_gray, img_gray, transform, params0=None)

        shift_matrix = tx.affine
        dx, dy = shift_matrix[0, 2], shift_matrix[1, 2]

        if save_as: 
            
            fig, axes = plt.subplots(2, 2, figsize=(8, 4))

            axes[0, 0].imshow(ref_img, cmap="gray")
            axes[0, 0].set_title("ref img")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(img, cmap="gray")
            axes[0, 1].set_title("img")
            axes[0, 1].axis("off")

            # Red = reference
            # Green = current frame
            stereo = np.zeros((62, 512, 3), dtype=np.uint8)
            stereo[..., 0] = ref_img[..., 0]
            stereo[..., 1] = img[..., 1]

            axes[1, 0].imshow(stereo)
            axes[1, 0].set_title("superimposed")
            axes[1, 0].axis("off")

            shifted_img = tx.transform(img_gray)   # or apply dx,dy to the color img yourself
            
            stereo2 = np.zeros((62, 512, 3), dtype=np.uint8)
            stereo2[..., 0] = ref_img[..., 0]
            stereo2[..., 1] = img_as_ubyte(np.clip(shifted_img, 0, 1))
    
    
            axes[1, 1].imshow(stereo2)
            axes[1, 1].set_title(f"shifted (dx={dx:.1f}, dy={dy:.1f})")
            axes[1, 1].axis("off")
    
            fig.tight_layout()
            fig.savefig(save_as)
            plt.close(fig)

        return dx, dy




    def annotate_video(self, 
                       output_path: Path,
                       bodyparts: list = ["finger_1", "finger_2", "finger_3", "soft_pad"], 
                       video_path: Path=None, 
                       csv_path: Path=None, 
                       radius=5, 
                       likelihood_threshold=0.8, 
                       skeleton_path=None):
        """
        Annotate a video with pose estimation data stored in csv.

        This function overlays colored dots on each frame to represent
        detected body part positions. One color is assigned per body part.
        Annotations are applied only when the likelihood exceeds a given threshold.

        Parameters
        ----------
        video_path : pathlib.Path
            Path to the input video file.
        csv_path : pathlib.Path
            csv file where data has a multi-index header with:
            ``(frame_num, bodyparts, coords)``,
            where ``coords`` includes ``x``, ``y``, and ``likelihood``.
        output_path : pathlib.Path
            Path where the annotated video will be saved.
        radius : int, optional
            Radius (in pixels) of the circles drawn for each body part.
            Default is 5.
        likelihood_threshold : float, optional
            Minimum likelihood required to draw a body part.
            Default is 0.5.

        Returns
        -------
        None
        """
        if video_path is None: 
            video_path = self.path
        if csv_path is None: 
            csv_path = video_path.parent / f"pred_results_{video_path.stem}.csv"

        import matplotlib
        import numba
        from skimage.draw import line_aa

        def circle_offsets(radius):
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            circle = x**2 + y**2 <= radius**2

            ys, xs = np.where(circle)  # matching shapes (N,)
            ys = ys - radius           # convert grid index back to coordinates
            xs = xs - radius

            return np.column_stack((xs, ys))

        # Fast stamping
        @numba.njit
        def stamp_circles(frame, xs, ys, ps, coords_list, colors, threshold):
            frame_h, frame_w, _ = frame.shape

            for bp in range(xs.shape[0]):
                if ps[bp] < threshold:
                    continue

                cx = xs[bp]
                cy = ys[bp]

                if cx < 0 or cy < 0:
                    continue

                coords = coords_list[bp]
                color = colors[bp]

                for k in range(coords.shape[0]):
                    xi = cx + coords[k, 0]
                    yi = cy + coords[k, 1]

                    if 0 <= xi < frame_w and 0 <= yi < frame_h:
                        frame[yi, xi, 0] = color[0]     # red
                        frame[yi, xi, 1] = color[1]     # green
                        frame[yi, xi, 2] = color[2]     # blue

        def stamp_skeleton(frame, xs, ys, ps, skeleton, bodypart_to_idx, threshold):
            for bp1, bp2 in skeleton:

                i1 = bodypart_to_idx[bp1]
                i2 = bodypart_to_idx[bp2]

                # Check likelihood
                if ps[i1] < threshold or ps[i2] < threshold:
                    continue

                x1, y1 = xs[i1], ys[i1]
                x2, y2 = xs[i2], ys[i2]

                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue

                # Draw line (BGR: black)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)


        # Load CSV
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        
        # Drop scorer level to get only (bodypart, coord)
        df.columns = df.columns.droplevel(0)
        df = df.iloc[1:].reset_index(drop=True)

        num_bodyparts = len(bodyparts)
        num_frames = len(df)

        # Extract arrays: (frames, bodyparts)
        x = np.stack([df[bp]["x"].to_numpy() for bp in bodyparts], axis=1).astype(int)
        y = np.stack([df[bp]["y"].to_numpy() for bp in bodyparts], axis=1).astype(int)
        p = np.stack([df[bp]["likelihood"].to_numpy() for bp in bodyparts], axis=1)

        # Replace NaNs with off-screen values
        x[np.isnan(x)] = -radius - 1
        y[np.isnan(y)] = -radius - 1

        # Video IO
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )

        # Colors per bodypart
        # cmap = cm.get_cmap("jet", num_bodyparts)    old version
        cmap = matplotlib.colormaps.get_cmap("jet").resampled(num_bodyparts)
        colors = np.array(
            [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_bodyparts)],
            dtype=np.uint8,
        )

        # Precompute circle offsets
        circle_coords = [circle_offsets(radius) for _ in range(num_bodyparts)]

        # Main loop
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            stamp_circles(frame, x[i], y[i], p[i], 
                        circle_coords, colors, likelihood_threshold)
            
            if skeleton_path :
                with open(skeleton_path, "r") as f: 
                    skeleton = yaml.safe_load(f)
                bodypart_to_idx = {bp: i for i, bp in enumerate(bodyparts)}

                stamp_skeleton(
                    frame, x[i], y[i], p[i],
                    skeleton,
                    bodypart_to_idx,
                    likelihood_threshold
                )

            out.write(frame)

        cap.release()
        out.release()