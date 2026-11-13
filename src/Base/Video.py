# src/Base/Video.py

from pathlib import Path
import cv2
from .File import File as BaseFile

class Video: 

    file_cls = BaseFile   # overridden by subclasses

    def __init__(self, video_path: Path):

        self.file = self.file_cls(video_path)
        self.is_openable = True
        self.is_readable = True

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            self.is_openable = False
            print(f"[ERROR] {video_path}: cannot be opened")

        self.date = self.file.date     # now this line is safe, IF file_cls has .date
        self.name = self.file.name
        self.path = self.file.path
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_count < 10: 
            self.is_readable = False
            print(f"[ERROR] {video_path}: is too short ({self.frame_count} frames)")

        cap.release()


    def split_video(self, input_path: Path, output_dir: Path,
                CLIP_DURATION: float = 3, # seconds
                FPS: int = None,
                CRF: int = 13) -> None:
        """
        Split a video into fixed-duration clips using FFmpeg.

        The video is first re-encoded to a constant frame rate and
        compressed using H.264. It is then split into multiple clips
        without re-encoding.
        Requires FFmpeg to be installed and available in PATH.

        Parameters
        ----------
        input_path : pathlib.Path
            Path to the input video file.
        output_path : pathlib.Path
            Directory where output clips will be saved.
            Output clip names follow the pattern: ``{output_path.stem}_clip_XX.mp4``
        CLIP_DURATION : float, optional
            Duration of each clip in seconds. Default is 3.
        FPS : int, optional
            Target frame rate. If None, the original FPS is preserved.
        CRF : int, optional
            Constant Rate Factor for H.264 compression
            (lower = better quality). Default is 13.

        Returns
        -------
        None
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        fixed_video_path = output_dir / "fixed_125fps.mp4"

        print("\nSplitting video in one go ...")

        # ------------------ STEP 1: reinterpret frames as N fps + compression

        print("\nCompressing video ...\n")
        ffmpeg_args = [
            "ffmpeg",
            "-y"]

        if FPS : 
            ffmpeg_args += ["-r", str(FPS)]

        ffmpeg_args += [
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", str(CRF),
            "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            str(fixed_video_path)
        ]
        self._run_ffmpeg(ffmpeg_args)

        # Re-probe FIXED video
        total_duration = self.frame_count / FPS if FPS is not None else self.frame_count / self.fps
        if FPS is None:
            FPS = self.fps

        print(f"\nCRF : {CRF} | Video FPS : {FPS} | Video Duration : {total_duration:.2f} sec")
        print(f"Clip duration : {CLIP_DURATION}  |  Number of output clips : {round(total_duration / CLIP_DURATION)}\n")

        # ------------------ STEP 2: split normaly (NO re-encode)

        start_time = 0.0
        i = 0
        while start_time < total_duration:
            print(f"\n# Clipping video from {start_time:.2f} - {start_time+CLIP_DURATION}, clip N°{i}\n")

            ffmpeg_args = [
                "ffmpeg",
                "-y",

                "-ss", str(start_time),
                "-i", str(fixed_video_path),
                "-t", str(CLIP_DURATION),

                # "-c", "copy",

                str(output_dir / f"{output_dir.stem}_clip_{i:02d}.mp4")
            ]
            self._run_ffmpeg(ffmpeg_args)

            start_time += CLIP_DURATION
            i += 1

        # ------------------ STEP 3: cleanup

        fixed_video_path.unlink()


    @staticmethod
    def _run_ffmpeg(ffmpeg_args: list[str]) -> None : 
        """
        Execute an FFmpeg command.
        Requiere FFmpeg to be installed
        """
        import subprocess

        try:
            subprocess.run(ffmpeg_args, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed:\n{e}")


    
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


    

    def annotate_video(self, 
                       output_path: Path,
                       bodyparts: list = ["finger_1", "finger_2", "finger_3", "soft_pad"], 
                       video_path: Path=None, 
                       csv_path: Path=None, 
                       radius=5, 
                       likelihood_threshold=0, 
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