# src/Base/Prediction.py

from pathlib import Path
from .Video import Video
import pandas as pd

class Prediction: 

    def __init__(self, video: Video):
         self.video: Video = video


    def split_video(self, input_path: Path, output_dir: Path,
                CLIP_DURATION: float = 3,
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
        total_duration = self.video.frame_count / FPS if FPS is not None else self.video.frame_count
        if FPS is None : 
            FPS = self.video.fps

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


    @staticmethod
    def dlc_predict(model_path: Path, 
                    video_path: Path, 
                    output_csv_path : Path = None) -> pd.DataFrame:
        import tempfile
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch import set_load_weights_only

        with tempfile.TemporaryDirectory() as dlc_dest:
            # print(dlc_dest)
            deeplabcut.analyze_videos(
                f'{model_path}/config.yaml',
                [str(video_path)],
                save_as_csv=False,
                # gputouse=0,
                destfolder=dlc_dest
            )

            h5_file = next(Path(dlc_dest).glob("*.h5"), None)
            df = pd.read_hdf(h5_file)

        df.index.name="frame_num"

        if output_csv_path : 
            df.to_csv(output_csv_path)

        return df
