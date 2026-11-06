# src/Dopa/Leds.py


from src.Base.Leds import Leds as BaseLed

import pandas as pd
from pathlib import Path
import operator


class Leds(BaseLed): 

    def __init__(self, video_path, label_studio_annotation, fig_output_path = None, csv_output_path = None):
        super().__init__(video_path, label_studio_annotation, fig_output_path, csv_output_path)

        self.task = self.define_task()

        # cue S1: Left paw : LED 3
        # cue S2: Right paw: LED 2
        self.task_pad = "LED_3" if self.task == "CueS1" else "LED_2"

        self.time_pad_off = self.get_time_led_state(self.task_pad, "OFF", min_duration=10,  in_sec=True)
        self.time_laser_on = self.get_time_led_state("LED_4", "ON", in_sec=True)
        self.time_reward = self.get_time_led_state( "LED_5", "ON", in_sec=True)
        self.laser_state = "LaserOn" if self.time_laser_on is not None else "LaserOff"


    def define_task(self, threshold=100, min_duration=5, long_duration=30) -> str:
        """
        Determine the cue type based on LED luminosity over time.
        Note: Must be applied only on LED_1.

        Cue classification:
        - 1 short activation  -> ``CueS1``
        - 2 short activations -> ``CueS2``
        - 1 long activation   -> ``CueNoGo`` (300ms = ~37 frames at 125 fps)
        - Otherwise           -> ``NoCue``

        Parameters
        ----------
        threshold : int or float, default=100
            Luminosity threshold used to detect LED activation.

        min_duration : int, default=5
            Minimum number of consecutive frames above the threshold
            required to consider an LED activation valid.

        long_duration : int, default=30
            Minimum number of consecutive frames above the threshold
            required to classify an activation as a long activation (CueNoGo).

        Returns
        -------
        str
            Detected task: ``"CueS1"``, ``"CueS2"``, ``"CueNoGo"`` or ``"NoCue"``.
        """

        activation_durations = []
        time = 0

        for luminosity in self.luminosities["LED_1"]:
            luminosity = float(luminosity)

            if luminosity >= threshold:
                time += 1

            elif time > min_duration:
                activation_durations.append(time)
                time = 0

            else:
                time = 0

        # Account for an activation that extends to the last frame
        if time > min_duration:
            activation_durations.append(time)

        cue_count = len(activation_durations)

        if cue_count == 1:
            if activation_durations[0] >= long_duration:
                cue_type = "CueNoGo"
            else:
                cue_type = "CueS1"

        elif cue_count >= 2:
            cue_type = "CueS2"

        else:
            cue_type = "NoCue"

        return cue_type



    @staticmethod
    def led_state(luminosities: pd.DataFrame,
                threshold: float = 100,
                min_duration: int = 10,
                comparator: operator = operator.lt,) -> tuple[bool, int]:
        
        consecutive = 0
        start_index = None

        for t, value in enumerate(luminosities):
            value = float(value)

            if comparator(value, threshold):
                if consecutive == 0:
                    start_index = t
                consecutive += 1

                if consecutive > min_duration:
                    return True, start_index
            else:
                consecutive = 0
                start_index = None

        return False, None



    def get_time_led_state(self,
                        LED: str = "LED_3", 
                        state: str = "ON",
                        min_duration: int = 5,
                        in_sec: bool = False, 
                        fps: int =125) -> float | int :

        if state == "ON" : 
            _, first_frame = self.led_state(self.luminosities[LED], min_duration=min_duration, comparator=operator.gt)  # gt: greater than = ON
        else : 
            _, first_frame = self.led_state(self.luminosities[LED], min_duration=min_duration, comparator=operator.lt)  # lt: less than = OFF
        
        if first_frame and in_sec : 
            first_time = first_frame / fps  
            # print(f"Laser one at {time_laser_off} sec, {frame_laser_off} frame")
        else : 
            first_time = first_frame
        
        return first_time





