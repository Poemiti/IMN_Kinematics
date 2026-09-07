# src/Base/Leds.py


import numpy as np 
import pandas as pd
from pathlib import Path
import cv2
import xarray as xr
import plotly.express as px
import tqdm



class Leds:

    def __init__(self, 
                 video_path: str, 
                 label_studio_annotation: int,
                 fig_output_path: str = None,
                 csv_output_path: str = None,):

        self.video_path = video_path
        self.annotation_num = label_studio_annotation
        self.fig_output_path = fig_output_path
        self.csv_output_path = csv_output_path

        self.label_studio_url = "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp"
        self.api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"

        self.luminosities = self.get_luminosity()
        self.clean_luminosities()

    
    def get_luminosity(self) -> pd.DataFrame :
        from label_studio_sdk import LabelStudio

        ls_client = LabelStudio(base_url=self.label_studio_url, api_key=self.api_key)
        data = ls_client.annotations.get(id=self.annotation_num).result
        led_info = {}
        for item in data:
            label = item["value"]["ellipselabels"][0]
            led_info[label] = {"x_per": item["value"]["x"], "y_per": item["value"]["y"], "radiusX_per": item["value"]["radiusX"], "radiusY_per": item["value"]["radiusY"]}
        leds = pd.DataFrame(led_info).T.to_xarray().rename(index="led_name")
        # print(leds)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        image = xr.Dataset()
        image["y"] = xr.DataArray(np.arange(h), dims="y")
        image["x"] = xr.DataArray(np.arange(w), dims="x")
        image["mask"] = ((image["x"] - leds["x_per"]*w/100)**2/(leds["radiusX_per"]*w/100)**2 + (image["y"] - leds["y_per"]*h/100)**2/(leds["radiusY_per"]*h/100)**2) < 1
        # print(image)

        if self.fig_output_path:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            fig = px.imshow(frame)
            image["color"] = xr.DataArray(["r", "g", "b", "a"], dims="color")
            image["mask_color"] = xr.DataArray([0, 200, 0, 0.5], dims="color")
            rgba_mask = image["mask"] * image["mask_color"]
            import plotly.graph_objects as go
            for i in range(rgba_mask.sizes["led_name"]):
                fig.add_trace(go.Image(z=rgba_mask.isel(led_name=i).transpose("y", "x", "color"), colormodel="rgba"))
            fig.write_html(self.fig_output_path)

        #Highly optimized code part, we convert everything to basic numpy and list, taking care of ordering
        n_leds = image.sizes["led_name"]
        mask_low_x = image["x"].where(image["mask"].any("y")).min("x").astype(int).to_numpy().tolist()
        mask_high_x = (image["x"].where(image["mask"].any("y")).max("x").astype(int).to_numpy()+1).tolist()
        mask_low_y = image["y"].where(image["mask"].any("x")).min("y").astype(int).to_numpy().tolist()
        mask_high_y = (image["y"].where(image["mask"].any("x")).max("y").astype(int).to_numpy()+1).tolist()
        cropped_masks = [image["mask"].isel(led_name=i).transpose("y", "x").to_numpy()[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]] for i in range(n_leds)]
        mask_low_x, mask_high_x, mask_low_y, mask_high_y

        cap = cv2.VideoCapture(self.video_path)
        luminosities = []

        if max_n_frames is None: 
            max_n_frames = num_frames
        else:
            max_n_frames = min(max_n_frames, num_frames)

        for i in tqdm.tqdm(range(max_n_frames), desc="Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lum = [np.sum(np.where(cropped_masks[i], gray[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]], 0)) for i in range(n_leds)]
            luminosities.append(lum)

        cap.release()
        #End of highly optimized code part

        luminosities = xr.DataArray(luminosities, dims=["t", "led_name"], name="luminosity")
        luminosities["t"] = np.arange(luminosities.sizes["t"])/fps
        luminosities["t"].attrs["fs"] = fps
        luminosities = luminosities/image["mask"].sum(["y", "x"])

        luminosity_df = luminosities.to_dataframe(name="luminosity").unstack("led_name")

        # save as a csv file
        if self.csv_ouput_path is not None : 
            luminosity_df.to_csv(self.csv_ouput_path)

        return luminosity_df

    def clean_luminosities(self) -> pd.DataFrame: 

        # clean luminosities dataframe
        self.luminosities.columns = self.luminosities.columns.droplevel(0)        # columns = LED_1 ...
        self.luminosities = self.luminosities.drop([1]).reset_index(drop=True)    # remove useless row

