from contextlib import ExitStack

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


class HailoInference:
    def __init__(self, hef_path: str):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.input_name = next(iter(self.input_vstreams_params))
        self.input_height, self.input_width = self.hef.get_input_vstream_infos()[0].shape[:2]
        print(f"[Hailo] Model input: {self.input_width}x{self.input_height}")
        # Activate the network group and open the vstream pipeline once, then run()
        # only pushes frames. activate() must be entered before InferVStreams: its
        # __enter__ starts reader threads that need an active group, otherwise reads
        # fail with HAILO_STREAM_NOT_ACTIVATED(72).
        self._stack = ExitStack()
        self._stack.enter_context(self.network_group.activate())
        self._pipeline = self._stack.enter_context(
            InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params))

    def preprocess(self, frame: np.ndarray) -> dict:
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return {self.input_name: np.expand_dims(rgb, axis=0)}

    def run(self, input_data: dict) -> dict:
        return self._pipeline.infer(input_data)

    def release(self):
        self._stack.close()
        self.target.release()
