from .IFUNet_arch import IFUNetModel
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
import typing
from comfy.model_management import soft_empty_cache, get_torch_device

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["IFUNet.pth"]

class IFUnet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
                "ensemble": ("BOOLEAN", {"default":True})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        clear_cache_after_n_frames: typing.SupportsInt = 1,
        multiplier: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        ensemble: bool = True,
        optional_interpolation_states: InterpolationStateList = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IFUNetModel()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())

        frames = preprocess_frames(frames, get_torch_device())
        
        # Ensure proper tensor dimensions
        frames = [frame.unsqueeze(0) for frame in frames]
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor, ensemble):
            return model(frame_0, frame_1, timestep=timestep, scale=scale_factor, ensemble=ensemble)
        
        args = [interpolation_model, scale_factor, ensemble]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states)
        )
        return (out,)
        
