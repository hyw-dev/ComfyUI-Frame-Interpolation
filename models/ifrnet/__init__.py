import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release, preprocess_frames, postprocess_frames
import typing
from .IFRNet_S_arch import IRFNet_S
from .IFRNet_L_arch import IRFNet_L
from comfy.model_management import soft_empty_cache, get_torch_device
from utils import generic_frame_loop, InterpolationStateList

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["IFRNet_S_Vimeo90K.pth", "IFRNet_L_Vimeo90K.pth"]

class IFRNet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
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
        optional_interpolation_states: InterpolationStateList = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IRFNet_S() if 'S' in ckpt_name else IRFNet_L()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())

        frames = preprocess_frames(frames, get_torch_device())
            
        # Ensure proper tensor dimensions
        frames = [frame.unsqueeze(0) for frame in frames]
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor):
            return model(frame_0, frame_1, timestep, scale_factor)
        
        args = [interpolation_model, scale_factor]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states)
        )
        return (out,)
