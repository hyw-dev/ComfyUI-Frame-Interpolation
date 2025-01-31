from .sepconv_enhanced import Network
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release, non_timestep_inference, preprocess_frames, postprocess_frames
import typing
from comfy.model_management import soft_empty_cache, get_torch_device
from utils import InterpolationStateList, generic_frame_loop

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["sepconv.pth"]


class SepconvVFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000})
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
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = Network()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())

        frames = preprocess_frames(frames, get_torch_device())
        # Ensure proper tensor dimensions
        frames = [frame.unsqueeze(0) for frame in frames]
        
        def return_middle_frame(frame_0, frame_1, int_timestep, model):
            return model(frame_0, frame_1, int_timestep)
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states)
        )
        return (out,)

        """ frame_dict = {
            str(i): frames[i].unsqueeze(0) for i in range(frames.shape[0])
        }

        if optional_interpolation_states is None:
            interpolation_states = [True] * (frames.shape[0] - 1)
        else:
            interpolation_states = optional_interpolation_states

        enabled_former_idxs = [i for i, state in enumerate(interpolation_states) if state]
        former_idxs_loader = DataLoader(enabled_former_idxs, batch_size=batch_size)

        for former_idxs_batch in former_idxs_loader:
            middle_frame_batches = non_timestep_inference(model, frames[former_idxs_batch], frames[former_idxs_batch + 1], multipler)
            for middle_i in range(1, multipler):
                _middle_frames = middle_frame_batches[middle_i - 1]
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        out_frames = torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
        return (postprocess_frames(out_frames), ) """
