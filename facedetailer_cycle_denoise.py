"""FaceDetailer variant with a per-cycle denoise value and optional colour matching.

The node subclasses Impact Pack's ``FaceDetailer`` and drives it through two
``DetailerHook`` subclasses:

* ``CycleDenoiseHook`` replaces the single ``denoise`` input with one value per
  cycle pass.
* ``ColorMatchHook`` re-maps the regenerated crop onto the colour statistics of
  the original upscaled crop before it is pasted back, removing the seam.

Depends on ComfyUI-Impact-Pack. See README.md for the compatibility contract.
"""

import logging

import torch

from modules.impact import hooks
from modules.impact.impact_pack import FaceDetailer


class CycleDenoiseHook(hooks.DetailerHook):
    def __init__(self, denoises):
        super().__init__()
        self.denoises = denoises

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        idx = min(self.cur_step, len(self.denoises) - 1)
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, self.denoises[idx]


def _channel_stats(x):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
    return mean, std


def _match_mean_std(target, ref):
    t_mean, t_std = _channel_stats(target)
    r_mean, r_std = _channel_stats(ref)
    t_std = t_std.clamp_min(1e-6)
    return (target - t_mean) / t_std * r_std + r_mean


def _luma(x):
    return 0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3]


def _match_mean_std_luma(target, ref):
    t_l = _luma(target)
    r_l = _luma(ref)
    t_mean = t_l.mean(dim=(1, 2), keepdim=True)
    r_mean = r_l.mean(dim=(1, 2), keepdim=True)
    t_std = t_l.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    r_std = r_l.std(dim=(1, 2), keepdim=True, unbiased=False)
    return (target - t_mean) / t_std * r_std + r_mean


def _match_histogram(target, ref):
    out = torch.empty_like(target)
    b_size, height, width, channels = target.shape
    for b in range(b_size):
        ref_b = ref[b if ref.shape[0] > 1 else 0]
        for c in range(channels):
            t = target[b, ..., c].reshape(-1)
            r = ref_b[..., c].reshape(-1)
            t_sorted, t_idx = torch.sort(t)
            r_sorted, _ = torch.sort(r)

            n_t = t.numel()
            n_r = r.numel()
            t_pos = (torch.arange(n_t, device=t.device, dtype=torch.float32) + 0.5) / n_t
            r_pos = (torch.arange(n_r, device=r.device, dtype=torch.float32) + 0.5) / n_r
            idx = torch.searchsorted(r_pos, t_pos).clamp_(0, n_r - 1)

            mapped = torch.empty_like(t)
            mapped[t_idx] = r_sorted[idx]
            out[b, ..., c] = mapped.view(height, width)
    return out


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def _rgb_to_lab(rgb):
    linear = _srgb_to_linear(rgb)

    r, g, b = linear[..., 0:1], linear[..., 1:2], linear[..., 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    z = z / 1.08883

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f(t):
        return torch.where(t > delta3, t.clamp_min(0.0) ** (1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([l, a, bb], dim=-1)


def _lab_to_rgb(lab):
    l, a, b = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def finv(t):
        t3 = t ** 3
        return torch.where(t3 > delta3, t3, 3.0 * delta ** 2 * (t - 4.0 / 29.0))

    x = finv(fx) * 0.95047
    y = finv(fy)
    z = finv(fz) * 1.08883

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b2 = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    linear = torch.cat([r, g, b2], dim=-1).clamp(0.0, 1.0)
    return _linear_to_srgb(linear)


def _match_lab(target, ref, l_hist=False):
    t_lab = _rgb_to_lab(target)
    r_lab = _rgb_to_lab(ref)

    if l_hist:
        l_out = _match_histogram(t_lab[..., 0:1], r_lab[..., 0:1])
    else:
        l_out = _match_mean_std(t_lab[..., 0:1], r_lab[..., 0:1])

    ab_out = _match_mean_std(t_lab[..., 1:3], r_lab[..., 1:3])

    return _lab_to_rgb(torch.cat([l_out, ab_out], dim=-1))


def _resize_ref(ref, shape):
    if tuple(ref.shape[1:3]) == tuple(shape[1:3]):
        return ref
    ref_nchw = ref.permute(0, 3, 1, 2)
    ref_nchw = torch.nn.functional.interpolate(ref_nchw, size=tuple(shape[1:3]), mode="bilinear", align_corners=False)
    return ref_nchw.permute(0, 2, 3, 1)


def _apply_color_match(target, ref, mode, strength):
    is_rgb = target.shape[-1] == 3 and ref.shape[-1] == 3

    if mode in ("lab_reinhard", "lab_hist_luma_chroma_meanstd") and not is_rgb:
        mode = "mean_std"

    if mode == "mean_std":
        matched = _match_mean_std(target, ref)
    elif mode == "mean_std_luma":
        matched = _match_mean_std_luma(target, ref)
    elif mode == "histogram":
        matched = _match_histogram(target, ref)
    elif mode == "lab_reinhard":
        matched = _match_lab(target, ref, l_hist=False)
    elif mode == "lab_hist_luma_chroma_meanstd":
        matched = _match_lab(target, ref, l_hist=True)
    else:
        return target

    if strength < 1.0:
        matched = target + strength * (matched - target)

    return matched.clamp(0.0, 1.0)


class ColorMatchHook(hooks.DetailerHook):
    def __init__(self, mode="mean_std", strength=1.0):
        super().__init__()
        self.mode = mode
        self.strength = strength
        self.ref = None

    def post_upscale(self, pixels, mask=None):
        self.ref = pixels.detach().clone()
        return pixels

    def post_decode(self, pixels):
        if self.ref is None or self.mode == "disabled" or self.strength <= 0.0:
            return pixels

        ref = _resize_ref(self.ref, pixels.shape).to(pixels.device)
        matched = _apply_color_match(pixels.float(), ref.float(), self.mode, self.strength)
        return matched.to(pixels.dtype)


class FaceDetailerCycleDenoise:
    @classmethod
    def INPUT_TYPES(s):
        types = FaceDetailer.INPUT_TYPES()

        del types["required"]["denoise"]

        for i in range(1, 11):
            types["required"][f"denoise_pass_{i}"] = ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01})

        color_match_tooltip = (
            "Method used to match each regenerated face back to the original crop before pasting (removes the brightness/color seam).\n"
            "- mean_std: Per-channel RGB mean & std (Reinhard). Fixes brightness, contrast and color cast. Fast; can slightly shift hue.\n"
            "- mean_std_luma: Matches luminance mean & std only, applied equally to all channels. Fixes brightness/contrast, preserves generated hue and saturation.\n"
            "- histogram: Per-channel full-distribution (CDF) match. Strongest tonal match; can flatten regenerated texture and shift hue.\n"
            "- lab_reinhard: Mean & std match of L, a, b in Lab space. Perceptually better brightness + color match than RGB.\n"
            "- lab_hist_luma_chroma_meanstd: Histogram-matches L (full tonal curve), mean/std-matches a & b (color). Recommended strongest mode that still preserves hue."
        )

        types["optional"]["color_match_enabled"] = ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled", "tooltip": "Master switch. When enabled, the regenerated face is color/brightness-matched back to the original crop before it is pasted, removing the brightness seam."})
        types["optional"]["color_match_mode"] = (["mean_std", "mean_std_luma", "histogram", "lab_reinhard", "lab_hist_luma_chroma_meanstd"], {"default": "mean_std", "tooltip": color_match_tooltip})
        types["optional"]["color_match_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend between no correction (0.0) and full correction (1.0). Lower values retain more of the generated face's original brightness/color."})

        return types

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Simple"

    DESCRIPTION = "Same as Impact Pack `FaceDetailer`, but lets you set a separate `denoise` value for each cycle pass (denoise_pass_1 .. denoise_pass_10). Passes beyond `cycle` are ignored. When `color_match_enabled` is on, the regenerated face is color/brightness-matched back to the original crop before it is pasted, removing the brightness seam. Modes: `mean_std` (per-channel RGB Reinhard), `mean_std_luma` (luminance only, preserves hue), `histogram` (per-channel full-distribution match), `lab_reinhard` (mean/std match of L, a, b in Lab), `lab_hist_luma_chroma_meanstd` (histogram L + mean/std chroma in Lab). `color_match_strength` blends the correction. Hover the `color_match_mode` widget for a description of each mode."

    def doit(self, image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle=1,
             denoise_pass_1=0.5, denoise_pass_2=0.5, denoise_pass_3=0.5, denoise_pass_4=0.5, denoise_pass_5=0.5,
             denoise_pass_6=0.5, denoise_pass_7=0.5, denoise_pass_8=0.5, denoise_pass_9=0.5, denoise_pass_10=0.5,
             color_match_enabled=True, color_match_mode="mean_std", color_match_strength=1.0,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
             scheduler_func_opt=None, tiled_encode=False, tiled_decode=False):

        denoises = [denoise_pass_1, denoise_pass_2, denoise_pass_3, denoise_pass_4, denoise_pass_5,
                    denoise_pass_6, denoise_pass_7, denoise_pass_8, denoise_pass_9, denoise_pass_10][:cycle]

        denoise_hook = CycleDenoiseHook(denoises)

        hook_chain = []
        if detailer_hook is not None:
            hook_chain.append(detailer_hook)
        hook_chain.append(denoise_hook)
        if color_match_enabled and color_match_mode != "disabled" and color_match_strength > 0.0:
            hook_chain.append(ColorMatchHook(color_match_mode, color_match_strength))

        detailer_hook = hook_chain[0]
        for next_hook in hook_chain[1:]:
            detailer_hook = hooks.DetailerHookCombine(detailer_hook, next_hook)

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []

        if len(image) > 1:
            logging.warning("[Impact Pack] WARN: FaceDetailer is not a node designed for video detailing. If you intend to perform video detailing, please use Detailer For AnimateDiff.")

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = FaceDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg, sampler_name, scheduler,
                positive, negative, denoises[0], feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt,
                tiled_encode=tiled_encode, tiled_decode=tiled_decode)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images