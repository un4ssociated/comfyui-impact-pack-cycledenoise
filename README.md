# Impact Pack — FaceDetailer Cycle Denoise

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that extends
[ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s `FaceDetailer` with
two things upstream doesn't have:

1. **A separate denoise value for every cycle pass** — instead of one `denoise` for all
   `cycle` iterations, you set `denoise_pass_1` … `denoise_pass_10` individually.
2. **Optional colour matching** — the regenerated face is matched back to the original crop
   before it is pasted, removing the brightness/colour "seam" that appears at high denoise.

Everything else — detection, crop, feathering, SAM refinement, the `detailer_hook` input, the
`DETAILER_PIPE` output — behaves exactly like the stock `FaceDetailer` node.

`FaceDetailer (cycle denoise)` appears under **ImpactPack/Simple**.

---

## Why use it

Stock `FaceDetailer` runs every cycle at one fixed denoise. That is a compromise: a value high
enough to fix a badly generated face also destroys likeness and skin texture, while a value low
enough to preserve likeness can't repair a broken face.

Splitting the value per pass lets you do things like:

| Goal | Pass 1 | Pass 2+ |
|---|---|---|
| Repair a broken face, then gently polish | `0.65` | `0.25` |
| Disagreement recovery (nuke it, then rebuild) | `0.85` | `0.20` |
| Light touch only | `0.35` | `0.35` |
| Progressive refinement | `0.50` | `0.35`, `0.20` |

Cycle passes are **cumulative**: pass 2 starts from the latent that pass 1 produced, not from the
original. This is upstream Impact Pack behaviour and is unchanged here.

---

## Requirements

| | |
|---|---|
| ComfyUI | Any recent version |
| **ComfyUI Impact Pack** | **Required.** Installed and loadable. |
| Tested against | Impact Pack **8.30** |
| Python deps | None beyond what ComfyUI and Impact Pack already provide |

This node imports Impact Pack's internals (`modules.impact.hooks`, `modules.impact.impact_pack`).
It is a companion to Impact Pack, not a replacement — Impact Pack must be installed.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/comfyui-impact-pack-cycledenoise.git
```

Restart ComfyUI **completely** (not just a queue refresh), then look for
`FaceDetailer (cycle denoise)` under `ImpactPack/Simple`.

If the folder is missing entirely from the node list, check the ComfyUI console for:

```
[FaceDetailerCycleDenoise] Cannot import Impact Pack.
```

That means Impact Pack is not installed, not loading, or sits outside `custom_nodes`.

## Uninstall

Delete the `comfyui-impact-pack-cycledenoise` folder and restart ComfyUI.

---

## Node reference

### Inputs

All inputs are identical to the stock `FaceDetailer`, **except**:

**Removed**

- `denoise` — replaced by the ten per-pass values below.

**Added**

| Input | Type | Default | Notes |
|---|---|---|---|
| `denoise_pass_1` … `denoise_pass_10` | FLOAT | `0.5` | Denoise for cycle pass 1–10. Absolute values, not deltas. Min `0.0001`, max `1.0`. |
| `color_match_enabled` | BOOLEAN | `true` | Master switch for colour matching. |
| `color_match_mode` | COMBO | `mean_std` | See modes below. |
| `color_match_strength` | FLOAT | `1.0` | `0.0` = no correction, `1.0` = full correction. |

Passes beyond the node's `cycle` value are ignored — with `cycle = 2` only
`denoise_pass_1` and `denoise_pass_2` are read.

### Outputs

Unchanged from stock `FaceDetailer`:

`image`, `cropped_refined`, `cropped_enhanced_alpha`, `mask`, `detailer_pipe`, `cnet_images`

---

## Colour matching

Regenerated faces often come back slightly brighter, warmer, or flatter than the surrounding
scene, which leaves a visible rectangle where the crop was pasted back. Colour matching takes the
upscaled **source** crop as a reference, re-maps the regenerated crop to match its colour
statistics, and only then pastes it.

The reference is captured once, before sampling; the correction is applied once, after the final
cycle. Intermediate passes are never touched.

### Modes

| Mode | What it matches | Use when |
|---|---|---|
| `mean_std` | Per-channel RGB mean and standard deviation (Reinhard). | **Default.** Fixes brightness, contrast and colour cast. Fast. Can shift hue slightly. |
| `mean_std_luma` | Luminance mean and standard deviation only, applied equally to R/G/B. | Fixes brightness and contrast while **preserving** the generated hue and saturation. Good for already-correct colour. |
| `histogram` | Per-channel full distribution (CDF match). | Strongest tonal match. Can flatten regenerated texture and shift hue. Slowest — involves a per-channel sort. |
| `lab_reinhard` | Mean and standard deviation of L, a, b in CIE Lab. | Perceptually better brightness + colour match than RGB. Good general-purpose step up from `mean_std`. |
| `lab_hist_luma_chroma_meanstd` | Histogram match of L, mean/std match of a and b. | Strongest mode that still preserves hue. Try this if others leave a visible seam. |

Modes marked Lab require 3-channel RGB input. Anything else (grayscale, RGBA) silently falls back
to `mean_std`.

### Strength

The correction is blended with `out = target + strength × (matched − target)` and then clamped to
`[0, 1]`. Lower values keep more of the generated face's original appearance; the clamp can clip
highlights and shadows, so if a face looks flat or blown out, drop the strength before changing
modes.

### Cost

`histogram` and `lab_hist_luma_chroma_meanstd` sort every pixel of the crop, per channel. On
1024 px+ crops that is measurable, roughly doubling the cost of the detail pass. The other three
modes are near-free.

---

## Interaction with other detailer hooks

If you connect a `detailer_hook`, the chain is applied in this order:

```
your hook  →  per-cycle denoise  →  colour match
```

Because hooks are combined in sequence, **this node's `denoise_pass_*` values override any
upstream hook that also sets denoise**, such as `DetailerDenoiseSchedulerHook`,
`SimpleDenoiseScheduleHook`, or `SimpleDetailerDenoiseSchedulerHook`. Combining them is not an
error, but the schedule from the other hook will have no effect.

Your hook's other behaviour (custom sampler, noise, SEGS filtering, `post_paste`) is unaffected.

---

## How it works

Two small `DetailerHook` subclasses, chained with `hooks.DetailerHookCombine`:

- `CycleDenoiseHook.pre_ksample` — Impact Pack calls `set_steps((i, cycle))` before each pass,
  then `pre_ksample`. The hook reads `self.cur_step` as the cycle index and returns
  `denoise_pass_1..10[cur_step]` in place of the inherited denoise.
- `ColorMatchHook.post_upscale` — clones the upscaled source crop as the reference.
  `ColorMatchHook.post_decode` — resizes the reference if needed and re-maps the decoded result.

Colour maths is implemented in plain torch: Reinhard transfer, inverse-CDF histogram matching, and
sRGB ↔ CIE Lab conversion with a D65 white point.

---

## Compatibility

`doit()` mirrors the argument list of `FaceDetailer` in Impact Pack 8.30, including newer kwargs
(`tiled_encode`, `tiled_decode`, `scheduler_func_opt`). Two consequences:

- If Impact Pack **adds** a required input to `FaceDetailer`, this node's `INPUT_TYPES()` will
  inherit it but `doit()` will reject it with a `TypeError`. Updating this file is required.
- If Impact Pack **removes or renames** one of those kwargs, the call to `enhance_face` breaks.

If Impact Pack changes and this node stops loading, pin Impact Pack to a known-good version rather
than editing the installed pack.

## Limitations

- **Not for video.** Same as upstream `FaceDetailer` — a warning is logged and each frame is
  processed independently, ignoring temporal consistency. Use `Detailer For AnimateDiff`.
- Colour matching uses **global per-crop statistics**. It cannot fix a face whose interior lighting
  legitimately differs from the surrounding scene; lower the strength instead.
- Matching applies to the whole crop, not a masked region.
- `histogram` modes assume the crop's tonal distribution should equal the source crop's. If the
  regenerated image legitimately has a very different tonal range (strong rim light, heavy shadow),
  it will be mapped toward the original — which may be the opposite of what you want.

## License

GPL-3.0. This node subclasses and imports code from ComfyUI Impact Pack, which is GPL-3.0, so the
combined work is GPL-3.0. See [LICENSE](LICENSE).

## Credits

- [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) by ltdrdata — the
  `FaceDetailer` implementation and hook API this node builds on.
- Colour transfer based on Reinhard et al., *Color Transfer between Images* (2001).
