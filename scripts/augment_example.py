"""Identity augmentation of one right-eye crop with FLUX and SAM 3 (Sec. 3.4, Supp. C).

    python scripts/augment_example.py --image eye_crop.png --out augmented.png

Needs diffusers>=0.36, transformers>=5.0, accelerate, sentencepiece, protobuf, access to
black-forest-labs/FLUX.1-dev and a GPU with about 40 GB of memory.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

TEMPLATE = "raw photo, close-up of {who} face around the eye, {variation}, shot on DSLR"
ORIENTATION_SUFFIX = ", inner eye corner and nose bridge on the right side, eyebrow sweeping from right to left"
DEFAULT_PROMPT = TEMPLATE.format(
    who="a person's",
    variation="freckled fair skin on forehead and cheek, sparse light brow, outdoor natural lighting",
) + ORIENTATION_SUFFIX
LANCZOS = Image.Resampling.LANCZOS


def segment_eye(sam, processor, image, device):
    """SAM 3 eye mask (uint8 0/1) at the image resolution, smallest instance; None if no eye is found."""
    inputs = processor(images=image, text=["eye"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam(**inputs)
    masks = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.7, target_sizes=[[image.height, image.width]])[0]["masks"]
    if masks.shape[0] == 0:
        return None
    return masks[masks.sum(dim=(1, 2)).argmin()].cpu().numpy().astype(np.uint8)


def canny_control(image, eye_mask):
    """ControlNet edge map: high-threshold Canny of the CLAHE-equalized crop, plus low-threshold
    edges inside the 15x15-dilated eye mask; 2x2 dilation, components below 30 px removed."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    edges = (cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 200, 255)
             | cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 2), 220, 255))
    eye_edges = cv2.Canny(gray, 10, 40) | cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 50)
    near_eye = cv2.dilate(eye_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    edges = edges | np.where(near_eye > 0, eye_edges, 0).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 30
    keep[0] = False
    return Image.fromarray(np.where(keep[labels], 255, 0).astype(np.uint8)).convert("RGB")


def clip_score(model, processor, image, prompt, device):
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", truncation=True, max_length=77)
    with torch.no_grad():
        return model(**inputs.to(device)).logits_per_image.item()


def eye_consistent(gen_mask, orig_mask, tolerance):
    """(passed, generated eye area): relative area change <= tolerance and IoU >= 1 - tolerance."""
    if gen_mask is None:
        return False, 0
    gen_area, orig_area = int(gen_mask.sum()), int(orig_mask.sum())
    if orig_area == 0:
        return gen_area > 0, gen_area
    if abs(gen_area - orig_area) / orig_area > tolerance:
        return False, gen_area
    a, b = orig_mask.astype(bool), gen_mask.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1) >= 1.0 - tolerance, gen_area


def load_models(device="cuda", flux_model="black-forest-labs/FLUX.1-dev",
                controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny", sam_model="facebook/sam3",
                clip_model="openai/clip-vit-large-patch14"):
    """SAM 3, FLUX.1-dev with the Canny ControlNet, and CLIP ViT-L/14, as a dict for ``augment``."""
    from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel
    from transformers import CLIPModel, CLIPProcessor, Sam3Model, Sam3Processor

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    return {
        "device": device,
        "sam": Sam3Model.from_pretrained(sam_model).to(device),
        "sam_processor": Sam3Processor.from_pretrained(sam_model),
        "flux": FluxControlNetInpaintPipeline.from_pretrained(
            flux_model, controlnet=controlnet, torch_dtype=torch.bfloat16).to(device),
        "clip": CLIPModel.from_pretrained(clip_model).to(device),
        "clip_processor": CLIPProcessor.from_pretrained(clip_model),
    }


def prepare(models, image, resolution=768):
    """Resized crop ``small``, its SAM 3 ``eye`` mask, the FLUX inpainting mask (everything but the eye) and
    the Canny ``control`` image."""
    small = image.resize((resolution, resolution), LANCZOS)
    eye = segment_eye(models["sam"], models["sam_processor"], small, models["device"])
    if eye is None or eye.sum() == 0:  # an empty mask would let FLUX repaint the eye and change the gaze
        raise ValueError("SAM 3 found no eye (the input must be a raw eye crop without the aperture)")
    return {"small": small, "eye": eye, "inpaint_mask": Image.fromarray(((1 - eye) * 255).astype(np.uint8)).convert("L"),
            "control": canny_control(small, eye)}


def generate(models, prep, prompt, seed=0, attempts=3, steps=40, guidance=4.5, strength=0.97, controlnet_scale=0.7,
             clip_threshold=23.0, eye_tolerance=0.15):
    """FLUX generations with seeds ``seed + 666 * attempt`` until one passes the CLIP and eye checks."""
    device, small, eye = models["device"], prep["small"], prep["eye"]
    for attempt in range(attempts):
        attempt_seed = seed + 666 * attempt
        generated = models["flux"](
            prompt=prompt, image=small, mask_image=prep["inpaint_mask"], control_image=prep["control"],
            controlnet_conditioning_scale=controlnet_scale, strength=strength,
            height=small.height, width=small.width, num_inference_steps=steps,
            guidance_scale=guidance, generator=torch.Generator().manual_seed(attempt_seed),
        ).images[0]
        score = clip_score(models["clip"], models["clip_processor"], generated, prompt, device)
        eye_ok, gen_area = eye_consistent(segment_eye(models["sam"], models["sam_processor"], generated, device), eye,
                                          eye_tolerance)
        accepted = score >= clip_threshold and eye_ok
        print(f"seed {attempt_seed}: CLIP score {score:.2f}, eye area {int(eye.sum())} -> {gen_area} px, "
              f"{'accepted' if accepted else 'rejected'}")
        if accepted:
            break
    return {**prep, "generated": generated, "seed": attempt_seed, "clip_score": score, "accepted": accepted,
            "eye_area_gen": gen_area}


def augment(models, image, prompt=DEFAULT_PROMPT, seed=0, attempts=3, resolution=768, steps=40, guidance=4.5,
            strength=0.97, controlnet_scale=0.7, clip_threshold=23.0, eye_tolerance=0.15):
    """``prepare`` + ``generate`` for one crop; ``result`` is the generation at the crop size."""
    out = generate(models, prepare(models, image, resolution), prompt, seed, attempts, steps, guidance, strength,
                   controlnet_scale, clip_threshold, eye_tolerance)
    out["result"] = out["generated"].resize(image.size, LANCZOS)
    return out


def strip(out, side=384):
    """input | eye mask | Canny control | generation, from the dict returned by ``augment``."""
    panels = [out["small"], Image.fromarray(out["eye"] * 255).convert("RGB"), out["control"], out["generated"]]
    image = Image.new("RGB", (4 * side, side))
    for k, panel in enumerate(panels):
        image.paste(panel.resize((side, side), LANCZOS), (k * side, 0))
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="eye crop (right-eye oriented)")
    ap.add_argument("--out", default="augmented.png")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--strength", type=float, default=0.97)
    ap.add_argument("--controlnet-scale", type=float, default=0.7)
    ap.add_argument("--clip-threshold", type=float, default=23.0)
    ap.add_argument("--eye-tolerance", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flux-model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--controlnet-model", default="InstantX/FLUX.1-dev-Controlnet-Canny")
    ap.add_argument("--sam-model", default="facebook/sam3")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    models = load_models(args.device, args.flux_model, args.controlnet_model, args.sam_model, args.clip_model)
    try:
        out = augment(models, Image.open(args.image).convert("RGB"), args.prompt, args.seed, args.attempts,
                      args.resolution, args.steps, args.guidance, args.strength, args.controlnet_scale,
                      args.clip_threshold, args.eye_tolerance)
    except ValueError as e:
        sys.exit(f"{args.image}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out["result"].save(args.out)
    stem, _ = os.path.splitext(args.out)
    strip(out, args.resolution // 2).save(stem + "_strip.png")
    print(f"saved {args.out} and {stem}_strip.png")


if __name__ == "__main__":
    main()
