import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import torchvision.transforms as T
from picarnet import PiCarNet
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_PATH    = os.path.join(REPO_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "outputs", "models")
HEATMAPS_DIR = os.path.join(REPO_DIR, "heatmaps")
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name"   : "crop_oversample_corners_best_model.pth",
    "img_dir"      : os.path.join(DATA_PATH, "training_images"),
    "n_images"     : 10,
    "skip"         : 0,
    "crop_top"     : 0.30,
    "crop_bottom"  : 0,
    "img_h"        : 120,
    "img_w"        : 160,
    "single_output": True,    # True = angle only, False = angle + speed
}

# ── Transforms ────────────────────────────────────────────────────────────────
def build_transform(crop_top, crop_bottom, img_h, img_w):
    steps = [T.ToPILImage()]
    if crop_top > 0 or crop_bottom > 0:
        steps.append(T.Lambda(lambda img: T.functional.crop(
            img,
            top    = int(img.height * crop_top),
            left   = 0,
            height = int(img.height * (1 - crop_top - crop_bottom)),
            width  = img.width,
        )))
    steps += [
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    return T.Compose(steps)


def crop_np(img_np, crop_top, crop_bottom):
    h      = img_np.shape[0]
    top    = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom)) if crop_bottom > 0 else h
    return img_np[top:bottom, :]

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def get_gradcam(model, image_tensor, output_idx=0):
    model.eval()
    gradients, activations = [], []

    target_layer = model.backbone.features[-1]
    fh = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0]))

    output = model(image_tensor)
    model.zero_grad()
    output[0, output_idx].backward()

    fh.remove()
    bh.remove()

    grads   = gradients[0].squeeze(0)
    acts    = activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    heatmap = F.relu((weights[:, None, None] * acts).sum(dim=0))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.detach().cpu().numpy()


def overlay_gradcam(img_bgr, heatmap):
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

# ── Main ──────────────────────────────────────────────────────────────────────
def run(cfg):
    model_path = os.path.join(MODELS_DIR, cfg["model_name"])
    model = PiCarNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Loaded model: {cfg['model_name']}")

    transform = build_transform(
        cfg["crop_top"], cfg["crop_bottom"], cfg["img_h"], cfg["img_w"]
    )

    img_files = sorted(Path(cfg["img_dir"]).glob("*.png"))
    img_files = img_files[cfg["skip"] : cfg["skip"] + cfg["n_images"]]
    if not img_files:
        print(f"No images found in {cfg['img_dir']}")
        return

    outputs = {"angle": 0}
    if not cfg["single_output"]:
        outputs["speed"] = 1

    tag         = f"crop{cfg['crop_top']}_{cfg['crop_bottom']}"
    run_vis_dir = os.path.join(
        HEATMAPS_DIR, f"{Path(cfg['model_name']).stem}_{tag}"
    )
    os.makedirs(run_vis_dir, exist_ok=True)
    print(f"Saving to: {run_vis_dir}")

    for fpath in img_files:
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            print(f"Skipping unreadable file: {fpath}")
            continue

        img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor       = transform(img_rgb).unsqueeze(0)
        img_rgb_crop = crop_np(img_rgb, cfg["crop_top"], cfg["crop_bottom"])
        img_bgr_crop = crop_np(img_bgr, cfg["crop_top"], cfg["crop_bottom"])

        for label, idx in outputs.items():
            heatmap = get_gradcam(model, tensor, output_idx=idx)
            overlay = overlay_gradcam(img_bgr_crop, heatmap)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(img_rgb_crop)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Grad-CAM ({label})")
            axes[1].axis("off")
            plt.tight_layout()

            save_name = f"gradcam_{label}_{tag}_{fpath.name}"
            plt.savefig(os.path.join(run_vis_dir, save_name), dpi=150)
            plt.close()
            print(f"Saved: {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default=CONFIG["model_name"])
    parser.add_argument("--img_dir",       default=CONFIG["img_dir"])
    parser.add_argument("--n",             type=int,   default=CONFIG["n_images"])
    parser.add_argument("--skip",          type=int,   default=CONFIG["skip"])
    parser.add_argument("--crop_top",      type=float, default=CONFIG["crop_top"])
    parser.add_argument("--crop_bottom",   type=float, default=CONFIG["crop_bottom"])
    parser.add_argument("--single_output", action="store_true", default=CONFIG["single_output"])
    args = parser.parse_args()

    CONFIG.update({
        "model_name"   : args.model,
        "img_dir"      : args.img_dir,
        "n_images"     : args.n,
        "skip"         : args.skip,
        "crop_top"     : args.crop_top,
        "crop_bottom"  : args.crop_bottom,
        "single_output": args.single_output,
    })
    run(CONFIG)