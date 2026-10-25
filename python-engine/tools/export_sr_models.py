"""Build models/sr/*.onnx for enhance.py from the official Real-ESRGAN weights.

    python tools/export_sr_models.py <folder holding the .pth files>

The weights are the two compact (SRVGGNetCompact) models from the Real-ESRGAN
v0.2.5.0 release, BSD-3-Clause, by Xintao Wang et al.:

    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

The SHA-256 of each download is checked before it is loaded, and it is loaded
with ``weights_only=True`` - unpickling the file never executes code.

Needs torch and onnx, which the service itself does NOT: the service runs the
exported graphs with OpenCV's own dnn module. Run this from a throwaway venv:

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "sr"

# name -> (weights file, sha256, conv blocks). Both are 64 features wide and 4x.
MODELS = {
    "realesr-animevideov3": (
        "realesr-animevideov3.pth",
        "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        16,
    ),
    "realesr-general-x4v3": (
        "realesr-general-x4v3.pth",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        32,
    ),
}


class SRVGGNetCompact(nn.Module):
    """Same layer layout as realesrgan.archs.srvgg_arch, so the state dict loads as-is."""

    def __init__(self, num_conv: int, num_feat: int = 64, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main(weights_dir: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (filename, sha256, num_conv) in MODELS.items():
        path = weights_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise SystemExit(f"{filename}: sha256 {digest} does not match the release")

        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        model = SRVGGNetCompact(num_conv)
        model.load_state_dict(state, strict=True)
        model.eval()

        out = OUT_DIR / f"{name}.onnx"
        torch.onnx.export(
            model,
            torch.rand(1, 3, 64, 64),
            str(out),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h4", 3: "w4"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
