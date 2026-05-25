"""Generate poster-ready QR codes for SemanticVLA's public URLs.

Outputs high-error-correction PNGs at a print-friendly resolution
(~1500x1500 px) suitable for poster printing.

Run:
    python docs/make_qr.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H

URLS = {
    "qr_github": "https://github.com/Fei-Ni/SemanticVLA_Offcial",
    "qr_hf_collection_model_zoo": "https://hf.co/collections/spikefly/semanticvla-model-zoo",
    "qr_hf_collection_datasets": "https://hf.co/collections/spikefly/semanticvla-datasets",
}


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "qr")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        qr = qrcode.QRCode(
            version=None,
            error_correction=ERROR_CORRECT_H,
            box_size=40,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        path = out_dir / f"{name}.png"
        img.save(path)
        print(f"{path}  <-  {url}")


if __name__ == "__main__":
    main()
