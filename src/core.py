import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
from pathlib import Path
import argparse
import os
import time

from utility import image_loader, calc_content_loss, calc_style_loss, calculate_loss, VGG

# calculate_loss usa alpha/beta globali → li settiamo da CLI
alpha = 1.0
beta  = 0.05

ROOT = Path(__file__).resolve().parent.parent
DEF_CONTENT = ROOT / "data" / "Themothee.png"
DEF_STYLE   = ROOT / "data" / "Notte stellata.jpg"
DEF_OUTPUT  = ROOT / "results" / "out.png"

def main():
    global alpha, beta

    p = argparse.ArgumentParser(description="Neural Style Transfer (usa le tue funzioni)")
    p.add_argument("--content_path", default=str(DEF_CONTENT))
    p.add_argument("--style_path",   default=str(DEF_STYLE))
    p.add_argument("--output_path",  default=str(DEF_OUTPUT))
    p.add_argument("--imsize",       type=int,   default=256)
    p.add_argument("--steps",        type=int,   default=200)
    p.add_argument("--lr",           type=float, default=0.02 )
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--beta",         type=float, default=50000)
    p.add_argument("--init", choices=["content","noise"], default="content")
    args = p.parse_args()

    alpha = args.alpha
    beta  = args.beta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Carica immagini (utility.image_loader usa un proprio device; le riallineo comunque)
    content = image_loader(args.content_path).to(device)
    style   = image_loader(args.style_path).to(device)

    # Feature extractor
    model = VGG().to(device).eval()

    # Immagine generata
    generated_image = (content.clone() if args.init == "content" else torch.randn_like(content)).requires_grad_(True)

    # Optimizer sui pixel
    optimizer = optim.Adam([generated_image], lr=args.lr)

    # Feature fisse (content/style)
    with torch.no_grad():
        content_features = model(content)
        style_features   = model(style)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    print(f"Device: {device} (CUDA={torch.cuda.is_available()})")
    start = time.time()

    for e in range(args.steps):
        gen_features = model(generated_image)
        total_loss = calculate_loss(gen_features, content_features, style_features,alpha,beta)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if e % 10 == 0:
            loss_val = total_loss.detach().item()          # evita il warning del cast
            grad_norm = generated_image.grad.detach().abs().mean().item()
            elapsed = time.time() - start
            print(f"[{e:5d}/{args.steps}] loss={loss_val:.6f}  grad={grad_norm:.3e}  t={elapsed:.1f}s", flush=True)
            save_image(generated_image, args.output_path)  # checkpoint

    # Salvataggio finale
    save_image(generated_image, args.output_path)
    print(f"Saved -> {args.output_path}")

if __name__ == "__main__":
    main()
