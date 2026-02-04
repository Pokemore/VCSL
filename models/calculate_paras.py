import torch
from sfd_module import DConv  # path: RDfusionLQVG/models/sfd_module.py
import sys

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def forward_check(atoms, groups=None, device='cpu'):
    # Create module using current sfd_module.DConv implementation.
    # Note: current implementation of DConv in sfd_module.py doesn't accept groups param;
    # this script just measures parameter count for the module as-is.
    m = DConv(in_channels=512, alpha=0.8, atoms=atoms).to(device)
    print(f"atoms={atoms}, params={count_params(m)}")
    x = torch.randn(2, 512, 8, 8).to(device)  # small N=2 spatial demo
    with torch.no_grad():
        out = m(x)
    print("out shape:", out.shape)
    print("out contains NaN?", torch.isnan(out).any().item())
    print("out mean/std:", float(out.mean().cpu()), float(out.std().cpu()))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    for a in [512, 1024]:
        forward_check(a, device=device)