"""Phase-0 gate: determine whether Nunchaku/SVDQuant covers FLUX.2-klein-4B (Path B go/no-go).

The prior-art pass found Nunchaku quants only for the non-commercial 9B/9B-kv; the 4B
(the commercial path) is unconfirmed. This probes HF for a 4B INT4/SVDQuant quant. A
'NONE FOUND' result means Path B requires self-quantizing the 4B (or is unavailable), and
the Phase-4 bake-off drops to TensorRT-only — record the call in NUNCHAKU_DECISION.md.
"""
from __future__ import annotations

from huggingface_hub import HfApi


def main() -> None:
    api = HfApi()
    hits = list(api.list_models(search="FLUX.2-klein-4B", limit=100))
    nunchaku = sorted(
        h.id for h in hits
        if "nunchaku" in h.id.lower() or "svdq" in h.id.lower()
    )
    print("Candidate 4B Nunchaku/SVDQuant quants:")
    for h in nunchaku:
        print("  ", h)
    if not nunchaku:
        print("NONE FOUND — Path B requires self-quantizing the 4B or is unavailable.")
    # Also surface non-Nunchaku quants for context (GGUF != Nunchaku kernel path).
    others = sorted(h.id for h in hits if h.id not in nunchaku and "4b" in h.id.lower())
    print("\nOther 4B derivatives (context; GGUF/AWQ are NOT the Nunchaku path):")
    for h in others[:20]:
        print("  ", h)


if __name__ == "__main__":
    main()
