#!/usr/bin/env python3
"""CPU reference dumps for the native MiniMax Music 3 port (xpu_music3), from ComfyUI's own model code.

Run with the ComfyUI venv, GPUs hidden:
  CUDA_VISIBLE_DEVICES= nice -n 10 ~/ai/ComfyUI/.venv/bin/python ref_music3.py {dit,cond,dav,ar} OUTDIR

Every tensor is written as raw little-endian f32 (or int32 for codes) plus meta.txt ("key value" lines).
  dit   one DiffusionTransformer call (random x, condition, timestep) -> out.f32
  cond  aligned_condition of random AR hidden frames -> cond.f32
  dav   DAV decode of a random latent -> wav.f32
  ar    prefill of a short prompt, then teacher-forced frames (codes sampled here and dumped) -> hidden frames
"""
import os
import sys

sys.argv = [sys.argv[0], "--cpu"] + sys.argv[1:]  # comfy.cli_args reads sys.argv at import
MODE, OUT = sys.argv[2], sys.argv[3]
sys.argv = sys.argv[:2]
COMFY = "/home/horvay/ai/ComfyUI"
sys.path.insert(0, COMFY)

import numpy as np
import torch

import comfy.options

comfy.options.enable_args_parsing()

torch.set_num_threads(8)
torch.manual_seed(0)

import comfy.model_management as mm
import comfy.ops
import comfy.utils

MODELS = os.path.join(COMFY, "models")
TE = os.path.join(MODELS, "text_encoders", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors")
DIT = os.path.join(MODELS, "diffusion_models", "minimax_music3_dit_int8_convrot.safetensors")
DAV = os.path.join(MODELS, "vae", "minimax_music3_dav.safetensors")

os.makedirs(OUT, exist_ok=True)
meta = {}


def dump(name, t, dtype=np.float32):
    np.ascontiguousarray(t.detach().cpu().numpy().astype(dtype)).tofile(os.path.join(OUT, name))


def hadamard(g):
    h4 = torch.tensor([[1.0 if a + b != 3 else -1.0 for b in range(4)] for a in range(4)]) / 2
    h = torch.ones(1, 1)
    while h.shape[0] < g:
        h = torch.kron(h, h4)
    return h


def dequant_sd(sd):
    """int8 convrot linears -> dense f32 W = (q s) R (R symmetric); everything else -> f32."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k == "tokenizer_json":
            continue
        if v.dtype == torch.int8:
            base = k[: -len(".weight")]
            s = sd[base + ".weight_scale"].float().reshape(-1, 1)
            w = v.float() * s
            cfg = bytes(sd[base + ".comfy_quant"].tolist()).decode()
            if '"convrot": true' in cfg:
                g = 256
                rows, cols = w.shape
                w = (w.reshape(rows, cols // g, g) @ hadamard(g)).reshape(rows, cols)
            out[k] = w
        else:
            out[k] = v.float()
    return out


def write_meta():
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k} {v}\n")


if MODE in ("dit", "cond"):
    from comfy.ldm.minimax_music.dit import MiniMaxMusic3DiT, latent_length

    sd = dequant_sd(comfy.utils.load_torch_file(DIT))
    model = MiniMaxMusic3DiT(dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", missing, "unexpected", unexpected)
    mm.in_training = True  # pure-torch RoPE path (same math as the comfy_kitchen kernel)
    with torch.inference_mode():
        if MODE == "dit":
            T = int(os.environ.get("T", "96"))
            x = torch.randn(1, 128, T)
            cond = torch.randn(1, 2048, T) * 0.5
            t = torch.tensor([0.3])
            out = model.diffusion_transformer(x, t, cond)
            dump("x.f32", x)
            dump("cond.f32", cond)
            dump("out.f32", out)
            meta.update(T=T, t=0.3)
        else:
            F = int(os.environ.get("F", "40"))
            hidden = torch.randn(1, F, 8 * 4096) * 2.0
            c = model.aligned_condition(hidden)
            dump("hidden.f32", hidden)
            dump("cond.f32", c)
            meta.update(F=F, L=c.shape[-1])
            print("cond", c.shape, "latent_length", latent_length(F))
elif MODE == "dav":
    import comfy.sd

    sd, md = comfy.utils.load_torch_file(DAV, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=md)
    T = int(os.environ.get("T", "48"))
    z = torch.randn(1, 128, T)
    with torch.inference_mode():
        wav = vae.first_stage_model.float().decode(z)
    dump("z.f32", z)
    dump("wav.f32", wav)
    meta.update(T=T, N=wav.shape[-1])
    print("wav", wav.shape)
elif MODE == "ar":
    import comfy.sd
    from comfy.ldm.minimax_music.ar import sample_topk
    from comfy.ldm.minimax_music.prompt import SPECIAL_TOKEN_IDS

    clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.MINIMAX)
    te = clip.cond_stage_model
    if not hasattr(te, "generate"):
        te = getattr(te, te.clip) if hasattr(te, "clip") else te.model
    mm.load_model_gpu(clip.patcher)
    caption = os.environ.get("CAPTION", "Fully instrumental. Slow solo piano, 66 bpm, F minor, soft and close.")
    lyrics = os.environ.get("LYRICS", "[intro]\n\n[outro]")
    tok = clip.tokenize(caption, lyrics=lyrics, seed=1, max_audio_frames=8)
    ids = [t for t, _ in tok["minimax_music3"][0]]
    nframes = int(os.environ.get("FRAMES", "6"))
    dt = torch.float32
    model = te.model
    print("tokens", len(ids), "pruned emb", model.pruned_embedding, "pruned head", model.pruned_lm_head)
    with torch.inference_mode():
        input_ids = torch.tensor([ids])
        unc = input_ids.clone()
        unc[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = torch.cat((input_ids, unc), 0)
        emb = model.embed_tokens_prefill(text_ids, out_dtype=dt)
        past = model.init_kv_cache(2, len(ids) + nframes + 2, "cpu", dt)
        out = model(None, embeds=emb, past_key_values=past, dtype=dt)
        last = out[0][:, -1]
        past = out[2]
        gen = torch.Generator().manual_seed(5)
        dec = model.audio_decoder
        codes_all, hid_all, logits0 = [], [], []
        for f in range(nframes):
            logits = model.lm_head_pruned(last).float()
            logits0.append(logits)
            guided = te._guided_c0(logits, 1.5, 50)
            code = sample_topk(guided, 50, gen)
            code = torch.where(code == 0, torch.ones_like(code), code)  # never stop in the reference
            c0 = (code - 1)[:1].repeat(2)
            c0e = model.embed_tokens_audio(c0, out_dtype=dt)
            seq = [dec.projection(last).unsqueeze(1), dec.projection(c0e).unsqueeze(1)]
            codes = [c0]
            parts = []
            for i in range(1, 8):
                o = dec(torch.cat(seq, 1))[:, -1]
                parts.append(o[:1])
                lg = dec.audio_heads[i - 1](o).float()
                c = sample_topk(lg[1:2] + (lg[:1] - lg[1:2]) * 1.5, 50, gen).repeat(2)
                codes.append(c)
                if i < 7:
                    e = model.audio_extra_embedding(c + (i - 1) * 1024, out_dtype=dt)
                    seq.append(dec.projection(e).unsqueeze(1))
            codes = torch.stack(codes, 1)
            codes_all.append(codes[0])
            hid_all.append(torch.cat([last[:1]] + parts, -1)[0])
            c0f = model.embed_tokens_audio(codes[:, 0], out_dtype=dt)
            offs = torch.arange(7) * 1024
            extra = model.audio_extra_embedding(codes[:, 1:] + offs.unsqueeze(0), out_dtype=dt).sum(1)
            fb = ((c0f + extra) * 8 ** -0.5).unsqueeze(1)
            out = model(None, embeds=fb, past_key_values=past, dtype=dt)
            last = out[0][:, -1]
            past = out[2]
        dump("ids.i32", torch.tensor(ids), np.int32)
        dump("codes.i32", torch.stack(codes_all), np.int32)
        dump("hidden.f32", torch.stack(hid_all))
        dump("logits0.f32", torch.stack(logits0))
        meta.update(P=len(ids), FRAMES=nframes)
        print("codes", torch.stack(codes_all).tolist())
write_meta()
print("wrote", OUT, meta)
