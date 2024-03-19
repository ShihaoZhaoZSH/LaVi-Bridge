import os
import sys
sys.path.append("../")
import argparse
from tqdm.auto import tqdm
from PIL import Image

import torch
from diffusers import AutoencoderKL, UniPCMultistepScheduler, Transformer2DModel
from transformers import AutoTokenizer, T5EncoderModel

from modules.lora import monkeypatch_or_replace_lora_extended
from modules.adapters import TextAdapter


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    return args


def main(args, prompts):
    os.makedirs(args.output_dir, exist_ok=True)
    
    VIS_REPLACE_MODULES = {"Attention", "GEGLU"}
    TEXT_ENCODER_REPLACE_MODULES = {"T5Attention"}
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    torch_device = "cuda"
    pos_prompt = ", best quality, extremely detailed, 4k resolution"

    # Modules of T2I diffusion models
    vae = AutoencoderKL.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="vae").to(torch_device)
    vis = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="transformer").to(torch_device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    text_encoder = T5EncoderModel.from_pretrained("t5-large").to(torch_device)
    adapter = TextAdapter.from_pretrained(os.path.join(args.ckpt_dir, f"adapter"), use_safetensors=True).to(torch_device)

    # LoRA
    monkeypatch_or_replace_lora_extended(
        vis, 
        torch.load(os.path.join(args.ckpt_dir, f"lora_vis.pt")), 
        r=32, 
        target_replace_module=VIS_REPLACE_MODULES,
    )
    monkeypatch_or_replace_lora_extended(
        text_encoder, 
        torch.load(os.path.join(args.ckpt_dir, f"lora_text.pt")), 
        r=32, 
        target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
    )

    vae.eval()
    vis.eval()
    text_encoder.eval()
    adapter.eval()

    # Inference
    with torch.no_grad():
        for prompt in prompts:
            print(prompt)
            orig_prompt = prompt
            prompt += pos_prompt

            # Text embeddings
            text_inputs = tokenizer(prompt, padding="max_length", max_length=120, add_special_tokens=True, return_tensors="pt", truncation=True).to(torch_device)
            text_input_ids = text_inputs.input_ids
            prompt_attention_mask = text_inputs.attention_mask
            encoder_hidden_states = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
            encoder_hidden_states = adapter(encoder_hidden_states).sample
            neg_text_inputs = tokenizer([""], padding="max_length", max_length=120, add_special_tokens=True, return_tensors="pt", truncation=True).to(torch_device)
            neg_text_input_ids = neg_text_inputs.input_ids
            neg_prompt_attention_mask = neg_text_inputs.attention_mask
            neg_encoder_hidden_states = text_encoder(neg_text_input_ids, attention_mask=neg_prompt_attention_mask)[0]
            neg_encoder_hidden_states = adapter(neg_encoder_hidden_states).sample
            text_embeddings = torch.cat([neg_encoder_hidden_states, encoder_hidden_states])
            attention_mask = torch.cat([neg_prompt_attention_mask, prompt_attention_mask])

            # Latent preparation
            latents = torch.randn((1, vis.in_channels, height // 8, width // 8)).to(torch_device)
            latents = latents * noise_scheduler.init_noise_sigma

            # Model prediction
            noise_scheduler.set_timesteps(num_inference_steps)
            for t in tqdm(noise_scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = vis(
                    latent_model_input, 
                    encoder_hidden_states=text_embeddings,
                    encoder_attention_mask=attention_mask,
                    timestep=torch.Tensor([t.item()]).expand(latent_model_input.shape[0]).to(torch_device),
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.chunk(2, dim=1)[0]
                latents = noise_scheduler.step(noise_pred, t, latents)[0]

            # Decoding
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
            image.save(f"{args.output_dir}/{orig_prompt[: 200]}.png")


if __name__ == "__main__":
    prompts = [
        "city",
        "Oppenheimer sits on the beach on a chair, watching a nuclear exposition with a huge mushroom cloud, 120mm.",
        "Tiny potato kings wearing majestic crowns, sitting on thrones, overseeing their vast potato kingdom filled with potato subjects and potato castles.",
        "An illustration of a human heart made of translucent glass, standing on a pedestal amidst a stormy sea. Rays of sunlight pierce the clouds, illuminating the heart, revealing a tiny universe within.",
        "freshly made hot floral tea in glass kettle on the table.",
        "In homage to old-world botanical sketches, an illustration is rendered with detailed lines and subtle watercolor touches. The artwork captures an unusual fusion: a cactus bearing not just thorns but also the fragrant and delicate blooms of a lilac, all while taking on the mesmerizing form of a Möbius strip, capturing the essence of nature’s diverse beauty and mathematical intrigue.",
        "Portrait photography, a woman in a glamorous makeup, wearing a mask with tassels, in the style of midsommar by Ari Aster, made of flowers, bright pastel colors, prime lense.",
        "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
        "Dark high contrast render of a psychedelic tree of life illuminating dust in a mystical cave.",
    ]
    args = parse_args()
    main(args, prompts)
