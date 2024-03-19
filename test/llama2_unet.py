import os
import sys
sys.path.append("../")
import argparse
from tqdm.auto import tqdm
from PIL import Image

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from transformers import LlamaForCausalLM, LlamaTokenizer

from modules.lora import monkeypatch_or_replace_lora_extended
from modules.adapters import TextAdapter


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llama2_dir", type=str, default="")
    args = parser.parse_args()
    return args


def main(args, prompts):
    os.makedirs(args.output_dir, exist_ok=True)
    
    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
    TEXT_ENCODER_REPLACE_MODULES = {"LlamaAttention"}
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    torch_device = "cuda"
    pos_prompt = ", best quality, extremely detailed, 4k resolution"

    # Modules of T2I diffusion models
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)
    vis = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    tokenizer = LlamaTokenizer.from_pretrained(args.llama2_dir)
    # To perform inference on a 24GB GPU memory, llama2 was converted to half precision
    text_encoder = LlamaForCausalLM.from_pretrained(args.llama2_dir, torch_dtype=torch.float16).to(torch_device)
    adapter = TextAdapter.from_pretrained(os.path.join(args.ckpt_dir, f"adapter")).to(torch_device)
    tokenizer.pad_token = '[PAD]'

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
            text_ids = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt", truncation=True).input_ids.to(torch_device)
            text_embeddings = text_encoder(input_ids=text_ids, output_hidden_states=True).hidden_states[-1].to(torch.float32)
            text_embeddings = adapter(text_embeddings).sample
            uncond_input = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            # Convert the text embedding back to full precision
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device), output_hidden_states=True).hidden_states[-1].to(torch.float32)
            uncond_embeddings =  adapter(uncond_embeddings).sample
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Latent preparation
            latents = torch.randn((1, vis.in_channels, height // 8, width // 8)).to(torch_device)
            latents = latents * noise_scheduler.init_noise_sigma

            # Model prediction
            noise_scheduler.set_timesteps(num_inference_steps)
            for t in tqdm(noise_scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = vis(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

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
