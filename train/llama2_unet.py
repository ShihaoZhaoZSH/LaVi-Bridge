import os
import sys
sys.path.append("../")
import math
import inspect
import random
import itertools
import argparse
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from modules.lora import inject_trainable_lora_extended, save_lora_weight
from modules.adapters import TextAdapter


# Dataset
class ImageTextDataset(Dataset):
    def __init__(self, anno_path, image_size):
        f = open(anno_path)
        lines = f.readlines()
        f.close()
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.data = [line.strip().split("\t") for line in lines]

    def __getitem__(self, index):
        try:
            image = Image.open(self.data[index][0]).convert("RGB")
            image = self.preprocess(image)
            prompt = self.data[index][1]
            return image, prompt
        except:
            # Resample a new one
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def __len__(self):
        return len(self.data)


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LaVi-Bridge Training")
    parser.add_argument("--anno_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llama2_dir", type=str, default="")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Blocks to inject LoRA
    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
    TEXT_ENCODER_REPLACE_MODULES = {"LlamaAttention"}

    # Modules of T2I diffusion models
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vis = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    tokenizer = LlamaTokenizer.from_pretrained(args.llama2_dir)
    text_encoder = LlamaForCausalLM.from_pretrained(args.llama2_dir)
    adapter = TextAdapter(4096, 2432, 768)
    
    tokenizer.pad_token = '[PAD]'
    vis.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # LoRA injection
    vis_lora_params, _ = inject_trainable_lora_extended(
        vis, 
        r=args.lora_rank, 
        target_replace_module=VIS_REPLACE_MODULES,
    )
    text_encoder_lora_params, _ = inject_trainable_lora_extended(
        text_encoder, 
        r=args.lora_rank, 
        target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
    )

    # Optimizer and scheduler
    optimizer_class = torch.optim.AdamW
    params_to_optimize = ([
        {"params": itertools.chain(*vis_lora_params), "lr": 1e-4},
        {"params": itertools.chain(*text_encoder_lora_params), "lr": 5e-6},
        {"params": adapter.parameters(), "lr": 1e-4},
    ])
    optimizer = optimizer_class(params_to_optimize, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.max_train_steps)

    # Dataset and dataloader
    train_dataset = ImageTextDataset(anno_path=args.anno_path, image_size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1)

    vis, text_encoder, adapter, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vis, 
        text_encoder, 
        adapter, 
        vae, 
        optimizer, 
        train_dataloader, 
        lr_scheduler,
    )

    global_step = 0
    last_save = 0
    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    if accelerator.is_main_process:
        accelerator.init_trackers("training", config=vars(args))

    # Log
    print(f"Num examples = {len(train_dataset)}")
    print(f"Total batch size = {args.train_batch_size * accelerator.num_processes}")
    print(f"Num Epochs = {num_train_epochs}")
    print(f"Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training
    for _ in range(num_train_epochs):
        vis.train()
        text_encoder.train()
        adapter.train()

        for _, batch in enumerate(train_dataloader):
            # Latent preparation
            latents = vae.encode(batch[0]).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            timesteps = timesteps.long()

            # Model prediction
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_input = tokenizer(
                batch[1], 
                padding="max_length", 
                max_length=77, 
                return_tensors="pt", 
                truncation=True,
            ).input_ids.to(accelerator.device)
            encoder_hidden_states_pre = text_encoder(text_input, output_hidden_states=True).hidden_states[-1]
            encoder_hidden_states = adapter(encoder_hidden_states_pre).sample
            model_pred = vis(noisy_latents, timesteps, encoder_hidden_states).sample

            # Optimization
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (itertools.chain(vis.parameters(), text_encoder.parameters(), adapter.parameters()))
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            progress_bar.update(1)

            # Saving
            if accelerator.sync_gradients and accelerator.is_main_process and global_step - last_save >= args.save_steps:
                accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(inspect.signature(accelerator.unwrap_model).parameters.keys())
                extra_args = ({"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {})
                save_lora_weight(
                    accelerator.unwrap_model(vis, **extra_args), 
                    f"{args.output_dir}/s{global_step}_lora_vis.pt", 
                    target_replace_module=VIS_REPLACE_MODULES,
                )
                save_lora_weight(
                    accelerator.unwrap_model(text_encoder, **extra_args), 
                    f"{args.output_dir}/s{global_step}_lora_text.pt", 
                    target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
                )
                accelerator.unwrap_model(adapter, **extra_args).save_pretrained(f"{args.output_dir}/s{global_step}_adapter")
                last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
