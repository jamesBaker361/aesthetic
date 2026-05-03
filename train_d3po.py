from collections import defaultdict
import contextlib
import os
import copy
import datetime
from concurrent import futures
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import cast_training_params, compute_snr
import numpy as np
import d3po_prompts
import d3po_rewards
from pipeline_with_logprob import pipeline_with_logprob
from ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def unet_lora_state_dict(unet: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    # The matrix name can either be "down" or "up".
                    lora_state_dict[f"{name}.lora.{lora_layer_matrix_name}"] = lora_param

    return lora_state_dict


def train_and_save(config,
                   size:int,
         project_name:str,
         pretrained_model:str,
         prompt_fn_name:str,
         reward_fn_name:str,
         batch_size:int,save_dir:str)->StableDiffusionPipeline:

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=project_name, config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )
    # set seed (device_specific is very important to get different prompts on different devices)
    np.random.seed(config.seed)
    available_devices = accelerator.num_processes
    random_seeds = np.random.randint(0,100000,size=available_devices)
    device_seed = random_seeds[accelerator.process_index]
    set_seed(123)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16)
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    if not config.use_lora and config.train.activation_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    ref =  copy.deepcopy(pipeline.unet)
    for param in ref.parameters():
        param.requires_grad = False
        
    if config.use_lora:
        # Set correct lora layers
        unet=pipeline.unet
        UNET_TARGET_MODULES = [
            "to_q",
            "to_k",
            "to_v",
        ]
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=True,
        )
        
        unet.requires_grad_(False)
        unet.add_adapter(lora_config)
        trainable_layers = filter(lambda p: p.requires_grad, unet.parameters())
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                safe_serialization=True,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        # returns a tuple of state dictionary and network alphas
        lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(input_dir)

        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # throw warning if some unexpected keys are found and continue loading
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32
        if config.mixed_precision in ["fp16"]:
            cast_training_params([unet_], dtype=torch.float32)

    # Support multi-dimensional comparison. Default demension is 1. You can add many rewards instead of only one to judge the preference of images.
    # For example: A: clipscore-30 blipscore-10 LAION aesthetic score-6.0 ; B: 20, 8, 5.0  then A is prefered than B
    # if C: 40, 4, 4.0 since C[0] = 40 > A[0] and C[1] < A[1], we do not think C is prefered than A or A is prefered than C 
    def compare(a, b):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        if len(a.shape)==1:
            a = a[...,None]
            b = b[...,None]

        a_dominated = torch.logical_and(torch.all(a <= b, dim=1), torch.any(a < b, dim=1))
        b_dominated = torch.logical_and(torch.all(b <= a, dim=1), torch.any(b < a, dim=1))

        c = torch.zeros([a.shape[0],2],dtype=torch.float,device=a.device)

        c[a_dominated] = torch.tensor([-1., 1.],device=a.device)
        c[b_dominated] = torch.tensor([1., -1.],device=a.device)

        return c


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(d3po_prompts, prompt_fn_name)
    reward_fn = getattr(d3po_rewards, reward_fn_name)()

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )


    assert config.sample.batch_size >= batch_size
    assert config.sample.batch_size % batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0
    
    os.makedirs(save_dir,exist_ok=True)
    config.resume_from=save_dir
    if "checkpoint_" not in os.path.basename(config.resume_from):
        # get the most recent checkpoint in this directory
        checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
        if len(checkpoints) == 0:
            first_epoch = 0
        else:
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
            accelerator.load_state(config.resume_from)
            first_epoch = int(config.resume_from.split("_")[-1]) + 1

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompt_metadata = None
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts1, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(batch_size)]
            )
            prompts2 = prompts1
            # encode prompts
            prompt_ids1 = pipeline.tokenizer(
                prompts1,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
            prompt_embeds2 = prompt_embeds1

            # sample
            with autocast():
                images1, _, latents1, log_probs1 = pipeline_with_logprob(
                    pipeline,
                    height=size,
                    width=size,
                    prompt_embeds=prompt_embeds1,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
                latents1 = torch.stack(latents1, dim=1)
                log_probs1 = torch.stack(log_probs1, dim=1)
                images2, _, latents2, log_probs2 = pipeline_with_logprob(
                    pipeline,
                    height=size,
                    width=size,
                    prompt_embeds=prompt_embeds2,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    latents = latents1[:,0,:,:,:]  # same init noise; requires eta > 0 to diverge
                )
                latents2 = torch.stack(latents2, dim=1)
                log_probs2 = torch.stack(log_probs2, dim=1)

            latents = torch.stack([latents1,latents2], dim=1)  # (batch_size, 2, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack([log_probs1,log_probs2], dim=1)  # (batch_size, num_steps, 1)
            prompt_embeds = torch.stack([prompt_embeds1,prompt_embeds2], dim=1)
            images = torch.stack([images1,images2], dim=1)
            current_latents = latents[:, :, :-1]
            next_latents = latents[:, :, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards concurrently
            future1 = executor.submit(reward_fn, images1, prompts1, prompt_metadata)
            future2 = executor.submit(reward_fn, images2, prompts2, prompt_metadata)
            rewards1 = future1.result()[0]
            rewards2 = future2.result()[0]
            if isinstance(rewards1, np.ndarray):
                rewards = np.c_[rewards1, rewards2]
            else:
                rewards1 = rewards1.cpu().detach().numpy()
                rewards2 = rewards2.cpu().detach().numpy()
                rewards = np.c_[rewards1, rewards2]
            eval_rewards = None
            if epoch%config.sample.eval_epoch==0:
                eval_prompts, eval_prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.eval_batch_size)])

                # encode prompts
                eval_prompt_ids = pipeline.tokenizer(
                    eval_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
                eval_prompt_embeds = pipeline.text_encoder(eval_prompt_ids)[0]
                eval_sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.eval_batch_size, 1, 1)
                # sample
                with autocast():
                    eval_images, _, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=eval_prompt_embeds,
                        negative_prompt_embeds=eval_sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                    )    
                eval_rewards = executor.submit(reward_fn, eval_images, eval_prompts, eval_prompt_metadata).result()[0]     
                samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "prompts": prompts1,
                    "timesteps": timesteps,
                    "latents": current_latents,  # each entry is the latent before timestep t
                    "next_latents": next_latents,  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "images":images,
                    "rewards":torch.as_tensor(rewards, device=accelerator.device),
                    "eval_rewards":torch.as_tensor(eval_rewards, device=accelerator.device),
                }
                )
            else:
                prompts1 = list(prompts1)
                samples.append(
                    {
                        "prompt_embeds": prompt_embeds,
                        "prompts": prompts1,
                        "timesteps": timesteps,
                        "latents": current_latents,  # each entry is the latent before timestep t
                        "next_latents": next_latents,  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "images":images,
                        "rewards":torch.as_tensor(rewards, device=accelerator.device),
                    }
                )
        prompts = samples[0]["prompts"]
        del samples[0]["prompts"]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        images = samples["images"]
        rewards = accelerator.gather(samples["rewards"]).cpu().detach().numpy()
        #record the better images' reward based on the reward model.
        if eval_rewards is not None:
            eval_rewards = accelerator.gather(samples["eval_rewards"]).cpu().detach().numpy()
            accelerator.log(
                {"eval_reward": eval_rewards, "num_samples": epoch*available_devices*config.sample.batch_size, "eval_reward_mean": eval_rewards.mean(), "eval_reward_std": eval_rewards.std()},
                step=global_step,
            )
            del samples["eval_rewards"]
        else:
            accelerator.log(
                {"reward": rewards, "num_samples": epoch*available_devices*config.sample.batch_size, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
                step=global_step,
            )
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards[:,0]))
                    ],
                },
                step=global_step,
            )
        # save prompts
        del samples["images"]
        torch.cuda.empty_cache()
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps
        orig_sample = copy.deepcopy(samples)
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in orig_sample.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents"]:
                tmp = samples[key].permute(0,2,3,4,5,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
                samples[key] = tmp.permute(0,5,1,2,3,4)
            samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms].unsqueeze(1).repeat(1,2,1)
            tmp = samples["log_probs"].permute(0,2,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            samples["log_probs"] = tmp.permute(0,2,1)
            # rebatch for training
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i in tqdm(range(0,total_batch_size,batch_size),
                        desc="Update",
                        position=2,
                        leave=False, 
                          ):
                sample_0 = {}
                sample_1 = {}
                for key, value in samples.items():
                    sample_0[key] = value[i:i+batch_size, 0]
                    sample_1[key] = value[i:i+batch_size, 1]
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"]])
                    embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"]])
                else:
                    embeds_0 = sample_0["prompt_embeds"]
                    embeds_1 = sample_1["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=3,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):  
                    with accelerator.accumulate(pipeline.unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred_0 = pipeline.unet(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                                noise_pred_0 = noise_pred_uncond_0 + config.sample.guidance_scale * (noise_pred_text_0 - noise_pred_uncond_0)

                                noise_ref_pred_0 = ref(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                                noise_ref_pred_0 = noise_ref_pred_uncond_0 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                                )

                                noise_pred_1 = pipeline.unet(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                                noise_pred_1 = noise_pred_uncond_1 + config.sample.guidance_scale * (noise_pred_text_1 - noise_pred_uncond_1)

                                noise_ref_pred_1 = ref(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                                noise_ref_pred_1 = noise_ref_pred_uncond_1 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                                )

                            else:
                                noise_pred_0 = pipeline.unet(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample
                                noise_ref_pred_0 = ref(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample

                                noise_pred_1 = pipeline.unet(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample
                                noise_ref_pred_1 = ref(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample



                            # compute the log prob of next_latents given latents under the current model
                            _, total_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_ref_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                            _, total_ref_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                        human_prefer = compare(sample_0['rewards'],sample_1['rewards'])
                        # clip the Q value
                        ratio_0 = torch.clamp(torch.exp(total_prob_0-total_ref_prob_0),1 - config.train.eps, 1 + config.train.eps)
                        ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1),1 - config.train.eps, 1 + config.train.eps)
                        loss = -torch.log(torch.sigmoid(config.train.beta*(torch.log(ratio_0))*human_prefer[:,0] + config.train.beta*(torch.log(ratio_1))*human_prefer[:, 1])).mean()

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_layers, config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        info["loss"].append(loss.item())

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                assert (j == num_train_timesteps - 1) and (
                    i + 1
                ) % config.train.gradient_accumulation_steps == 0
                # log training-related stuff
                info = {k: torch.tensor(v).mean() for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients

            if epoch!=0 and (epoch+1) % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()
    return pipeline


if __name__ == "__main__":
    from d3po.config.base import get_config
    config=get_config()
    config.sample.batch_size=1
    config.sample.num_steps=2
    config.num_epochs=2
    train_and_save(
        config,64,"testing_d3po","SimianLuo/LCM_Dreamshaper_v7","merged_prompts","aesthetic_score",1,"ddpo_save_dir"
    )
