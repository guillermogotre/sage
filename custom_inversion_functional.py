import os

import torch
import torch.nn.functional as nnf
from torchvision.transforms.functional import pil_to_tensor

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler 

import gradio as gr
import numpy as np
from tqdm.auto import tqdm
import PIL.Image
import matplotlib.pyplot as plt

from sage_tools.guided_editing import EditableAttnProcessor, AttnStore, AttnReweight, \
    AttnCost, AttnDummy, AttnMask, AttnReplace, \
    get_attn_scale, get_prompt_embedding, get_index, compose_editing
    
from sage_tools.visualization import plot_mask_history, save_image, generate_random_path, \
    plot_z0_history, torch_to_pil, latent_to_torch, latent_to_torch,\
    plot_cross_attn_history

is_cuda = torch.cuda.is_available()
if is_cuda:
    DTYPE = torch.float16
    DEVICE = torch.device("cuda:0")
    DEFAULT_LOSS_SCALE = 500
else:
    DTYPE = torch.float32
    DEVICE= 'cpu'
    DEFAULT_LOSS_SCALE = 300
    
loss_dict = dict(mae=nnf.l1_loss, mse=nnf.mse_loss)

pipe = None
forward_scheduler = None
backward_scheduler = None

last_model_id = None


def load_model(model_id):
    global pipe, last_model_id
    if model_id != last_model_id:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        last_model_id = model_id
        
def load_all(model_id, ddim_steps):
    load_model(model_id)

def parse_attn_scale(attn_str):
    if not attn_str:
        return None
    key_value_pairs = [kv.split(":") for kv in attn_str.split(",")]
    return {k: float(v) for k, v in key_value_pairs}

def parse_replace(replace_str):
    if not replace_str:
        return None
    return tuple(replace_str.split(","))

def clean_str(s):
    return " ".join(s.split())

import json
def remove_non_serializable_elements(d):
    non_serializable_keys = []
    for key, value in d.items():
        try:
            json.dumps(value)
        except TypeError:
            non_serializable_keys.append(key)
    
    for key in non_serializable_keys:
        del d[key]
        

def run(
    # Input
    input_image: PIL.Image.Image,
    prompt_str,
    edited_prompt_str,
    attn_scale = None,
    replace = None,
    blend = None,
    
    # System parameters
    seed = 8888,
    side = 512,
    ddim_steps=50,
    use_vae_mean = True,
    model_id = "CompVis/stable-diffusion-v1-4",
    use_trailing = False,
    
    # I2I parameters
    self_latent_guidance_scale = 250.,
    cross_replace_steps = 0.,
    sag_mask_min = 1.,
    cfg_mask_min = 1.,
    zt_replace_steps_min = 0.,
    cfg_value = 7.5,
    loss_scale = None,
    max_steps = 40,
    use_monotonical_scale = True,
    noise_alpha = 0.0,
    loss_type = "mae",
    
    self_layer=1,
    cross_layer=2,
    reconstruction_type="sage",
    
    # Return history
    return_vae_rec = False,
    return_ddim_inv_rec = False,
    return_mask_history = False,
    return_z0_estimation_history = False,
    return_cross_attn_history = False,
    return_pixelwise_epsilon_norm_history = False,
    return_pixelwise_selfattn_grad_norm_history = False,
    
    # Save copy
    save_copy=False,
    disable_tqdm=False,
):
    cross_guidance = False
    z0_guidance = False
    replace_self_attn = False
    
    if reconstruction_type == "cross":
        cross_guidance = True
    elif reconstruction_type == "z0":
        z0_guidance = True
    elif reconstruction_type == "replace":
        replace_self_attn = True
        
    loss_scale = DEFAULT_LOSS_SCALE if loss_scale is None or loss_scale == 0 else loss_scale
        
        
        
    # Set up args
    args_dict = locals().copy()
    del args_dict["input_image"]
    remove_non_serializable_elements(args_dict)
    
    
    do_mask = (sag_mask_min < 1) or (cfg_mask_min < 1) or (zt_replace_steps_min > 0)
    if  return_mask_history and not do_mask:
        do_mask = True
        print("Masks are not used, but return_mask_history is set to True. Setting do_mask to True.")
        
    do_replace = cross_replace_steps > 0
    if return_cross_attn_history and not do_replace:
        do_replace = True
        print("Cross attention is not used, but return_cross_attn_history is set to True. Setting do_replace to True.")
        
    do_store_inversion_latents = zt_replace_steps_min > 0
    
    replace = parse_replace(replace)
    blend = parse_replace(blend)
    attn_scale = parse_attn_scale(attn_scale)
    side = int(side)
    ddim_steps = int(ddim_steps)
    seed = int(seed)
    self_layer = int(self_layer)
    cross_layer = int(cross_layer)
    prompt_str = clean_str(prompt_str)
    edited_prompt_str = clean_str(edited_prompt_str)
    
    
    # Assertions
    assert not (do_replace and replace is None)
    assert not (do_mask and blend is None)
    assert not (cross_guidance and z0_guidance)
    
    # Default return values
    mask_history_img = None
    vae_reconstruction = None
    ddim_reconstruction = None
    z0_estimation = None
    cross_attn_history = None
    prompt_latent_norm = None
    pixelwise_epsilon_norm_history = None
    pixelwise_selfattn_grad_norm_history = None
    
    # Load model
    load_all(model_id, ddim_steps)
    
    # Schedulers
    forward_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler= DDIMInverseScheduler.from_config(pipe.scheduler.config)
    
    if use_trailing:
        forward_scheduler.config.timestep_spacing = "trailing"
        inverse_scheduler.config.timestep_spacing = "trailing"
        
    forward_scheduler.set_timesteps(ddim_steps)
    inverse_scheduler.set_timesteps(ddim_steps)
    
    timesteps = forward_scheduler.timesteps
    timesteps = timesteps.tolist() + [0]
        
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set up input
    input_image = input_image.convert("RGB").resize((side, side), PIL.Image.LANCZOS)
    img_tensor = pil_to_tensor(input_image).unsqueeze(0).to(DEVICE, DTYPE) / 127.5 - 1.0
    self_side=side//(8*(2**self_layer))
    cross_side=side//(8*(2**cross_layer))
    
    with torch.no_grad():
        # VAE Encode
        latents_t0 = pipe.vae.encode(img_tensor)
        if use_vae_mean:
            latents_t0 = latents_t0.latent_dist.mean
        else:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            latents_t0 = latents_t0.latent_dist.sample(generator=generator)
        
        latents_t0 *= pipe.vae.config.scaling_factor
        
        # VAE Reconstruction
        if return_vae_rec:
            # image_rec = pipe.vae.decode(latents_t0  / pipe.vae.config.scaling_factor ).sample
            # image_rec = image_rec.clamp(-1,1) * 0.5 + 0.5
            # vae_reconstruction = PIL.Image.fromarray((image_rec.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))
            vae_reconstruction = torch_to_pil(latent_to_torch(latents_t0, pipe.vae))
        
        # Input transform
        attn_scale = {} if attn_scale is None else attn_scale
        
        # Prompt
        edited_prompt_tokens, edited_prompt_embeds = get_prompt_embedding(
            pipe,
            prompt_str,
            edited_prompt_str,
        )
        source_embeddings, edited_embeddings  = edited_prompt_embeds.chunk(2)
        prompt_latent_norm = torch.norm(source_embeddings - edited_embeddings).item()
        args_dict["prompt_latent_norm"] = prompt_latent_norm
        prompt_latent_norm = f"{prompt_latent_norm:.4f}"
        
        # Attention editing
        if return_cross_attn_history:
            attn_store = AttnStore(DEVICE, DTYPE, self_side=self_side, cross_side=cross_side)
        elif cross_guidance:
            attn_store = AttnStore(DEVICE, DTYPE, self_side=None, cross_side=cross_side)
        elif z0_guidance:
            attn_store = AttnStore(DEVICE, DTYPE, self_side=None, cross_side=None)
        else:
            attn_store = AttnStore(DEVICE, DTYPE, self_side=self_side)
        cross_attn_scale = get_attn_scale(attn_scale, edited_prompt_tokens[-1], pipe.tokenizer).to(DEVICE, DTYPE)
        attn_reweight = AttnReweight(cross_attn_scale).disable()
        attn_mask = AttnDummy()
        attn_replace = AttnDummy()

        if replace is not None and do_replace:
            # Attention Replace
            source_id = get_index(prompt_str, replace[0], pipe.tokenizer)
            edited_id = get_index(edited_prompt_str, replace[1], pipe.tokenizer)
            
            attn_replace = AttnReplace(DEVICE, DTYPE, source_id, edited_id, cross_side=cross_side, max_t=timesteps[int(len(timesteps)*cross_replace_steps)])    
        
        if blend is not None and do_mask:
            attn_mask = AttnMask(pipe.tokenizer, [prompt_str, edited_prompt_str], blend, DEVICE, DTYPE, cross_side=cross_side).disable()

        # TODO add mask if replace is not None
        editing = compose_editing(attn_replace, attn_mask, attn_reweight)

        # Configure diffusion
        attn_cost = AttnCost(store=attn_store, editing=editing, replace=replace_self_attn)
        unet = pipe.unet
        unet.set_attn_processor(EditableAttnProcessor(attn_cost))
        
        #
        # DDIM inversion
        #
        latents = latents_t0
        unedited_latents_history = [] if do_store_inversion_latents else None
        for i, t in enumerate(tqdm(timesteps[::-1][:ddim_steps], desc="DDIM Inversion", leave=False, disable=disable_tqdm)):
            attn_store.set_t(t)
            attn_replace.set_t(t)
            
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=source_embeddings).sample
            latents = inverse_scheduler.step(noise_pred, t, latents, ).prev_sample
            if do_store_inversion_latents:
                unedited_latents_history.append(latents.detach().clone())
            
        init_latents = latents.detach().clone()    
        # Fix reference attention maps    
        attn_store.fix()
        attn_replace.fix()
        
        
        # DDIM inversion reconstruction
        if return_ddim_inv_rec:
            attn_store.disable()
            attn_mask.disable()
            attn_replace.disable()
            attn_reweight.disable()
            for i, t in enumerate(tqdm(timesteps[:ddim_steps], desc="DDIM Inversion Reconstruction", leave=False, disable=disable_tqdm)): 
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=source_embeddings).sample
                latents = forward_scheduler.step(noise_pred, t, latents, ).prev_sample
            attn_store.enable()
            attn_replace.enable()
            ddim_reconstruction = torch_to_pil(latent_to_torch(latents, pipe.vae))
            
            latents = init_latents.detach().clone()
        
        
        
        # Add noise
        if noise_alpha > 0.:
            latents_noise = torch.randn_like(init_latents)
            assert 0 <= noise_alpha <= 1
            b = noise_alpha
            a = (1-noise_alpha)
            latents = (a * latents + b * latents_noise)/np.sqrt(a**2 + b**2)
            
        #
        # Backward diffusion
        #
        attn_mask.reset()
        attn_mask.enable()
        
    history = {
        "cond": [],
        "uncond": [],
        "cfg": [],
    } if return_z0_estimation_history else None
        
    mask_history = [] if do_mask else None
    epsilon_norm_history = [] if return_pixelwise_epsilon_norm_history else None    
    selfattn_grad_norm_history = [] if return_pixelwise_selfattn_grad_norm_history else None

    do_grad = cross_guidance or z0_guidance or not replace_self_attn
    for i, t in enumerate(tqdm(timesteps[:ddim_steps], desc="Guided Diffusion", leave=False, disable=disable_tqdm)):    
        t_next = timesteps[i+1]
        attn_store.set_t(t_next)
        attn_replace.set_t(t_next)
        latents_ori = latents.detach().clone()
        latents_ori.requires_grad = do_grad
        
        with torch.no_grad(): # Positive prompt / p^{out}
            attn_store.disable()
            attn_reweight.enable()
            attn_replace.enable()
            
            if attn_mask is not None:
                attn_mask.set_prompt_idx(1) # p^{out}
                
            cond_pred = pipe.unet(latents_ori, t, encoder_hidden_states=edited_embeddings).sample    
            
            # Reconstruction history
            if return_z0_estimation_history:
                history["cond"].append(
                    forward_scheduler.step(cond_pred, t, latents_ori, return_dict=True).pred_original_sample.detach().clone().cpu()
                )
            
            attn_store.enable()
            attn_reweight.disable()
            attn_replace.disable()
        
        # Negative prompt / p^{in}
        attn_mask.set_prompt_idx(0) # p^{in}
        uncond_pred = pipe.unet(latents_ori, t, encoder_hidden_states=source_embeddings).sample    
        
        # Reconstruction history
        if return_z0_estimation_history:
            with torch.no_grad():
                history["uncond"].append(
                    forward_scheduler.step(uncond_pred, t, latents_ori, return_dict=True).pred_original_sample.detach().clone().cpu()
                )
    
        # TODO compute "CFG noise_pred" & "self-att guidance" pixelwise norm
        if cfg_mask_min < 1: 
            guidance_value = attn_mask.mask_like(cond_pred).clamp(cfg_mask_min,1)*cfg_value
        else:
            guidance_value = cfg_value
        pred_diff = cond_pred - uncond_pred
        noise_pred = uncond_pred + guidance_value * pred_diff
        
        grad = [0.]
        if do_grad:
            # Get loss
            attn_pair = attn_store.get_t(t_next)
            ref_attention = attn_pair.reference
            cur_attention = attn_pair.current
            
            if cross_guidance:
                guidance_loss = loss_dict[loss_type](
                    torch.cat(cur_attention.cross).to(DEVICE),  # torch.cat([e for e in cur_attention.self]).to(DEVICE), 
                    torch.cat(ref_attention.cross).to(DEVICE))
            elif z0_guidance:
                t0_estimation = forward_scheduler.step(uncond_pred, t, latents_ori, ).pred_original_sample
                guidance_loss = loss_dict[loss_type](t0_estimation, latents_t0)
            else:
                guidance_loss = loss_dict[loss_type](
                    torch.cat(cur_attention.self).to(DEVICE),  # torch.cat([e for e in cur_attention.self]).to(DEVICE), 
                    torch.cat(ref_attention.self).to(DEVICE))
            loss = guidance_loss * loss_scale
            grad = torch.autograd.grad(loss, latents_ori)
        else:
            attn_store.clear_t(t_next)
            
        
        # Apply reconstruction guidance
        with torch.no_grad():
            monotonical_scale = 1 - (i / len(timesteps)) if use_monotonical_scale else 1
            if i > max_steps:
                monotonical_scale = 0
            
            noise_step = forward_scheduler.step(noise_pred, t, latents_ori, )
            latents = noise_step.prev_sample
            
            if return_z0_estimation_history:
                history["cfg"].append(
                    noise_step.pred_original_sample.detach().clone().cpu()
                )
                
            grad_guidance = self_latent_guidance_scale * grad[0] * monotonical_scale
            
            if sag_mask_min < 1:
                grad_guidance = grad_guidance * (1 - attn_mask.mask_like(latents)).clamp_(sag_mask_min,1)
            latents -= grad_guidance
            
            if do_mask:
                mask_history.append(attn_mask.mask.detach().cpu())
                
            if return_pixelwise_epsilon_norm_history:
                epsilon_norm_history.append(torch.norm(pred_diff, dim=(0,1), keepdim=True).detach().cpu().float())
                
            if return_pixelwise_selfattn_grad_norm_history:
                selfattn_grad_norm_history.append(torch.norm(grad[0], dim=(0,1), keepdim=True).detach().cpu().float())
        
        with torch.no_grad():        
            # Replace latents
            if i < (zt_replace_steps_min * ddim_steps):
                orig_latents = unedited_latents_history[-(i+1)]
                replacement_mask = attn_mask.mask_like(latents)
                latents = replacement_mask * latents + (1-replacement_mask) * orig_latents
    # np.concatenate([*np.concatenate([*torch.cat(mask_history).squeeze().cpu().numpy().reshape(5,10,16,16)],1)],1)
    # Get reconstruction
    edited_image_tensor = pipe.vae.decode(latents  / pipe.vae.config.scaling_factor ).sample
    random_path = generate_random_path('png')    
    edited_image_pil = torch_to_pil(edited_image_tensor)
    
    edited_image = save_image(
        edited_image_pil,
        os.path.join("generated_images", random_path) if save_copy else "/tmp/temp_image.png",
        args_dict
    )
    
    if return_mask_history:
        mask_history_img = plot_mask_history(mask_history)
        
    if return_z0_estimation_history:
        z0_estimation = plot_z0_history(history, pipe)
        
    if return_cross_attn_history:
        cross_attn_history = plot_cross_attn_history(attn_store, source_id, edited_id, cross_side)
        
    if return_pixelwise_epsilon_norm_history:
        pixelwise_epsilon_norm_history = plot_mask_history(epsilon_norm_history)
        
    if return_pixelwise_selfattn_grad_norm_history:
        pixelwise_selfattn_grad_norm_history = plot_mask_history(selfattn_grad_norm_history)
    
    plt.close()
    
    return (edited_image, vae_reconstruction, ddim_reconstruction, mask_history_img, z0_estimation, 
            cross_attn_history, prompt_latent_norm, 
            pixelwise_epsilon_norm_history, pixelwise_selfattn_grad_norm_history)