import io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nnf
import PIL
import PIL.Image
from PIL.PngImagePlugin import PngInfo
import json
import random
import string



def generate_random_path(extension: str, length: int = 10) -> str:
    if not extension.startswith('.'):
        extension = '.' + extension
    chars = string.ascii_letters + string.digits
    random_path = ''.join(random.choice(chars) for _ in range(length)) + extension
    return random_path

def latent_to_torch(latent, vae):
    image_rec = vae.decode(latent  / vae.config.scaling_factor ).sample
    return image_rec

def torch_to_pil(torch_img):
    return PIL.Image.fromarray((torch_img*127.5+127.5).clamp(0,255).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8).squeeze())

def save_image(pil_img, path, config_dict):
    # im = torch_to_pil(torch_img)
    im = pil_img
    meta = PngInfo()
    config_d = config_dict
    meta.add_text("config", json.dumps(config_d))
    im.save(path, pnginfo=meta)
    return im

def load_image_and_info(path):
    im = PIL.Image.open(path)
    config_d = json.loads(im.info["config"])
    return im, config_d


def plt_figure_to_PIL(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Load the BytesIO object into a PIL image
    img = PIL.Image.open(buf)
    return img

def plot_mask_history(mask_history, rows=5, cols=None):
    n = len(mask_history)
    cols = cols or int(n // rows + np.ceil(n % rows))
    
    assert rows * cols >= n
    
    fig,axs = plt.subplots(rows,cols,dpi=100,figsize=(2*cols,2*rows))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(mask_history[i].float().cpu().numpy().squeeze())
        ax.set_title(f"{i}")
        ax.axis("off")
    return plt_figure_to_PIL(fig)

@torch.no_grad()
def plot_z0_history(history, pipe, img_size=128):
    res = []
    for k in ['uncond', 'cfg', 'cond']:
        res.append(torch.cat([
            nnf.interpolate(latent_to_torch(e.cuda(), pipe.vae), size=img_size, mode='bilinear', align_corners=False)
            for e in history[k]], dim=0))
    # res [3, 50?, 3, img_size, img_size]
    # transform to [img_size*3, img_size*50, 3]
    res = torch.cat(res,dim=2)
    res = torch.cat([*res],dim=2)
    return torch_to_pil(res.unsqueeze(0))
        
    
def plot_z0_history_plt(history, pipe, from_i=0, to_i=49, rows=3, n=18):
    assert n % rows == 0
    n_per_row = n // rows
    fig, axs = plt.subplots(3*rows, n//rows, figsize=(3*n_per_row, 5*rows))
    for i,idx in enumerate(np.linspace(from_i, to_i, n, dtype=int)):
        r = i // n_per_row
        c = i % n_per_row
        im_uncond = np.array(torch_to_pil(pipe.vae.decode(history['uncond'][idx].cuda()  / pipe.vae.config.scaling_factor ).sample))
        im_cond = np.array(torch_to_pil(pipe.vae.decode(history['cond'][idx].cuda()  / pipe.vae.config.scaling_factor ).sample))
        im_cfg = np.array(torch_to_pil(pipe.vae.decode(history['cfg'][idx].cuda()  / pipe.vae.config.scaling_factor ).sample))
        axs[r*3,c].imshow(im_uncond)
        axs[r*3+1,c].imshow(im_cfg)
        axs[r*3+2,c].imshow(im_cond)
        # axs[r*3,c].set_title(f"{idx}")
        
        if c == 0:
            axs[r*3,c].set_ylabel("p_from")
            axs[r*3+1,c].set_ylabel("CFG")
            axs[r*3+2,c].set_ylabel("p_to")
    # for ax in axs.flatten():
    #     ax.axis('off')
    for ax in axs.flatten():
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        
    return plt_figure_to_PIL(fig)

@torch.no_grad()
def plot_cross_attn_history(attn_store, source_id, edited_id, cross_side, upscale=4):
    source_attn = [torch.cat(v)[...,source_id].mean((0,1,3)).reshape(cross_side,cross_side) for k,v in attn_store.store_ref.items() if k[1]]
    target_attn = [torch.cat(v)[...,edited_id].mean((0,1,3)).reshape(cross_side,cross_side) for k,v in attn_store.store.items() if k[1]]
    for e in source_attn:
        e /= e.mean()
    for e in target_attn:
        e /= e.mean()
    result = torch.cat([torch.cat(source_attn,dim=1),torch.cat(target_attn,dim=1)],dim=0)
    # result to [-1,1]
    result = (result - result.min()) / (result.max() - result.min()) * 2 - 1
    result = result[None,None,...].float()
    # resize
    result = nnf.interpolate(result, scale_factor=upscale, mode='nearest')
    return torch_to_pil(result)