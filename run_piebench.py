import sys
import os

from piebench.piebench_dataset import PieBenchDataset 
from custom_inversion_functional import run
from sage_tools.visualization import save_image

import PIL.Image
from tqdm.auto import tqdm
import json

disable_tqdm=True


def main(
        label,
        zt_replace_steps_min,
        cross_replace_steps,
        cfg_value,
        self_latent_guidance_scale,
        attn_reweight,
        other_args,
        piebench_root,
        output_path = "./piebench_output"
         ):
    
    os.makedirs(output_path, exist_ok=True)
    
    json_path = os.path.join(piebench_root,"mapping_file.json")
    images_path = os.path.join(piebench_root,"annotation_images")
    ds = PieBenchDataset(json_path=json_path, images_path=images_path)

    # filename_template = "piebench_{task}_{i:04d}_{label}.png"
    def filename_template(label, image_path):
        # remove ext
        image_path = os.path.splitext(image_path)[0]
        filename_template = "{label}/annotation_images/{image_path}.png"
        return filename_template.format(label=label, image_path=image_path)
    
    for i,item in enumerate(tqdm(ds, desc="piebench", total=len(ds))):
        """item = {
            'editing_type_id': row.editing_type_id, #int
            'image': image, # PIL.Image
            'blend': blend, # ('word','word') or None
            'replace': replace, # ([words], [words]) or None
            'mask': mask, # np.array (H,W,1)
            'original_prompt': original_prompt, # str
            'editing_prompt': editing_prompt # str
        }"""
        image = item['image']
        blend = item['blend']
        blend_parsed = ','.join(blend) if blend else None
        replace = item['replace']
        # replace_parsed: "full sentence 1,full sentence 2"
        replace_parsed = (' '.join(replace[0]) + ',' + ' '.join(replace[1])) if replace else None
        original_prompt = item['original_prompt']
        editing_prompt = item['editing_prompt']
        editing_words = item['editing_words']
        # task_id = item['editing_type_id']
        image_path = item['image_path']
        
        
        
        file_path = os.path.join(
            output_path, 
            filename_template(label=label, image_path=image_path))
        if os.path.exists(file_path):
            continue
        
        args = dict(
            disable_tqdm=True,
            input_image = image,
            prompt_str = original_prompt,
            edited_prompt_str=editing_prompt,
            replace=replace_parsed,
            blend=blend_parsed,
            zt_replace_steps_min=zt_replace_steps_min,
            cross_replace_steps=cross_replace_steps,
            cfg_value=cfg_value,
            self_latent_guidance_scale=self_latent_guidance_scale,
            attn_scale = None,
            **json.loads(other_args)
        )
        
        # set zt_replace_steps_min to 0 if not blend
        if not blend:
            args['zt_replace_steps_min'] = 0
        # set cross_replace_steps to 0 if not replace
        if not replace:
            args['cross_replace_steps'] = 0
        if attn_reweight and attn_reweight != 1 and len(editing_words) > 0:
            args['attn_scale'] = ",".join(f"{w}:{attn_reweight}" for w in editing_words)
        
        
        args_copy = args.copy()
        del args_copy['input_image']
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            res = run(**args)
            edited_image = res[0]
            save_image(
                edited_image,
                file_path,
                args_copy
            )
        except Exception as e:
            str_e = str(e)
            print(i,label,str_e)
            with open(
                file_path.replace(".png", ".error"), 
                'w') as f:
                f.write(str_e)
                f.write("\n")
                f.write(json.dumps(args_copy))

import argparse        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--label", type=str, required=True)
    args.add_argument("--piebench_root", type=str, required=True)
    args.add_argument("--zt_replace_steps_min", type=float, default=0.8)
    args.add_argument("--cross_replace_steps", type=float, default=0.1)
    args.add_argument("--cfg_value", type=float, default=7.5)
    args.add_argument("--self_latent_guidance_scale", type=float, default=200)
    args.add_argument("--attn_reweight", type=float, default=1)
    args.add_argument("--other_args", type=str, default='{}')
    
    
    main(**args.parse_args().__dict__)
