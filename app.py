from custom_inversion_functional import run
import gradio as gr
import torch
import os
from functools import partial

# get GRADIO_PORT environment variable
gradio_port = os.environ.get('GRADIO_PORT', None)
if gradio_port is not None:
    gradio_port = int(gradio_port)

# false if SAGE_MEMORY_INTENSIVE exists
low_memory = not os.getenv("SAGE_MEMORY_INTENSIVE", False)

is_cuda_available = torch.cuda.is_available()

def gradio_main():
    with gr.Blocks(
        css='static/custom_inversion_functional.css', js='static/custom_inversion_functional.js') as demo:
        
        gr.Markdown("# SAGE: Self-Attention Guidance for Image Editing")
        if not is_cuda_available:
            gr.HTML("""
                    <div class="alert alert-warning" role="alert" style="color:red">
                        <strong>Warning!</strong> No GPU detected. We strongly advise the user against running it on CPU.
                        <br>
                        You can visit <a href="https://github.com/guillermogotre/sage">SAGE webpage</a> for more information about how to run it on GPU for free.
                        <br>
                        If you're running this on HF, you can duplicate the project and launch it on a GPU server.
                        <br>
                        You can also run it on Google Colab.
                        <br>
                        <a href="https://colab.research.google.com/github/guillermogotre/sage/blob/main/app_colab.ipynb">
                            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
                        </a>
                    </div>
                    """)
        gr.Markdown("""
                    Recomended Settings: 
                    - SD1.4 & SD1.5 - 512x512, Self-attn layer: 1, Cross-attn layer: 2,  Self-attn Guidance Scale: 200
                    - SD2.1 - 768x768, Self-attn layer: 2, Cross-attn layer: 2, Self-attn Guidance Scale: 50
                    """)
        

        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image", height=512, width=512)
            output_image = gr.Image(label="Output Image", type="pil",elem_classes=["gr-image-output"], height=512, width=512)
            
        with gr.Column():
            button = gr.Button("Run")
            
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    prompt = gr.Textbox(label="Prompt")
                    edited_prompt = gr.Textbox(label="Edited Prompt")
                    attention_scale = gr.Textbox(label="Attention Scale", placeholder="token1:0.5,token2:0.5")
                    replace = gr.Textbox(
                        label="Replace", placeholder="token1,token2",
                        info="Replace tokens separated by spaces in the prompt with tokens in the edited prompt (1:1, N:1, or 1:N)")
                    blend = gr.Textbox(label="Blend", placeholder="token1,token2",
                                       info="Blend tokens separated by spaces in the prompt with tokens in the edited prompt (N:N)")
                
                with gr.Group():
                    z_t_replacement_steps_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Blend steps", value=0., 
                                                          info="Blend steps as a fraction of total steps")
                    cross_replace_steps = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Cross Replace Steps", value=0.0, 
                                                    info="Cross Replace Steps as a fraction of total steps")
                    
                    self_attention_guidance_mask_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Self-attention guidance mask min", value=1., interactive=False)
                    classifier_free_guidance_mask_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Classifier-free guidance mask min", value=1., interactive=False)
            
            with gr.Column(): 
                with gr.Group():   
                    model_id = gr.Dropdown(["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"], label="Model ID", value="CompVis/stable-diffusion-v1-4", )
                    with gr.Row():
                        self_layer = gr.Dropdown([1,2,3], label="Self-attn layer", value=1)
                        cross_layer = gr.Dropdown([1,2,3], label="Cross-attn layer", value=2)
                        reconstruction_type = gr.Dropdown([
                                ("Self-Attn guidance", "sage"),
                                ("Cross-Attn guidance", "cross"),
                                ("z0 estimation", "z0"),
                                ("Replace self-attention", "replace"),
                                
                            ], label="Reconstruction", value="sage")
    
                with gr.Row():
                    cfg_value = gr.Number(label="Classifier-free Guidance", value=7.5)
                    self_latent_guidance_scale = gr.Number(label="Self-Attention Guidance", value=200.)
                    max_steps = gr.Number(label="Max Steps", value=40)
                    
                    use_trailing = gr.Checkbox(label="Use Trailing", value=False)
                    use_vae_mean = gr.Checkbox(label="Use VAE Mean", value=True)
                    use_monotonical_scale = gr.Checkbox(label="Use Monotonical Scale", value=True)
                    
                    ddim_steps = gr.Number(label="DDIM Steps", value=50)
                    # loss_scale = gr.Number(label="Loss Scale", value=5e2, info="FP16:500 FP32:250")
                    loss_scale = gr.Number(label="Loss Scale", value=None, info="FP16:500 FP32:300", interactive=False)
                    noise_alpha = gr.Number(label="Noise Alpha", value=0.0)
                    
                    seed = gr.Number(label="Seed", value=8888)
                    side = gr.Number(label="Side", value=512)
                    loss_type = gr.Dropdown(["mae", "mse"], label="Loss Type", value="mae")
                    
                with gr.Accordion(label="Return History"):
                    with gr.Row():
                        return_vae_rec = gr.Checkbox(label="VAE Reconstruction", value=False, elem_classes=["history-checkbox"])
                        return_ddim_inv_rec = gr.Checkbox(label="DDIM Inverse Reconstruction", value=False, elem_classes=["history-checkbox"])
                        return_mask_history = gr.Checkbox(label="Mask History", value=False, elem_classes=["history-checkbox"])
                        return_z0_estimation_history = gr.Checkbox(label="z0 estimation", value=False, elem_classes=["history-checkbox"])
                        return_cross_attn_history = gr.Checkbox(label="Cross Attn History", value=False, elem_classes=["history-checkbox"], interactive=(not low_memory), info="Not available on low memory environments" if low_memory else None)
                        return_pixelwise_epsilon_norm_history = gr.Checkbox(label="Pixelwise Epsilon Norm History", value=False, elem_classes=["history-checkbox"])
                        return_pixelwise_selfattn_grad_norm_history = gr.Checkbox(label="Pixelwise Selfattn Grad Norm History", value=False, elem_classes=["history-checkbox"])
                    # Create a function to activate all checkboxes
                    def foo():
                        pass
                        
                    # Create a button to activate all checkboxes
                    with gr.Row():
                        activate_button = gr.Button(value="Activate All Checkboxes", elem_id="activate_button")
                        deactivate_button = gr.Button(value="Deactivate All Checkboxes", elem_id="deactivate_button")
                        activate_button.click(foo,None,None)
                        deactivate_button.click(foo,None,None)
                        
                        
        prompt_latent_norm = gr.Textbox(label="Prompt Latent Norm")
        with gr.Row():    
            vae_reconstruction = gr.Image(label="VAE Reconstruction", type="pil", elem_classes=["gr-image-output"])
            ddim_reconstruction = gr.Image(label="DDIM Inversion Reconstruction", type="pil", elem_classes=["gr-image-output"])
            mask_history = gr.Image(label="Mask History", type="pil", elem_classes=["gr-image-output"])
            z0_estimation_history = gr.Image(label="z0 estimation", type="pil", elem_classes=["gr-image-output"])
            cross_attn_history = gr.Image(label="Cross Attn History", type="pil", elem_classes=["gr-image-output"])
            pixelwise_epsilon_norm_history = gr.Image(label="Pixelwise Epsilon Norm History", type="pil", elem_classes=["gr-image-output"])
            pixelwise_selfattn_grad_norm_history = gr.Image(label="Pixelwise Selfattn Grad Norm History", type="pil", elem_classes=["gr-image-output"])
        
            
        gr.HTML('''<div id="myModal" class="modal">
                            <img class="modal-content" id="img01">
                        </div>''')
            

        button.click(
            partial(
                run,
                low_memory=low_memory),
            inputs=[
                input_image, prompt, edited_prompt, attention_scale, replace, blend, seed, side, ddim_steps, use_vae_mean, model_id, 
                use_trailing, self_latent_guidance_scale, cross_replace_steps, self_attention_guidance_mask_min, 
                classifier_free_guidance_mask_min, z_t_replacement_steps_min, cfg_value, loss_scale, max_steps, use_monotonical_scale, noise_alpha, loss_type,
                self_layer,cross_layer,reconstruction_type,
                return_vae_rec, return_ddim_inv_rec, return_mask_history, return_z0_estimation_history, return_cross_attn_history, 
                return_pixelwise_epsilon_norm_history, return_pixelwise_selfattn_grad_norm_history
                
            ],
            outputs=[output_image, vae_reconstruction, ddim_reconstruction, mask_history, z0_estimation_history, cross_attn_history, prompt_latent_norm, pixelwise_epsilon_norm_history, pixelwise_selfattn_grad_norm_history]
        )
    # demo.launch(debug=True, server_port=8088)
    demo.launch(debug=True, share=gradio_port is None, server_port=gradio_port)
    
if __name__ == "__main__":    
    gradio_main()