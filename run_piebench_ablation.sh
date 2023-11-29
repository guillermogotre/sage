python run_piebench.py --label abblationcrossguidance_i --zt_replace_steps_min 0 --cross_replace_steps 0 --other_args '{"cross_layer": 1, "reconstruction_type": "cross"}'
python run_piebench.py --label abblationz0guidance_ii --zt_replace_steps_min 0 --cross_replace_steps 0 --self_latent_guidance_scale 1 --other_args '{"reconstruction_type": "z0"}'
python run_piebench.py --label abblationselfreplace_iii --zt_replace_steps_min 0 --cross_replace_steps 0 --other_args '{"reconstruction_type": "replace"}'
python run_piebench.py --label abblationbase_iv --zt_replace_steps_min 0 --cross_replace_steps 0
python run_piebench.py --label abblationcrossreplace_v --zt_replace_steps_min 0
python run_piebench.py --label sage_vi