import torch
import torch.nn.functional as nnf
from diffusers.models.attention_processor import Attention

import PIL

import json
from collections import namedtuple

import sage_tools.ptp_utils_updated as ptp_utils


def find_sublist_indices(lst, sublist):
    sub_len = len(sublist)
    indices = []

    for i in range(len(lst) - sub_len + 1):
        if (lst[i:i+sub_len] == sublist).all():
            indices.extend(range(i, i+sub_len))

    return indices

def get_attn_scale(attn_scale, prompt_tokens, tokenizer):
    scale = torch.ones(prompt_tokens.shape)
    if 'default_' in attn_scale:
        scale *= attn_scale['default_']
    for k,v in attn_scale.items():
        if k == 'default_':
            continue
        k = tokenizer(k, return_tensors="pt").input_ids[0,1:-1].to(prompt_tokens.device)
        idxs = find_sublist_indices(prompt_tokens, k)
        scale[idxs] = v
    return scale

class EditableAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, m_edit=None):
        self.m_edit = m_edit if m_edit is not None else self._empty_M_edit
        
    def _empty_M_edit(self, M):
        return M
        
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # # reshape (batch * head, ...) -> (batch, head, ...)
        M = attention_probs.reshape(batch_size,-1,*attention_probs.shape[1:])
        M = self.m_edit(M, is_cross)
        attention_probs = M.reshape(attention_probs.shape)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

TStore = namedtuple("TStore", ["cross", "self"])
TStorePair = namedtuple("TStorePair", ["reference", "current"])

def compose_editing(*funcs):
    def inner(M, is_cross):
        for f in funcs:
            M = f(M, is_cross)
        return M
    return inner

class AttnStore:
    def __init__(self, device, dtype, self_side, cross_side=None, threshold=0.3):
    # def __init__(self, device, dtype, threshold=0.3, self_side=SIDE//(8*2), cross_side=None):
        self.self_side = self_side
        self.cross_side = cross_side
        self.threshold = threshold
        self.device = device
        self.dtype = dtype
        
        self.t = 0
        self.store_ref = None
        self.clear()
        self._is_disabled = False
        
    def disable(self):
        self._is_disabled = True
        return self
        
    def enable(self):
        self._is_disabled = False
        return self
        
    def get_t(self, t):
        return TStorePair(
            reference=TStore(
                cross=self.store_ref.get((t, True), []),
                self=self.store_ref.get((t, False), []),
            ),
            current=TStore(
                cross=self.store.get((t, True), []),
                self=self.store.get((t, False), []),
            )
        )
        
    def set_t(self, t):
        self.t = t
        
    def clear(self):
        self.store = {}
        
    def clear_t(self, t):
        for k in [(t, True), (t, False)]:
            if k in self.store:
                del self.store[k]
    
    def fix(self):
        for k,v in self.store.items():
            self.store[k] = [e.detach()for e in v]
        self.store_ref = self.store
        self.clear()
    
    def store_attention(self, M, is_cross):
        empty_return = (None, None)
        if self._is_disabled:
            return empty_return
        
        side = int(M.shape[-2]**0.5)
        if (is_cross and side != self.cross_side) or (not is_cross and side != self.self_side):
            return empty_return
        
        # Store attention maps
        t_key = (self.t, is_cross)
        l = self.store.get(t_key, [])
        l.append(M.cpu())
        self.store[t_key] = l
        
        return (
            self.store_ref[t_key][len(l)-1] if self.store_ref is not None else None,
            self.store[t_key][len(l)-1]
        )
        
class AttnReplace:
    def __init__(self, device, dtype, from_ids, to_ids, cross_side, threshold=0.3, max_t=-1):
        self.cross_side = cross_side
        
        self.from_ids = from_ids.to(device)
        self.to_ids = to_ids.to(device)
        
        # average attention maps
        from_len = len(from_ids)
        to_len = len(to_ids)
        # cases
        # 1. from_len == to_len
        # 2. from_len = 1 && to_len > 1
        # 3. from_len > 1 && to_len = 1
        if from_len > 1 and to_len == 1:
            self.do_avg = True
        else:
            self.do_avg = False
        # 4. from_len > 1 && to_len > 1 && from_len != to_len
        if from_len != to_len and from_len != 1 and to_len != 1: 
            raise ValueError("from_len != to_len")
        
        self.threshold = threshold
        self.device = device
        self.dtype = dtype
        
        self.max_t = max_t
        
        self.set_t(0)
        self.store_ref = None
        self.clear()
        self._is_disabled = False
        
    @property
    def is_editing(self):
        return self.store_ref is not None
        
    def disable(self):
        self._is_disabled = True
        return self
        
    def enable(self):
        self._is_disabled = False
        return self
        
    def get_t(self, t):
        return TStorePair(
            reference=TStore(
                cross=self.store_ref.get(t, []),
            )
        )
        
    def set_t(self, t):
        self.t = t
        self.i = 0
        
    def clear(self):
        self.store = {}
        
    def clear_t(self, t):
        if t in self.store:
            del self.store[t]
        
    def fix(self):
        self.store_ref = self.store
        self.clear()
    
    def __call__(self, M, is_cross):
        if not is_cross or self._is_disabled or self.t < self.max_t: # check the impact of the first t steps vs last t steps
            return M
        
        side = int(M.shape[-2]**0.5)
        if side != self.cross_side:
            return M
        
        # Store attention maps
        t_key = self.t
        # self.store_ref[t_key][self.i].mean((0,1,3)).reshape(16,16)
        # M[...,self.to_ids].mean((0,1,3)).reshape(16,16)
        if self.is_editing:
            attn = self.store_ref[t_key][self.i].to(self.device, self.dtype)
            self.i += 1
            
            # attn_target = torch.zeros_like(M)
            # attn_target[...,self.to_ids] = attn
            # attn_mask = torch.ones_like(M)
            # attn_mask[...,self.to_ids] = 0. 
            
            # M = M * attn_mask + attn_target
            
            M[...,self.to_ids] = attn
            
        else:
            attn = M.index_select(-1,self.from_ids)
            if self.do_avg:
                attn = attn.mean(-1,keepdim=True)
            attn = attn.detach().cpu()
            l = self.store.get(t_key, [])
            l.append(attn)
            self.store[t_key] = l        
            
        return M
            
class AttnCost():
    def __init__(self, store=None, editing=None, replace=False):
        self.store = store
        self.editing = editing
        self.replace = replace
        
    def __call__(self, M, is_cross):
        # null_M, prompt_M = M.chunk(2)
        # self.store.store_attention(prompt_M, is_cross)
        ref_att, cur_attn = self.store.store_attention(M, is_cross)
        if ref_att is not None and self.replace:
            M = ref_att.to(M.device, M.dtype)
        if self.editing is not None:
            M = self.editing(M, is_cross)
        return M
    
class AttnMask:
    def __init__(self, tokenizer, prompts, replace, device, dtype, cross_side, threshold=0.3, max_num_tokens=77):
        self.side = cross_side
        self.threshold = threshold
        self.device = device
        self.dtype = dtype
        
        self.token_mask = self._parse(tokenizer, prompts, replace, max_num_tokens).to(device, dtype)
        self.reset()
        self.enable()
        
    def enable(self):
        self.is_enabled = True
        return self
    
    def disable(self):
        self.is_enabled = False
        return self
    
    def reset(self):
        self.store = torch.zeros(2, 1, self.side, self.side).to(self.device, self.dtype)
        self.mask_idx = None
    
    @staticmethod    
    def _parse(tokenizer, prompts, replace, max_num_tokens):
        alpha_layers = torch.zeros(len(prompts), 1, 1, max_num_tokens)
        for i, (prompt, words_) in enumerate(zip(prompts, replace)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, ind] = 1
        return alpha_layers
    
    def set_prompt_idx(self, i=None):
        self.mask_idx = i
    
    @torch.no_grad()    
    def __call__(self, M, is_cross):
        if not is_cross or not self.is_enabled:
            return M
        
        side = int(M.shape[-2]**0.5)
        if side != self.side:
            return M
        
        # Store attention maps
        if self.mask_idx != None:
            msk = self.token_mask[self.mask_idx].unsqueeze(0)
        else:
            msk = self.token_mask
        self.store += (M.detach() * msk).sum(-1).mean(1).reshape(-1,1,self.side,self.side)
        return M
        
        
    @property
    def mask(self):
        mask = nnf.max_pool2d(self.store, 3, stride=1, padding=1) 
        mask = mask / mask.amax((2,3), keepdim=True)
        mask = mask.gt(self.threshold).any(0, keepdim=True).to(self.device, self.dtype)
        return mask
    
    def mask_like(self, x):
        w,h = x.shape[-2:]
        mask = nnf.interpolate(self.mask, size=(w,h))
        mask.to(x.device, x.dtype)
        return mask
    
    # def blend(self, x_t):
    #     mask = self.mask
    #     mask = nnf.interpolate(mask, size=x_t.shape[-2:])
    #     return x_t[:1] + mask * (x_t - x_t[:1]) 
        

def load_image_and_info(path):
    im = PIL.Image.open(path)
    config_d = json.loads(im.info["config"])
    return im, config_d


class AttnReweight:
    def __init__(self, cross_attn_scale):
        self.cross_attn_scale = cross_attn_scale
        self.is_enabled = True
        
    def __call__(self, attn_map, is_cross):
        if is_cross:
            attn_map = attn_map * self.cross_attn_scale
        return attn_map
    
    def enable(self):
        self.is_enabled = True
        return self
        
    def disable(self):
        self.is_enabled = False
        return self
    
class AttnDummy:
    def __init__(self,*args,**kwargs):
        pass
    
    def __call__(self, attn_map, is_cross):
        return attn_map
    
    def enable(self):
        return self
    
    def disable(self):
        return self
    
    def fix(self):
        pass
    
    def reset(self):
        pass
    
    def set_prompt_idx(self, i=None):
        pass
    
    def set_t(self, t):
        pass
    
    
def get_index(prompt_str, substr, tokenizer):
    main_list = tokenizer.encode(prompt_str)
    sublist = tokenizer.encode(substr)[1:-1]
    """Find the starting index of sublist in main_list."""
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i:i+len(sublist)] == sublist:
            return torch.arange(i, i+len(sublist), dtype=torch.long)
    raise ValueError("Subsequence not found!")

def get_prompt_embedding(pipe, *prompt_str_list):
    prompt_input = pipe.tokenizer(
        prompt_str_list,  # Text with replaced word
        padding="max_length", # pad to max length (77) instead of to the input length
        truncation=True, # truncate to max length
        return_tensors="pt", # return pytorch tensors
        ).input_ids.to(pipe.device)
    prompt_embeds = pipe.text_encoder(
        prompt_input
        ).last_hidden_state
    return prompt_input, prompt_embeds


