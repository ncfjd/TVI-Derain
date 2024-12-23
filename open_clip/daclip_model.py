from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy

from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
#####################################################

from .tokenizer import tokenize
from collections import OrderedDict
#########################################################################################################
# class CustomCLIP(nn.Module):
#     def __init__(self,clip_model):
#         super().__init__()
#         self.prompt_learner =PromptLearner(clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.visual = clip_model.visual
#         self.text = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.meta_net = self.prompt_learner.meta_net
#         self.ori_embedding = self.prompt_learner.text_features
#
#
#
#
#
# class PromptLearner(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.use_prompt_stage = True
#         if self.use_prompt_stage:
#
#             n_ctx = 4
#             ctx_dim = clip_model.ln_final.weight.shape[0]
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             self.ctx = nn.Parameter(ctx_vectors)
#             self.n_ctx = n_ctx         #4 512
#         # prompts = ["a photo of a rainy","a photo of a snowy","A picture with some raindrops"]
#         # tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
#         prompts="a photo of a rainy"
#         prompts1 = "a photo of a snowy"
#         prompts2 = "a picture with some raindrops"
#
#         tokenized_prompts = tokenize(prompts)   # 1 77
#         tokenized_prompts1 = tokenize(prompts1)  # 1 77
#         tokenized_prompts2 = tokenize(prompts2)  # 1 77
#         ########################
#         prompts_ =tokenized_prompts
#         with torch.no_grad():
#             text_features = clip_model.encode_text( prompts_)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#
#         self.text_features = text_features
#         #######################
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts)    #1 77 512
#             embedding1 = clip_model.token_embedding(tokenized_prompts1)  # 1 77 512
#             embedding2 = clip_model.token_embedding(tokenized_prompts2)  # 1 77 512
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#         self.register_buffer("token_prefix1", embedding1[:, :1, :])  # SOS
#         self.register_buffer("token_suffix1", embedding1[:, 1 + n_ctx:, :])  # CLS, EOS
#         self.register_buffer("token_prefix2", embedding2[:, :1, :])  # SOS
#         self.register_buffer("token_suffix2", embedding2[:, 1 + n_ctx:, :])  # CLS, EOS
#         self.n_ctx = n_ctx
#         #self.tokenized_prompts = tokenized_prompts
#         self.tokenized_prompts = torch.cat((tokenized_prompts,tokenized_prompts1,tokenized_prompts2),dim=1)
# ######################################
#         self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(512, 512)),
#             ("relu", nn.ReLU(inplace=True))
#             # ("linear2", nn.Linear(128, 512))
#         ]))
#         ############################
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx1 = ctx.unsqueeze(0).expand(22, -1, -1)    #1 4 512
#         prefix = self.token_prefix
#         prefix = prefix.expand(22, -1, -1)
#         suffix = self.token_suffix
#         suffix = suffix.expand(22, -1, -1)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 ctx1,     # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#
#         prefix = self.token_prefix1
#         prefix = prefix.expand(22, -1, -1)
#         suffix = self.token_suffix1
#         suffix = suffix.expand(22, -1, -1)
#
#         prompts1 = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 ctx1,  # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#         ctx = ctx.unsqueeze(0).expand(20, -1, -1)  # 1 4 512
#         prefix = self.token_prefix2
#         prefix = prefix.expand(20, -1, -1)
#         suffix = self.token_suffix2
#         suffix = suffix.expand(20, -1, -1)
#
#         prompts2 = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 ctx,  # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#         prompts=torch.cat((prompts,prompts1,prompts2),0)
#         return prompts
#
# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.token_embedding = clip_model.token_embedding
#
#     def forward(self, prompts,tokenized_prompts):
#         cast_dtype = self.transformer.get_cast_dtype()
#         x = prompts + self.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).to(cast_dtype)
#
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         # x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
#         #
#         # x = x + self.positional_embedding
#         # x = x.permute(1, 0, 2)  # NLD -> LND
#         # x = self.transformer(x, output_hiddens=output_hiddens, control=control)
#         # if output_hiddens:
#         #     x, hiddens = x
#         # x = x.permute(1, 0, 2)  # LND -> NLD
#         # x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         # # take features from the eot embedding (eot_token is the highest number in each sequence)
#         # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#         # if output_hiddens:
#         #     return x, hiddens
#         # return x
#
#         return x
#
# #######################################################################################################
# class Adapter(nn.Module):
#     def __init__(self, c_in, reduction=4):
#         super(Adapter, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, c_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // reduction, c_in, bias=False),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         return x
#

class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.clip = clip_model
        # self.clip1 = clip_model1
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

        # self.adapter = Adapter(512, 4)

        # self.clip1.text.transformer = ControlTransformer(self.clip1.text.transformer)

        # self.prompt_learner = clip_model1.prompt_learner
        # self.text_encoder = clip_model1.text
        # self.ori_embedding = self.prompt_learner.text_features
        # self.meta_net = self.prompt_learner.meta_net
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #################################################################################
        # self.transformer = clip_model.transformer
        # self.token_embedding = clip_model.token_embedding
        # self.positional_embedding = clip_model.positional_embedding
        # self.ln_final = clip_model.ln_final
        # self.text_projection = clip_model.text_projection
        # ##################################################################################


    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(),
                                                self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(),
                                    self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)

    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False
            ###########################
        # for param in self.visual.parameters():
        #     param.requires_grad = False
            #############################

        # for name, param in self.clip1.named_parameters():
        #     if "prompt_learner" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, hiddens = self.visual_control(image, output_hiddens=True,control=None)
            image_features = self.clip.visual(image, control=hiddens)
            ######################################################################################
            # x = self.adapter(image_features)
            # ratio = 0.2
            # image_features = ratio * x + (1 - ratio) * image_features

            #######################################################################################
            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        image_features = self.clip.encode_text(text, normalize)
        # x = self.adapter(image_features)
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features

        return image_features

    # def text_encoder(self, text):
    #     return self.clip1.text(text)
    # #############################################################################################
    # def text_encoder(self, prompts,tokenized_prompts):
    #     return self.clip1.text(prompts,tokenized_prompts)
    # #############################################################################################
    # def encode_text(self, text, normalize: bool = False,output_hiddens=False ,control=False):
    #     return self.clip.encode_text(text, normalize,output_hiddens, control)
    # def text_encoder(self, text,output_hiddens=True):
    #     return self.clip1.text(text, output_hiddens)
    # def encode_image1(self, image, normalize: bool = False):
    #     GQ_features = self.visual(image)
    #     return F.normalize(GQ_features, dim=-1) if normalize else GQ_features
    ##############################################################

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)  # 64 77     64 77

        image_features, image_degra_features = self.encode_image(image, control=True,
                                                                 normalize=True) if image is not None else None

        text_features = self.encode_text(caption, normalize=True) if text is not None else None

        ##########################################################################
        # GQ_features = self.encode_image1(image , normalize=True) if image is not None else None
        # prompts = self.clip1.prompt_learner()
        # tokenized_prompts = self.clip1.tokenized_prompts
        # text_degra_features = self.text_encoder(prompts, tokenized_prompts)
        # text_degra_features = F.normalize(text_degra_features, dim=-1)
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        # text_features_old = self.ori_embedding
        # text_features_old = F.normalize(text_features_old, dim=-1)
        # text_features_old = text_features_old.cuda()

        # score = cos(text_degra_features, text_features_old)
        # score = 1.0 - torch.mean(score)

        # text_degra_features,text_hiddens = self.text_encoder(degradation,output_hiddens=True) if degradation is not None else None
        # text_features = self.encode_text(caption,normalize=True,output_hiddens=False, control=text_hiddens) if text is not None else None

        ##########################################################################
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }


