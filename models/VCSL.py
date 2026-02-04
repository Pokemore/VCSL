import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .sfd_module import DConv

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class VCSL(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, args=None):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # --- Build Input Projections for Main Backbone ---
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.backbone = backbone
        self.crop_enabled = (args is not None)
        # Note: self.crop_backbone is removed. We use self.backbone for crops.

        if self.crop_enabled:
             print(f"[VCSL] Crop processing enabled (Sharing Backbone).")
        # --- MODIFICATION END ---

        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # self.tokenizer = RobertaTokenizerFast.from_pretrained('./weights/tokenizer')
        # self.text_encoder = RobertaModel.from_pretrained('./weights/text_encoder')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/data/26fa1f99/code/VCSL/weights/roberta-base')
        try:
            self.text_encoder = RobertaModel.from_pretrained('/data/26fa1f99/code/VCSL/weights/roberta-base', use_safetensors=False)
        except TypeError:
            self.text_encoder = RobertaModel.from_pretrained('/data/26fa1f99/code/VCSL/weights/roberta-base')

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # resize the bert output channel to transformer d_model
        self.crop_dropout_prob = getattr(args, "crop_dropout_prob", 0.5)
        #parameters p_drop can
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.fusion_module_text = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)

        # Optional sfd modules . Enable via `args.use_sfd`.
        use_sfd = False if args is None else getattr(args, "use_sfd", False)
        if use_sfd:
            sfd_atoms = getattr(args, "sfd_atoms", 512)
            sfd_alpha = getattr(args, "sfd_alpha", 0.8)
            try:
                backbone_outs = list(backbone.num_channels[-3:])
            except Exception:
                backbone_outs = []
            self.sfd_modules = nn.ModuleList()
            for ch in backbone_outs:
                self.sfd_modules.append(DConv(in_channels=ch, alpha=sfd_alpha, atoms=sfd_atoms))
        else:
            self.sfd_modules = None
        # --- Build Crop Input Projections ---
        # Even though we share the backbone, we maintain separate projection layers
        # for crops to allow the model to adapt specific semantics for cropped regions.
        if self.crop_enabled:
            try:
                # Use self.backbone.num_channels directly since we share it
                crop_outs = list(self.backbone.num_channels[-3:])
            except Exception:
                crop_outs = []

            if crop_outs:
                crop_proj_list = []
                for in_ch in crop_outs:
                    crop_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                # if user requested more feature levels than backbone provides
                for _ in range(max(0, num_feature_levels - len(crop_outs))):
                    crop_proj_list.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                self.crop_input_proj = nn.ModuleList(crop_proj_list)
            else:
                self.crop_input_proj = None
        else:
            self.crop_input_proj = None

    def forward(self, samples: NestedTensor, captions, targets):

        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)

        # --- MODIFICATION START: Crop Logic using Shared Backbone ---
        crop_features_list = None
        crop_pos = None
        has_crop_capability = self.crop_enabled and targets is not None
        if self.training and has_crop_capability:
            # 生成随机数，如果小于dropout概率，则跳过crop处理
            if torch.rand(1).item() < self.crop_dropout_prob:
                # 直接设置has_crop_capability为False，跳过后续crop处理
                has_crop_capability = False




        try:
            # Check if crop is enabled and targets exist
            if has_crop_capability:
                b_tmp = len(captions)
                n_total = samples.tensors.shape[0]
                t_tmp = max(1, n_total // max(1, b_tmp))

                def _extract_box_from_target(target, H, W):
                    keys = ['referent_box', 'referent_bbox', 'referent_boxes', 'ref_box', 'ref_bbox', 'ref_boxes', 'boxes', 'bbox', 'gt_box']
                    for k in keys:
                        if k in target:
                            val = target[k]
                            if torch.is_tensor(val):
                                v = val.detach().cpu()
                                if v.numel() == 4:
                                    arr = v.view(-1).numpy().tolist()
                                elif v.ndim == 2 and v.shape[0] >= 1:
                                    arr = v[0].numpy().tolist()
                                else:
                                    continue
                            elif isinstance(val, (list, tuple)):
                                if len(val) == 4 and all(isinstance(x, (int, float)) for x in val):
                                    arr = list(val)
                                elif len(val) >= 1 and isinstance(val[0], (list, tuple)) and len(val[0]) == 4:
                                    arr = list(val[0])
                                else:
                                    continue
                            else:
                                continue

                            if max(arr) <= 1.0:
                                if arr[2] <= 1.0 and arr[3] <= 1.0:
                                    cx, cy, ww, hh = arr
                                    x1 = (cx - ww / 2.0) * W
                                    y1 = (cy - hh / 2.0) * H
                                    x2 = (cx + ww / 2.0) * W
                                    y2 = (cy + hh / 2.0) * H
                                else:
                                    x1, y1, x2, y2 = arr
                                    x1 *= W; x2 *= W; y1 *= H; y2 *= H
                            else:
                                if arr[2] > arr[0] and arr[3] > arr[1]:
                                    x1, y1, x2, y2 = arr
                                else:
                                    cx, cy, ww, hh = arr
                                    x1 = cx - ww / 2.0
                                    y1 = cy - hh / 2.0
                                    x2 = cx + ww / 2.0
                                    y2 = cy + hh / 2.0

                            x1 = max(0, min(W - 1, float(x1)))
                            y1 = max(0, min(H - 1, float(y1)))
                            x2 = max(0, min(W - 1, float(x2)))
                            y2 = max(0, min(H - 1, float(y2)))
                            if x2 <= x1 or y2 <= y1:
                                cx = (x1 + x2) / 2.0
                                cy = (y1 + y2) / 2.0
                                x1 = max(0, cx - 8); x2 = min(W - 1, cx + 8)
                                y1 = max(0, cy - 8); y2 = min(H - 1, cy + 8)
                            return [int(x1), int(y1), int(x2), int(y2)]
                    return None

                H = samples.tensors.shape[-2]
                W = samples.tensors.shape[-1]
                crops = []
                for i, target in enumerate(targets):
                    box = _extract_box_from_target(target, H, W)
                    if box is None:
                        crops.append(None)
                        continue
                    x1, y1, x2, y2 = box
                    idx = i * t_tmp
                    img = samples.tensors[idx]
                    x1c = max(0, min(W - 1, int(x1)))
                    y1c = max(0, min(H - 1, int(y1)))
                    x2c = max(1, min(W, int(x2)))
                    y2c = max(1, min(H, int(y2)))
                    if x2c <= x1c or y2c <= y1c:
                        crops.append(None)
                        continue
                    crop = img[:, y1c:y2c, x1c:x2c]
                    crop_resized = F.interpolate(crop.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]
                    crops.append(crop_resized)

                if any(c is not None for c in crops):
                    first_valid = next(c for c in crops if c is not None)
                    batch_crops = []
                    for c in crops:
                        if c is None:
                            batch_crops.append(torch.zeros_like(first_valid))
                        else:
                            batch_crops.append(c)
                    batch_crops = torch.stack(batch_crops, dim=0).to(samples.tensors.device)
                    crop_nested = nested_tensor_from_tensor_list(batch_crops)

                    # CRITICAL CHANGE: Use self.backbone instead of self.crop_backbone
                    # Both full images and crops share the same backbone instance.
                    crop_features_list, crop_pos = self.backbone(crop_nested)
                else:
                    crop_features_list = None

        except Exception as e:
            # print(f"Crop processing failed: {e}") # Optional debugging
            crop_features_list = None
        # --- MODIFICATION END ---

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples)

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
                pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            # t: num_frames -> 1
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        text_word_initial_features = text_word_features
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            if self.sfd_modules is not None:
                src_sfd = self.sfd_modules[l](src)

    # shape checking
                if src_sfd.shape != src.shape:
                    raise ValueError(f"Shape Mismatch! Input: {src.shape}, SFD Output: {src_sfd.shape}")

                src = src + src_sfd
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                         memory=src_proj_l,
                                                         memory_key_padding_mask=mask,
                                                         pos=pos_l,
                                                         query_pos=None)

            # If crop features exist for this level, project and fuse them with text as well.
            if crop_features_list is not None and getattr(self, 'crop_input_proj', None) is not None:
                try:
                    # align index to last-3 mapping
                    crop_feat = crop_features_list[-3:][l]
                    crop_src, crop_mask = crop_feat.decompose()
                    crop_proj = self.crop_input_proj[l](crop_src)
                    crop_proj = rearrange(crop_proj, '(b t) c h w -> (t h w) b c', b=b, t=t)
                    crop_mask = rearrange(crop_mask, '(b t) h w -> b (t h w)', b=b, t=t)
                    crop_pos_l = None
                    if crop_pos is not None and len(crop_pos) >= 3:
                        crop_pos_l = crop_pos[-3:][l]
                        crop_pos_l = rearrange(crop_pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                    # fuse text_word_features with crop memory
                    text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                                 memory=crop_proj,
                                                                 memory_key_padding_mask=crop_mask,
                                                                 pos=crop_pos_l,
                                                                 query_pos=None)

                    # Note: We skip fusing crop_proj back with text_word_initial_features here if not needed,
                    # but logic is kept consistent with original code flow if previously enabled.
                    # crop_proj = self.fusion_module(...)

                    crop_proj = rearrange(crop_proj, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                except Exception:
                    pass

            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_initial_features,
                                            memory_key_padding_mask=text_word_masks,
                                            pos=text_pos,
                                            query_pos=None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # take backbone output and apply RD before projection
                    src = features[-1].tensors

                    if self.sfd_modules is not None:
                        try:
                            src_sfd = self.sfd_modules[-1](src)
                            if src_sfd.shape == src.shape:
                                src = src + src_sfd
                            else:
                                try:
                                    c = src.shape[1]
                                    src = src + src_sfd[:, :c, ...]
                                except Exception:
                                    src = src_sfd
                        except Exception:
                            pass
                    src = self.input_proj[l](src)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                             memory=src,
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_l,
                                                             query_pos=None)
                src = self.fusion_module(tgt=src,
                                         memory=text_word_initial_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None
                                         )
                # fuse crop features for extra pyramid levels when available
                if crop_features_list is not None and getattr(self, 'crop_input_proj', None) is not None:
                    try:
                        crop_levels = crop_features_list[-(self.num_feature_levels):]
                        crop_idx = min(len(crop_levels) - 1, l - _len_srcs)
                        crop_feat = crop_levels[crop_idx]
                        crop_src, crop_mask = crop_feat.decompose()
                        crop_proj = self.crop_input_proj[crop_idx](crop_src)
                        crop_proj = rearrange(crop_proj, '(b t) c h w -> (t h w) b c', b=b, t=t)
                        crop_mask = rearrange(crop_mask, '(b t) h w -> b (t h w)', b=b, t=t)
                        crop_pos_l = None
                        if crop_pos is not None and len(crop_pos) >= 1:
                            crop_pos_l = rearrange(crop_pos[min(crop_idx, len(crop_pos) - 1)], '(b t) c h w -> (t h w) b c', b=b, t=t)

                        text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                                     memory=crop_proj,
                                                                     memory_key_padding_mask=crop_mask,
                                                                     pos=crop_pos_l,
                                                                     query_pos=None)

                        crop_proj = self.fusion_module(tgt=crop_proj,
                                                       memory=text_word_initial_features,
                                                       memory_key_padding_mask=text_word_masks,
                                                       pos=text_pos,
                                                       query_pos=None)
                        crop_proj = rearrange(crop_proj, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                    except Exception:
                        pass
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds)


        out = {}
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)  # NestedTensor
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RobertaPoolout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91  # for coco
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = VCSL(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        args=args
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors