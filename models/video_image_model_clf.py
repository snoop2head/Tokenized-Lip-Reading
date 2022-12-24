import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


class VideoImageModelForClassification(TransformerWrapper):
    def __init__(
        self,
        *,
        model,
        num_tokens,
        max_seq_len,
        attn_layers,
        channels=3,
        image_size=(29, 420),
        patch_size=(1, 420),
        dim=512,
        post_emb_norm=False,
        emb_dropout=0.0,
        num_classes=500
    ):
        super(TransformerWrapper, self).__init__()

        self.video_cnn = model.module.video_cnn.cuda()
        self.image_to_patch_embed = model.module.image_to_patch_embed
        self.cls_token = model.module.cls_token
        self.sep_token = model.module.sep_token
        self.attn_layers = attn_layers

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.attn_layers = attn_layers
        patch_height, patch_width = pair(patch_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(
        self, 
        img, 
        video,
        mask = None,
        return_embeddings = False,
        return_mems = False,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        return_hiddens = True,
        **kwargs        
    ):
        
        # https://github.com/lucidrains/x-transformers/blob/4b46febcc9442ad07e9ba6ea1dc4f9839975b27d/x_transformers/x_transformers.py#L1145
        # concatenate token embeddings
        img = self.image_to_patch_embed(img) # batch_size, 29, 512
        video = self.video_cnn(video) # batch_size, 29, 512

        # add special tokens
        batch_size, seq_len, hidden_size = img.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size) # repeat cls token for each batch
        sep_tokens = repeat(self.sep_token, '1 1 d -> b 1 d', b = batch_size) # repeat sep token for each batch
        x = torch.cat((cls_tokens, img, sep_tokens, video), dim=1) # batch_size, (1 + 29 + 1 + 29), 512
        
        # no emb norm
        x = self.emb_dropout(x)

        # attention layers
        x = self.attn_layers(x, mask = mask, mems = mems, **kwargs)

        # attention layers and hidden_states
        x = x.mean(dim = -2) # head similar to https://github.com/lucidrains/x-transformers/blob/4b46febcc9442ad07e9ba6ea1dc4f9839975b27d/x_transformers/x_transformers.py#L1120-L1143

         # multi-sample classifier
        return self.classifier(x)