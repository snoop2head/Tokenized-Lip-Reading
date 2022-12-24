import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers import TransformerWrapper, Encoder, Decoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from components import *
from models import *


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

class VideoImageModel(TransformerWrapper):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        channels=3,
        image_size=(29, 420),
        patch_size=(1, 420),
        dim=512,
        post_emb_norm=False,
        emb_dropout=0.0,
    ):
        super(TransformerWrapper, self).__init__()

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.attn_layers = attn_layers
        patch_height, patch_width = pair(patch_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.video_cnn = VideoCNN()
        self.image_to_patch_embed = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(channels * patch_height * patch_width, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim))

        self.gpt = TransformerWrapper(
            num_tokens=len(processor.tokenizer.get_vocab()),
            max_seq_len=60,
            attn_layers=Decoder(
                dim=dim,
                depth=6,
                heads=8,
                cross_attend=True,  # https://github.com/lucidrains/x-transformers#miscellaneous
                only_cross=False,  # only cross attention makes embeddings not able to self-attend
                attn_dropout=0.0,  # dropout post-attention
                ff_dropout=0.3,  # feedforward dropout
                rotary_pos_emb=True,  # turns on rotary positional embeddings
                use_rmsnorm=True,
                ff_glu=True,
            ),
        )

        self.decoder = AutoregressiveWrapper(
            self.gpt, mask_prob=0.15  # in paper, they use 15%, same as BERT
        )

    def forward(
        self,
        img,
        video,
        mask=None,
        input_id=None,
        return_embeddings=False,
        return_mems=False,
        return_attn=False,
        mems=None,
        pos=None,
        prepend_embeds=None,
        **kwargs,
    ):

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
        x = self.attn_layers(x, mask=mask, mems=mems, **kwargs)
        # no post norm

        # head similar to https://github.com/lucidrains/x-transformers/blob/4b46febcc9442ad07e9ba6ea1dc4f9839975b27d/x_transformers/x_transformers.py#L1120-L1143
        return self.decoder(input_id, context=x)  # loss
