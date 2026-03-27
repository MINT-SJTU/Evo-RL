from __future__ import annotations

import torch
import torch.nn as nn


class RLTokenModule(nn.Module):
    """RL Token encoder-decoder.

    Encoder appends a learned <rl> embedding to VLA tokens, runs a transformer,
    and reads out the last position as z_rl (B, D).

    Decoder autoregressively reconstructs VLA embeddings from the bottleneck
    using teacher forcing and causal masking.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        ff_dim: int | None = None,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * token_dim

        self.token_dim = token_dim
        self.rl_token_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out_proj = nn.Linear(token_dim, token_dim)

    def encode(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into a single RL token.

        Args:
            vla_tokens: (B, M, D) -- final VLA token embeddings (should be detached)

        Returns:
            z_rl: (B, D) -- the RL token
        """
        B = vla_tokens.shape[0]
        rl = self.rl_token_embed.expand(B, -1, -1)
        x = torch.cat([vla_tokens, rl], dim=1)  # (B, M+1, D)
        out = self.encoder(x)
        return out[:, -1, :]  # (B, D)

    def decode(self, z_rl: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Autoregressive reconstruction with teacher forcing.

        Args:
            z_rl: (B, D)
            teacher_tokens: (B, M, D) -- stop-gradiented VLA embeddings

        Returns:
            pred: (B, M, D)
        """
        M = teacher_tokens.shape[1]
        memory = z_rl.unsqueeze(1)  # (B, 1, D)
        # Shifted input: [z_rl, z_1, ..., z_{M-1}]
        dec_input = torch.cat([memory, teacher_tokens[:, :-1]], dim=1)  # (B, M, D)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_rl.device
        )
        out = self.decoder(tgt=dec_input, memory=memory, tgt_mask=causal_mask)
        return self.out_proj(out)

    def reconstruction_loss(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """L_ro = E[sum_i || pred_i - z_bar_i ||^2]"""
        z_bar = vla_tokens.detach()
        z_rl = self.encode(z_bar)
        pred = self.decode(z_rl, z_bar)
        return ((pred - z_bar) ** 2).mean()
