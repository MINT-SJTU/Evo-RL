from __future__ import annotations

import torch
import torch.nn as nn


class RLTokenModule(nn.Module):
    """RL Token encoder-decoder with configurable number of RL tokens.

    Encoder appends N learned <rl> embeddings to VLA tokens, runs a transformer,
    and reads out the last N positions as z_rl (B, N, D). For downstream RL,
    z_rl is mean-pooled to (B, D).

    Decoder autoregressively reconstructs VLA embeddings from the multi-token
    bottleneck using teacher forcing and causal masking.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        ff_dim: int | None = None,
        num_rl_tokens: int = 1,
        inference_only: bool = False,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * token_dim

        self.token_dim = token_dim
        self.num_rl_tokens = num_rl_tokens
        self.inference_only = inference_only
        self.rl_token_embed = nn.Parameter(torch.randn(1, num_rl_tokens, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        if not inference_only:
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
        """Encode VLA tokens into RL token(s).

        Args:
            vla_tokens: (B, M, D) -- final VLA token embeddings (should be detached)

        Returns:
            z_rl: (B, D) -- mean-pooled RL token for downstream RL state
        """
        z_rl_multi = self.encode_multi(vla_tokens)
        return z_rl_multi.mean(dim=1)  # (B, D)

    def encode_multi(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into multiple RL tokens (for decoder).

        Returns:
            z_rl: (B, N, D) where N = num_rl_tokens
        """
        B = vla_tokens.shape[0]
        rl = self.rl_token_embed.expand(B, -1, -1)  # (B, N, D)
        x = torch.cat([vla_tokens, rl], dim=1)  # (B, M+N, D)
        out = self.encoder(x)
        return out[:, -self.num_rl_tokens:, :]  # (B, N, D)

    def decode(self, z_rl: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Autoregressive reconstruction with teacher forcing.

        Args:
            z_rl: (B, D) single token or (B, N, D) multi-token
            teacher_tokens: (B, M, D) -- stop-gradiented VLA embeddings

        Returns:
            pred: (B, M, D)
        """
        M = teacher_tokens.shape[1]
        # Handle both single and multi-token z_rl
        if z_rl.dim() == 2:
            memory = z_rl.unsqueeze(1)  # (B, 1, D)
        else:
            memory = z_rl  # (B, N, D)
        N = memory.shape[1]
        # Shifted input: [rl_tokens..., z_1, ..., z_{M-N}]
        dec_input = torch.cat([memory, teacher_tokens[:, :M - N]], dim=1)  # (B, M, D)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_rl.device
        )
        out = self.decoder(tgt=dec_input, memory=memory, tgt_mask=causal_mask)
        return self.out_proj(out)

    def reconstruction_loss(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """L_ro = E[sum_i || pred_i - z_bar_i ||^2]"""
        z_bar = vla_tokens.detach()
        z_rl_multi = self.encode_multi(z_bar)
        pred = self.decode(z_rl_multi, z_bar)
        return ((pred - z_bar) ** 2).mean()
