"""ARTM on the final ladder (docs/artm_ladder.md; Transition Matching, Shaul et al. 2025, arXiv 2506.23589, section 2.2, "Autoregressive Transition Matching").

Generation takes T transitions X_0 -> X_1 -> ... -> X_T = image, X_0 = the initial noise. The supervising process is the independent linear process,
X_k = (1 - k/T) e_k + (k/T) x with a new e_k ~ N(0, I) for every k. Each transition is autoregressive over the patches, as an LLM: the model gives
the distribution of patch i of X_{k+1} given X_k (all of it), the patches of X_{k+1} before i, and the caption, and a small flow head (TokenFlowHead,
as the DTM head) draws the patch from it.

The backbone is the ladder's JiT_T2I, unchanged in its blocks; its input is 2N image tokens: the N patches of X_k (the context, full attention among
themselves) and the N patches of X_{k+1} shifted by one (position i holds patch i - 1, position 0 a learned BOS; causal: position i sees the context,
the target positions up to i and the caption). Both halves keep the 2D positions of their patches (the fixed sin-cos table and 2D RoPE); a learned
segment vector per half tells them apart. The adaLN condition is the time embedding of k / T. The output at target position i, after norm_final, is
the head's condition for patch i. FinalLayer's linear is dropped. The attention runs on the SDPA path with a boolean mask (GPIC_MFU_ATTN=sdpa).

artm_sample: the T transitions with a KV cache: per transition one pass over the context (its keys and values kept per block), then N one-token
steps; at step i the head draws patch i (art_head_steps Euler steps of its flow, from fresh noise). With art_T = 0 nothing changes in JiT_T2I.
"""
import torch
import torch.nn as nn

from src.models.transformer import JiT_T2I as J
from src.models.transformer.JiT_T2I import JiT_T2I, TokenFlowHead, modulate
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb_real


class ARTMJiT(JiT_T2I):
    is_artm = True

    def __init__(self, art_T=3, art_head_width=1024, art_head_depth=3, art_head_steps=32, art_t_eps=0.05, **kwargs):
        super().__init__(**kwargs)   # the point model's init, every RNG draw included
        assert art_T >= 1 and art_head_steps >= 1, (art_T, art_head_steps)
        assert not (self.gm_k or self.dtm_width or self.head_width or self.xi_dim or self.code_dim), "ARTM is a model of its own"
        self.art_T, self.art_head_steps, self.art_t_eps = art_T, art_head_steps, art_t_eps
        self.final_layer.linear = nn.Identity()   # the head reads norm_final's output; no unused parameter (DDP)
        with torch.random.fork_rng(devices=[]):   # after the point model's init, with the CPU RNG restored: every later draw is the point model's
            self.art_head = TokenFlowHead(self.in_channels * self.patch_size ** 2, self.hidden_size, art_head_width, art_head_depth)
            self.art_bos = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
            self.art_seg = nn.Parameter(torch.randn(2, self.hidden_size) * 0.02)

    # ------------------------------------------------------------------------------------------------------------------- shared pieces
    def _cond_text(self, t, y, y_mask):
        """condition [B, 1, D] (silu of the time embedding) and the refined caption [B, T, D], as JiT_T2I.forward's masked or unmasked path."""
        B = t.shape[0]
        te = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        ypos = self.y_pos_embedding if y_mask is None else self.y_pos_embedding[:, :y.shape[1]]
        y_dtype = y.dtype
        y = self.y_embedder(y).view(B, -1, self.hidden_size) + ypos.to(y_dtype)
        condition = nn.functional.silu(te)
        text_mask = None if y_mask is None else y_mask[:, None, None, :]
        for block in self.text_refine_blocks:
            y = block(y, condition) if text_mask is None else block(y, condition, text_mask)
        return condition, y

    def _tokens(self, x):
        """[B, C, H, W] -> [B, N, C p p], the layout of JiT_T2I.forward."""
        if J._PATCHIFY:
            return J.patchify(x, self.patch_size)
        return nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

    def _mask(self, B, N, T, y_mask, device):
        """Boolean [B, 1, 2N, 2N + T] (True = attend): context rows see the context; target row i sees the context and target rows <= i; every row
        sees the real caption tokens (all T of them without y_mask)."""
        img = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=device)
        img[:, :N] = True
        img[N:, N:] = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))
        txt = (torch.ones(B, 1, 1, T, dtype=torch.bool, device=device) if y_mask is None else y_mask[:, None, None, :]).expand(B, 1, 2 * N, T)
        return torch.cat([img[None, None].expand(B, 1, 2 * N, 2 * N), txt], dim=-1)

    def _head_draw(self, z):
        """One draw per row of z [M, D] from the head's flow: Euler on its x prediction in art_head_steps steps, from fresh noise -> [M, P] fp32."""
        M, P = z.shape[0], self.in_channels * self.patch_size ** 2
        grid = torch.linspace(0, 1, self.art_head_steps + 1, device=z.device, dtype=torch.float32)
        hq = torch.randn(M, P, device=z.device, dtype=torch.float32)
        for j in range(self.art_head_steps):
            hat = self.art_head(hq.to(z.dtype), grid[j].expand(M), z).float()
            hq = hat if j == self.art_head_steps - 1 else hq + (grid[j + 1] - grid[j]) * (hat - hq) / (1 - grid[j]).clamp_min(self.art_t_eps)
        return hq

    # ------------------------------------------------------------------------------------------------------------------- training
    def forward(self, x, t, y, y_mask=None, art_tgt=None, art_q=None, art_tau=None, **unused):
        """x: the context X_k as tokens [B, N, P]; t [B] = k / T; art_tgt [B, N, P]: X_{k+1}'s patches (the teacher's); art_q [B, N, P] and art_tau
        [B, N]: the head's state at each target patch. Returns the head's x_hat [B, N, P]. Without art_q: the head's tokens z [B, N, D] (tests)."""
        assert not unused.get("y_pack"), "packed captions (lever E) run only the point model"
        B, N, P = x.shape
        side = int(round(N ** 0.5))
        cos, sin = self.fetch_pos(side, side, x.device)
        pos = (torch.cat([cos, cos]), torch.cat([sin, sin]))
        condition, yr = self._cond_text(t, y, y_mask)
        xc = self.x_embedder(x)
        xc += self.pos_embed   # in place, as JiT_T2I.forward: the residual stream keeps the embedder's dtype (bf16 under autocast)
        xc += self.art_seg[0]
        xt = self.x_embedder(art_tgt[:, :-1])
        xt = torch.cat([self.art_bos.expand(B, 1, self.hidden_size).to(xt.dtype), xt], dim=1)
        xt += self.pos_embed
        xt += self.art_seg[1]
        h = torch.cat([xc, xt], dim=1)
        mask = self._mask(B, N, yr.shape[1], y_mask, x.device)
        for block in self.blocks:
            h = block(h, yr, condition, pos, mask)
        z = self.final_layer(h[:, N:])
        if art_q is None:
            return z
        return self.art_head(art_q.reshape(B * N, P).to(z.dtype), art_tau.reshape(B * N), z.reshape(B * N, -1)).view(B, N, P)

    # ------------------------------------------------------------------------------------------------------------------- sampling
    def _attn_kv(self, attn, h, pos):
        """The SDPA path of Attention for image tokens h [B, L, C]: q, k (q_norm, k_norm, RoPE) and v, [B, H, L, Hc] each."""
        B, L, C = h.shape
        H = attn.num_heads
        qkv = attn.qkv_x(h).reshape(B, L, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = attn.q_norm(q.contiguous()), attn.k_norm(k.contiguous())
        q, k = apply_rotary_emb_real(q, k, pos[0], pos[1])
        return q, k, v

    def _text_kv(self, attn, y):
        B, T, C = y.shape
        H = attn.num_heads
        kv = attn.kv_y(y).reshape(B, T, 2, H, C // H).permute(2, 0, 3, 1, 4)
        return attn.k_norm(kv[0].contiguous()), kv[1]

    def _block(self, block, h, c, attend):
        """FlattenDiTBlock.forward with the attention given as a function of the modulated input (both norm paths, as the block)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6, dim=-1)
        if J._FUSED_NORM:
            a = J.mod_rmsnorm(h, block.norm1.weight, shift_msa, scale_msa, block.norm1.variance_epsilon)
            h, a = J.res_mod_rmsnorm(h, attend(a), gate_msa, block.norm2.weight, shift_mlp, scale_mlp, block.norm2.variance_epsilon)
            return h + gate_mlp * block.mlp(a)
        h = h + gate_msa * attend(modulate(block.norm1(h), shift_msa, scale_msa))
        return h + gate_mlp * block.mlp(modulate(block.norm2(h), shift_mlp, scale_mlp))

    @torch.no_grad()
    def artm_sample(self, x0, y, y_mask=None, teacher=None):
        """x0 [B, C, H, W]: X_0 (the initial noise). Returns X_T [B, C, H, W] fp32. teacher (tests only): a list of art_T token tensors [B, N, P];
        then transition k reads teacher[k] instead of the head's draws, and the method returns (X_T, the list of each transition's z [B, N, D])."""
        B, C, H, W = x0.shape
        side = H // self.patch_size
        N, P = side * side, C * self.patch_size ** 2
        cos, sin = self.fetch_pos(side, side, x0.device)
        text_keys = None if y_mask is None else y_mask[:, None, None, :]
        xk = x0
        zs = []
        for k in range(self.art_T):
            t = torch.full((B,), k / self.art_T, device=x0.device, dtype=torch.float32)
            condition, yr = self._cond_text(t, y, y_mask)
            # the context: one pass, keys and values kept per block
            h = self.x_embedder(self._tokens(xk))
            h += self.pos_embed
            h += self.art_seg[0]
            caches = []
            for block in self.blocks:
                attn = block.attn
                ky, vy = self._text_kv(attn, yr)
                kv = {"ky": ky, "vy": vy}

                def attend(a, attn=attn, kv=kv):
                    q, kx, vx = self._attn_kv(attn, a, (cos, sin))
                    kv["kx"], kv["vx"] = kx, vx
                    m = None if text_keys is None else torch.cat([text_keys.new_ones(B, 1, 1, N), text_keys], dim=-1)
                    o = attention(q, torch.cat([kx, kv["ky"]], dim=2), torch.cat([vx, kv["vy"]], dim=2), attn_mask=m)
                    return attn.proj(o.transpose(1, 2).reshape(B, N, -1))
                h = self._block(block, h, condition, attend)
                kv["tk"] = kv["kx"].new_empty(B, attn.num_heads, N, kv["kx"].shape[-1])   # the target keys and values, filled step by step
                kv["tv"] = kv["vx"].new_empty(B, attn.num_heads, N, kv["vx"].shape[-1])
                caches.append(kv)
            # the target: N one-token steps
            out = torch.empty(B, N, P, device=x0.device, dtype=torch.float32)
            zk = []
            inp = self.art_bos.expand(B, 1, self.hidden_size).to(h.dtype).clone()
            inp += self.pos_embed[:, :1]
            inp += self.art_seg[1]
            for i in range(N):
                h = inp
                pos_i = (cos[i:i + 1], sin[i:i + 1])
                for block, kv in zip(self.blocks, caches):
                    attn = block.attn

                    def attend(a, attn=attn, kv=kv, i=i, pos_i=pos_i):
                        q, kx, vx = self._attn_kv(attn, a, pos_i)
                        kv["tk"][:, :, i:i + 1] = kx
                        kv["tv"][:, :, i:i + 1] = vx
                        kk = torch.cat([kv["kx"], kv["tk"][:, :, :i + 1], kv["ky"]], dim=2)
                        vv = torch.cat([kv["vx"], kv["tv"][:, :, :i + 1], kv["vy"]], dim=2)
                        m = None if text_keys is None else torch.cat([text_keys.new_ones(B, 1, 1, N + i + 1), text_keys], dim=-1)
                        o = attention(q, kk, vv, attn_mask=m)
                        return attn.proj(o.transpose(1, 2).reshape(B, 1, -1))
                    h = self._block(block, h, condition, attend)
                z = self.final_layer(h)[:, 0]   # [B, D]
                zk.append(z)
                tok = teacher[k][:, i].float() if teacher is not None else self._head_draw(z)
                out[:, i] = tok
                if i + 1 < N:
                    inp = self.x_embedder(tok[:, None])
                    inp += self.pos_embed[:, i + 1:i + 2]
                    inp += self.art_seg[1]
            zs.append(torch.stack(zk, dim=1))
            xk = self.unpatchify(out, H, W)
        return (xk, zs) if teacher is not None else xk
