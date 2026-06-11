import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, embedding_dim):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        flat = z_e.reshape(-1, self.embedding_dim)
        codebook = self.embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view_as(z_e)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()

        encodings = F.one_hot(indices, self.num_codes).type_as(z_e)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q, indices.view(z_e.shape[0], z_e.shape[1]), codebook_loss, commitment_loss, perplexity


class ConditionalGHDVQVAE(nn.Module):
    def __init__(
        self,
        seq_len=144,
        channels=3,
        num_types=3,
        hidden_dim=128,
        embedding_dim=64,
        num_codes=128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_types = num_types
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(channels + 1, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            ConvResidualBlock(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            ConvResidualBlock(hidden_dim),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3, padding=1),
        )
        self.quantizer = VectorQuantizer(num_codes=num_codes, embedding_dim=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim + num_types, hidden_dim, kernel_size=3, padding=1),
            ConvResidualBlock(hidden_dim),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            ConvResidualBlock(hidden_dim // 2),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, channels, kernel_size=3, padding=1),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(embedding_dim + num_types, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _condition(self, aneurysm_type, length):
        cond = F.one_hot(aneurysm_type.long(), num_classes=self.num_types).float()
        return cond, cond[:, None, :].expand(-1, length, -1)

    def encode(self, x, scale):
        scale_seq = scale[:, None, :].expand(-1, x.shape[1], -1)
        enc_in = torch.cat((x, scale_seq), dim=-1).transpose(1, 2)
        return self.encoder(enc_in).transpose(1, 2)

    def decode(self, z_q, aneurysm_type):
        cond, cond_seq = self._condition(aneurysm_type, z_q.shape[1])
        dec_in = torch.cat((z_q, cond_seq), dim=-1).transpose(1, 2)
        x_recon = self.decoder(dec_in).transpose(1, 2)
        pooled = z_q.mean(dim=1)
        scale_recon = self.scale_head(torch.cat((pooled, cond), dim=-1))
        return x_recon, scale_recon

    def forward(self, x, scale, aneurysm_type):
        z_e = self.encode(x, scale)
        z_q, indices, codebook_loss, commitment_loss, perplexity = self.quantizer(z_e)
        x_recon, scale_recon = self.decode(z_q, aneurysm_type)
        return {
            "x_recon": x_recon,
            "scale_recon": scale_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class CodeSequenceTransformerPrior(nn.Module):
    def __init__(
        self,
        num_codes,
        seq_len,
        num_types=3,
        model_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        mlp_ratio=4,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.seq_len = seq_len
        self.num_types = num_types
        self.bos_token = num_codes

        self.token_embedding = nn.Embedding(num_codes + 1, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        self.type_embedding = nn.Embedding(num_types, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_codes)

    def _causal_mask(self, length, device):
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )

    def _teacher_forcing_input(self, indices):
        bos = indices.new_full((indices.shape[0], 1), self.bos_token)
        return torch.cat((bos, indices[:, :-1]), dim=1)

    def forward(self, indices, aneurysm_type):
        if indices.shape[1] != self.seq_len:
            raise ValueError(f"Expected code length {self.seq_len}, got {indices.shape[1]}")
        tokens = self._teacher_forcing_input(indices)
        x = self.token_embedding(tokens)
        x = x + self.position_embedding[:, : self.seq_len]
        x = x + self.type_embedding(aneurysm_type.long())[:, None, :]
        x = self.transformer(x, mask=self._causal_mask(self.seq_len, indices.device))
        return self.head(self.norm(x))

    def loss(self, indices, aneurysm_type):
        logits = self(indices, aneurysm_type)
        return F.cross_entropy(logits.reshape(-1, self.num_codes), indices.reshape(-1))

    @torch.no_grad()
    def sample(self, aneurysm_type, temperature=1.0, top_k=None):
        if aneurysm_type.dim() == 0:
            aneurysm_type = aneurysm_type[None]
        device = aneurysm_type.device
        batch = aneurysm_type.shape[0]
        generated = torch.empty(batch, 0, dtype=torch.long, device=device)

        for _ in range(self.seq_len):
            padded = torch.cat(
                (
                    generated,
                    torch.zeros(batch, self.seq_len - generated.shape[1], dtype=torch.long, device=device),
                ),
                dim=1,
            )
            logits = self(padded, aneurysm_type)[:, generated.shape[1]]
            logits = logits / max(float(temperature), 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(int(top_k), self.num_codes), dim=-1)
                logits = logits.masked_fill(logits < values[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
