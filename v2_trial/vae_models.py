import torch
import torch.nn as nn

from models.vae_models import ResidualBlock


class TypeConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        withscale=False,
        num_types=3,
        condition_dim=8,
    ):
        super(TypeConditionalVAE, self).__init__()
        self.withscale = withscale
        self.input_dim = input_dim if not withscale else input_dim + 1
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_types = num_types
        self.condition_dim = condition_dim

        self.type_embedding = nn.Embedding(num_types, condition_dim)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.input_dim)

    def encode(self, x, scale=None):
        if scale is not None:
            x = torch.cat((x, scale), dim=1)
        x = self.fc1(x)
        x = self.res1(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, aneurysm_type, strip_scale=False):
        condition = self.type_embedding(aneurysm_type.long())
        x = self.fc3(torch.cat((z, condition), dim=1))
        x = self.res2(x)
        x = self.fc4(x)
        if self.withscale:
            return x[:, :-1] if strip_scale else (x[:, :-1], x[:, -1:])
        return x

    def forward(self, x, aneurysm_type, scale=None):
        mu, logvar = self.encode(x, scale)
        z = self.reparameterize(mu, logvar)
        if self.withscale:
            return *self.decode(z, aneurysm_type), mu, logvar
        return self.decode(z, aneurysm_type), mu, logvar
