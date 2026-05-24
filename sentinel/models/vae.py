import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # 1D CNN: input shape [batch, input_dim, window_size]
        # output: mu and log_var, each of shape [batch, latent_dim]
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(64, latent_dim)
        self.log_var_head = nn.Linear(64, latent_dim)
    def forward(self, x:torch.Tensor):
        h = self.conv(x.permute(0,2,1))
        h = h.mean(dim=-1)
        return self.mu_head(h), self.log_var_head(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, window_size: int):
        super().__init__()
        # input: z of shape [batch, latent_dim]
        # output: reconstruction of shape [batch, window_size, output_dim]
        self.window_size = window_size
        self.fc = nn.Linear(latent_dim, 64 * window_size)
        self.deconv = nn.Sequential(
            nn.Conv1d(64,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(32,output_dim, kernel_size=3, padding=1)
        )
    def forward(self, x:torch.Tensor):
        z = self.fc(x)
        z = z.view(x.size(0),64,self.window_size)
        z = self.deconv(z)
        return z.permute(0,2,1)
        
    

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, window_size: int):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim, window_size)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # implement the reparameterization trick
        # z = mu + sigma * epsilon,   where epsilon ~ N(0, 1)
        std = torch.exp(log_var * 0.5)
        epsilon = torch.randn_like(mu)
        return mu + std * epsilon

    def forward(self, x: torch.Tensor):
        # encode → reparameterize → decode
        # return reconstruction, mu, log_var
        mu, log_var = self.encoder(x)
        log_var = log_var.clamp(-4,4)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

def vae_loss(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # MSE reconstruction loss + KL divergence
    # return total loss
    recon_loss = torch.mean((x - recon).pow(2))
    loss = recon_loss + (-0.5 *  torch.mean(1+log_var - mu.pow(2) - log_var.exp()))
    return loss


if __name__ == "__main__":
    model = VAE(input_dim=14, latent_dim=16, window_size=30)
    x = torch.randn(32, 30, 14)  # [batch, window_size, input_dim]
    recon, mu, log_var = model(x)
    print(recon.shape)  # should be [32, 30, 14]
    loss = vae_loss(x, recon, mu, log_var)
    print(loss.item())