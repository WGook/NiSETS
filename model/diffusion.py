import torch
import numpy as np
import torch.nn.functional as F

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, estimator, n_feats,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.estimator = estimator

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None, gamma = 1):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            
            alpt = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            alpt_1 = get_noise(time-h, self.beta_min, self.beta_max, cumulative=True)
            beta_star = ((1-torch.exp(-alpt_1))/(1-torch.exp(-alpt))).sqrt()
            if alpt_1 < 0:
                beta_star = torch.zeros_like(beta_star)
            score_var = torch.sqrt(1.0 - torch.exp(-alpt))
            if stoc:  # adds stochastic term
                est_score = self.estimator(xt, mask, mu, t, spk)

                pf_score = torch.randn_like(est_score)*beta_star*score_var
                dxt_det = 0.5 * (mu - xt) - ((1-gamma)*pf_score + gamma*est_score)
                
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                est_score = self.estimator(xt, mask, mu, t, spk)
                
                pf_score = torch.randn_like(est_score)*beta_star*score_var
                dxt = 0.5 * (mu - xt - ((1-gamma)*pf_score + gamma*est_score))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None, gamma= 1, prior = False):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk, gamma, prior = prior)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)

        cum_noise_ = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        variance_ = 1.0 - torch.exp(-cum_noise_)
        x0_est = (xt + (torch.sqrt(1.0 - torch.exp(-cum_noise_))*noise_estimation) - mu*(1.0 - torch.exp(-0.5*cum_noise_)))/torch.exp(-0.5*cum_noise_)
        return loss, noise_estimation, xt, x0_est

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk), t
