import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_ll=True):
        if self.Fin == self.Fout:
            Xin = x
        else:
            Xin = self.fc3(x)
            Xin = self.ll(Xin)

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_ll:
            return self.ll(Xout)
        return Xout


class BPSVAE(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 dtype=torch.float64,
                 **kwargs):

        super(BPSVAE, self).__init__()

        self.cfg = cfg.MODEL.BACKBONE

        self.latentD = self.cfg.LATENT_DIM  # 5

        self.enc_bn1 = nn.BatchNorm1d(in_bps)
        self.enc_rb1 = ResBlock(in_bps, n_neurons)
        # why input in_bps again here?
        self.enc_rb2 = ResBlock(n_neurons + in_bps, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_logvar = nn.Linear(n_neurons, self.latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(self.latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_bps, n_neurons)

        self.dec_bps = nn.Linear(n_neurons, in_bps)

        self.dtype = dtype

    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on
        we are in train or eval mode.
        """
        self.rot_matrix = data["rot_matrix"].to(dtype=self.dtype)
        self.transl = data["transl"].to(dtype=self.dtype)
        self.joint_conf = data["joint_conf"].to(dtype=self.dtype)
        self.bps_object = data["bps_object"].to(dtype=self.dtype).contiguous()

        self.rot_matrix = self.rot_matrix.view(self.bps_object.shape[0], -1)

    def decode(self, Zin):

        X0 = Zin
        X = self.dec_rb1(X0, final_ll=True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), final_ll=True)

        bps = self.dec_bps(X)
        results = {"bps": bps, "z": Zin}

        return results

    def encode(self, data):
        self.set_input(data)
        X = self.bps_object

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return self.enc_mu(X), self.enc_logvar(X)

    def forward(self, data):
        # Encode data, get mean and logvar
        mu, logvar = self.encode(data) # [512,5], [512,5]

        std = logvar.exp().pow(0.5)
        q_z = torch.distributions.normal.Normal(mu, std)
        z = q_z.rsample()

        data_recon = self.decode(z)
        results = {'mu': mu, 'logvar': logvar}
        results.update(data_recon)

        return results