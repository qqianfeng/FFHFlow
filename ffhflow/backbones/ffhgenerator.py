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


class FFHGenerator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3 + 15,
                 dtype=torch.float64,
                 **kwargs):

        super(FFHGenerator, self).__init__()

        self.cfg = cfg.MODEL.BACKBONE

        self.latentD = self.cfg.LATENT_DIM  # 5

        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        # why input in_bps again here?
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_logvar = nn.Linear(n_neurons, self.latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(self.latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_bps, n_neurons)

        self.dec_joint_conf = nn.Linear(n_neurons, 15)
        self.dec_rot = nn.Linear(n_neurons, 6)
        self.dec_transl = nn.Linear(n_neurons, 3)

        self.fc_flow_feat = nn.Linear(10, 9)

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

    def decode(self, Zin, bps_object):

        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, final_ll=True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), final_ll=True)

        joint_conf = self.dec_joint_conf(X)
        rot_6D = self.dec_rot(X)
        transl = self.dec_transl(X)

        results = {"rot_6D": rot_6D, "transl": transl, "joint_conf": joint_conf, "z": Zin}

        return results

    def encode(self, data):
        self.set_input(data)
        X = torch.cat([self.bps_object, self.rot_matrix, self.transl, self.joint_conf], dim=1)

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

        # data_recon = self.decode(z, self.bps_object)
        # results = {'mu': mu, 'logvar': logvar}
        # results.update(data_recon)

        feat = torch.cat([mu, logvar],dim=1)
        feat = self.fc_flow_feat(feat)
        return feat

class ResNet_3layer(nn.Module):
    def __init__(self,
                 in_dim=4096,
                 hid_dim=512,
                 out_dim=128,
                 prob_flag=False,
                 dtype=torch.float64,
                 **kwargs):
        super().__init__()

        self.prob_flag = prob_flag
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.rb1 = ResBlock(in_dim, hid_dim)
        self.rb2 = ResBlock(in_dim + hid_dim, hid_dim)
        self.rb3 = ResBlock(in_dim + hid_dim, out_dim)
        if self.prob_flag:
            self.enc_mu = nn.Linear(out_dim, out_dim)
            self.enc_logvar = nn.Linear(out_dim, out_dim)

        self.dout = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()

        self.dtype = dtype

    def forward(self, data, return_mean_var=False):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, bps_object,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        X = data
        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))

        if self.prob_flag:
            mu, logvar = self.enc_mu(X), self.enc_logvar(X)
            if return_mean_var:
                return mu, logvar, self.sample(mu, logvar)
            else:
                return self.sample(mu, logvar)
        else:
            return X

    def sample(self, mu, logvar):
        assert self.prob_flag, "Only avaialble when cfg.MODEL.BACKBONE.PROBABILISTIC is True."
        # std = logvar.exp().pow(0.5)
        # q_z = torch.distributions.normal.Normal(mu, std)
        # z = q_z.rsample()
        # return z

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)



class BPSMLP(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 dtype=torch.float64,
                 **kwargs):
        super().__init__()

        self.prob_flag = cfg.MODEL.BACKBONE.PROBABILISTIC
        self.bn1 = nn.BatchNorm1d(in_bps)
        self.rb1 = ResBlock(in_bps, n_neurons)
        self.rb2 = ResBlock(in_bps + n_neurons, n_neurons)
        self.rb3 = ResBlock(in_bps + n_neurons, cfg.MODEL.FLOW.CONTEXT_FEATURES)
        if self.prob_flag:
            self.enc_mu = nn.Linear(cfg.MODEL.FLOW.CONTEXT_FEATURES, cfg.MODEL.FLOW.CONTEXT_FEATURES)
            self.enc_logvar = nn.Linear(cfg.MODEL.FLOW.CONTEXT_FEATURES, cfg.MODEL.FLOW.CONTEXT_FEATURES)

        self.dout = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()

        self.dtype = dtype

    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on
        we are in train or eval mode.
        """
        self.bps_object = data["bps_object"].to(dtype=self.dtype).contiguous()


    def forward(self, data, return_mean_var=False):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, bps_object,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        if isinstance(data, dict):
            self.set_input(data)
        else:
            self.bps_object = data
        X = torch.cat([self.bps_object], dim=1)

        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))

        if self.prob_flag:
            mu, logvar = self.enc_mu(X), self.enc_logvar(X)
            if return_mean_var:
                return mu, logvar, self.sample(mu, logvar)
            else:
                return self.sample(mu, logvar)
        else:
            return X

    def sample(self, mu, logvar):
        assert self.prob_flag, "Only avaialble when cfg.MODEL.BACKBONE.PROBABILISTIC is True."
        # std = logvar.exp().pow(0.5)
        # q_z = torch.distributions.normal.Normal(mu, std)
        # z = q_z.rsample()
        # return z

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
