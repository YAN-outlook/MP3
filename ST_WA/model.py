import torch
import torch.nn as nn
from attention import TemporalAttention, SpatialAttention
# from util import reparameterize

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.HS_fc = nn.Linear(dim, dim, bias=True)
        self.HT_fc = nn.Linear(dim, dim, bias=True)
        self.output_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, flow_eb, time_eb):
        XS = self.HS_fc(flow_eb)
        XT = self.HT_fc(time_eb)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, flow_eb), torch.multiply(1 - z, time_eb))
        H = self.output_fc(H)
        return H
class STWA(nn.Module):
    def __init__(self, device, num_nodes, out_dim, channels, dynamic, horizon, lag, memory_size, supports, dim_in, use_plugin=True):
        super(STWA, self).__init__()
        self.supports = supports
        self.num_nodes = num_nodes
        self.output_dim = out_dim
        self.channels = channels
        self.dynamic = dynamic
        self.horizon = horizon
        self.lag = lag
        self.memory_size = memory_size
        self.use_plugin = use_plugin
        input_dim = dim_in
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)

        if input_dim != 1:
            self.eval_dimin = nn.Linear(in_features=input_dim, out_features=1)

        self.layers = nn.ModuleList(
            [
                Layer(device=device, input_dim=self.channels, dynamic=self.dynamic, num_nodes=self.num_nodes, cuts=12,
                      cut_size=6, no_proxies=2, memory_size=self.memory_size),
                Layer(device=device, input_dim=self.channels, dynamic=self.dynamic, num_nodes=self.num_nodes, cuts=3,
                      cut_size=4, no_proxies=2, memory_size=self.memory_size),
                Layer(device=device, input_dim=self.channels, dynamic=self.dynamic, num_nodes=self.num_nodes, cuts=1,
                      cut_size=3, no_proxies=2, memory_size=self.memory_size),
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=12 * self.channels, out_features=256),
            nn.Linear(in_features=3 * self.channels, out_features=256),
            nn.Linear(in_features=1 * self.channels, out_features=256),
        ])

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.horizon * self.output_dim)])

        if self.dynamic:
            self.mu_estimator = nn.Sequential(*[
                nn.Linear(self.lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, self.memory_size)
            ])

            self.logvar_estimator = nn.Sequential(*[
                nn.Linear(self.lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, self.memory_size)
            ])
        if self.use_plugin:
            self.fusion = Fusion(32)
            self.lin_test = nn.Linear(3, 32)

    def forward(self, x, feat=None):
        if self.use_plugin:
            x = x.transpose(1, 2)
            x = self.lin_test(x)
            x = self.fusion(feat, x)
            if self.dynamic:
                if x.shape[-1] != 1:
                    x_dm = self.eval_dimin(x).squeeze(-1).transpose(1, 2)
                else:
                    x_dm = x
                mu = self.mu_estimator(x_dm)
                logvar = self.logvar_estimator(x_dm)
                z_data = reparameterize(mu, logvar)
            else:
                z_data = 0
        else:
            x_dm = x
            if self.dynamic:
                mu = self.mu_estimator(x_dm)
                logvar = self.logvar_estimator(x_dm)
                z_data = reparameterize(mu, logvar)
            else:
                z_data = 0
            x = x.transpose(1, 2).unsqueeze(-1)

        x = self.start_fc(x)
        batch_size = x.size(0)

        skip = 0
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x, z_data)
            skip_inp = x.transpose(2, 1).reshape(batch_size, self.num_nodes, -1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)
        out = self.projections(x)
        if self.output_dim == 1:
            out = out.transpose(2, 1).unsqueeze(-1)
        else:
            out = out.unsqueeze(-1).reshape(batch_size, self.num_nodes, self.horizon, -1).transpose(2, 1)

        return out, x_dm


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size, dynamic, memory_size, no_proxies):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.cuts = cuts
        self.cut_size = cut_size
        self.no_proxies = no_proxies
        self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
                                    requires_grad=True).to(device)

        self.temporal_att = TemporalAttention(input_dim, num_nodes=num_nodes, cut_size=cut_size)
        self.spatial_att = SpatialAttention(input_dim, num_nodes=num_nodes)

        if self.dynamic:
            self.mu = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)
            self.logvar = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)

        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x, z_data):
        # x shape: B T N C
        batch_size = x.size(0)

        if self.dynamic:
            z_sample = reparameterize(self.mu, self.logvar)
            z_data = z_data + z_sample

        temporal_parameters = [layer(x, z_data) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x, z_data) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t[:, :self.no_proxies, :, :], t, t, temporal_parameters)
            out = self.spatial_att(out, spatial_parameters)
            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)

class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
        else:
            print('Using FC')
            self.weights = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, x, memory=None):
        if self.dynamic:
            weights = self.weight_generator(memory).view(x.shape[0], self.num_nodes, self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], self.num_nodes, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases
