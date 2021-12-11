import torch
import torch.nn as nn
from utils_sag import StyleRandomization, ContentRandomization

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
OUT_DIM_84 = {2: 39, 4: 37, 6: 31}
OUT_DIM_108 = {4: 47}

 
class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.obs_dim = obs_shape[0]
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(self.obs_dim, self.feature_dim)
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, obs, detach=False):
        obs = obs.view(1, -1)

        x = self.fc1(obs)
        x = nn.ReLU()(x)
        x = self.fc1(x)
        return x

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass

class StyleAgnosticEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, sag_stage=3):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.sag_stage = sag_stage

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM_84[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        # content_net
        self.content_net = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
             nn.BatchNorm2d(num_filters)]
        ) 
        for i in range(num_layers):
            self.content_net.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            self.content_net.append(nn.BatchNorm2d(num_filters))

        self.fc_content = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln_content = nn.LayerNorm(self.feature_dim)
        self.reward_content = nn.Linear(self.feature_dim, 1)

        # style_net
        style_layers = []
        for i in range(num_layers - sag_stage):
            style_layers += [nn.Conv2d(num_filters, num_filters, 3, stride=1),
                             nn.BatchNorm2d(num_filters)]
        self.style_net = nn.Sequential(*style_layers)

        self.fc_style = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln_style = nn.LayerNorm(self.feature_dim)
        self.reward_style = nn.Linear(self.feature_dim, 1)

        # randomization layer
        self.style_randomization = StyleRandomization()
        self.content_randomization = ContentRandomization()
        #self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward(self, obs, detach=False):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        # first module is feature extractor
        x_cont = torch.relu(self.content_net[0](obs))
        self.outputs['content_conv1'] = x_cont

        for i in range(self.num_layers):
            if i+1 == self.sag_stage:
                # randomization
                x_cont = self.style_randomization(x_cont)
                x_style = self.content_randomization(x_cont)
                self.outputs['content_style%s' % (i + 1)] = x_style
            x_cont = torch.relu(self.content_net[i+1](x_cont))
            self.outputs['content_conv%s' % (i + 1)] = x_cont

        # content output
        h_cont = x_cont.view(x_cont.size(0), -1)
        if detach:
            h_cont = h_cont.detach()

        h_cont = self.fc_content(h_cont)
        self.outputs['cont_fc'] = h_cont
        h_cont = self.ln_content(h_cont)
        self.outputs['cont_ln'] = h_cont
        r_cont = self.reward_content(torch.relu(h_cont))
        self.outputs['cont_rew'] = h_cont

        # style output
        x_style = self.style_net(x_style)
        h_style = x_style.view(x_style.size(0), -1)
        if detach:
            h_style = h_style.detach()
        h_style = self.fc_style(h_style)
        self.outputs['style_fc'] = h_style
        h_style = self.ln_style(h_style)
        self.outputs['style_ln'] = h_style
        r_style = self.reward_style(torch.relu(h_style))
        self.outputs['style_rew'] = r_style

        # output
        if self.output_logits:
            return h_cont, h_style, r_cont, r_style
        else:
            return torch.tanh(h_cont), torch.tanh(h_style), torch.tanh(r_cont), torch.tanh(r_style)

    def content_params(self):
        params = []
        for m in [self.content_net, self.fc_content, self.ln_content, self.reward_content]:
            params += [p for p in m.parameters()]
        return params

    def style_params(self):
        params = []
        for m in [self.style_net, self.fc_style, self.ln_style, self.reward_style]:
            params += [p for p in m.parameters()]
        return params

    def adv_params(self):
        params = []
        for layer in self.content_net[:self.sag_stage]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    params += [p for p in m.parameters()]
        return params

    def copy_conv_weights_from(self, source):
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.content_net[i], trg=self.content_net[i])
            if i >= self.sag_stage:
                tie_weights(src=source.style_net[self.num_layers - i], trg=self.style_net[self.num_layers - i])

_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'style': StyleAgnosticEncoder}

def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )