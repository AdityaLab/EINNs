import torch.nn as nn
import numpy as np
import torch
import pdb 
import json
import math
import os
from data_utils import all_hhs_regions
from torch.nn.parameter import Parameter
no_states=5
no_ode_params=4
dtype = torch.float
torch.autograd.set_detect_anomaly(True)
EPS = 1e-10
N_ODES_MAX = 405 # instead of 500
SEED = 17 # for random number generators

def save_model(file_prefix: str, model: nn.Module):
    torch.save(model.feat_nn_mod.state_dict(), file_prefix + "_feat_nn_mod.pth")
    torch.save(model.time_nn_mod.state_dict(), file_prefix + "_time_nn_mod.pth")
    torch.save(model.ode.state_dict(), file_prefix + "_ode.pth")

def load_model(file_prefix: str, model: nn.Module):
    model.feat_nn_mod.load_state_dict(torch.load(file_prefix + "_feat_nn_mod.pth"))
    model.time_nn_mod.load_state_dict(torch.load(file_prefix + "_time_nn_mod.pth"))
    model.ode.load_state_dict(torch.load(file_prefix + "_ode.pth"))

# code from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class TorchStandardScaler:
    def fit(self, x, device):
        x = torch.tensor(x).float().to(device)
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        if torch.is_tensor(x):
            x -= self.mean
            x /= (self.std + 1e-7)
        else:
            x -= self.mean.cpu().numpy()
            x /= (self.std + 1e-7).cpu().numpy()
        return x

    def fit_transform(self, x, device):
        self.fit(x, device)
        return self.transform(x)

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x
    

class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(latent_seqs)
        return out

    def forward(self, seqs):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(latent_seqs)
        return out


class DecodeSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 5,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(DecodeSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.act_fcn = nn.Tanh()

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)

    def forward(self, Hi_data, hidden):
        # Hi_data is scaled time
        inputs = Hi_data.transpose(1,0)
        if self.bidirectional:
            h0 = hidden.expand(2,-1,-1).contiguous()
        else:
            h0 = hidden.unsqueeze(0)
        # Take last output from GRU
        latent_seqs = self.rnn(inputs, h0)[0]
        latent_seqs = latent_seqs.transpose(1,0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs


# Fourier feature mapping
# as per https://bmild.github.io/fourfeat/
def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

def get_B_gauss(mapping_size,region,scale=1):
    ''' returns always the same B gauss
            useful to have consistency across time and feat module
    ''' 
    region_int_encode = np.where(np.array(all_hhs_regions)==region)[0][0]
    np.random.seed(region_int_encode)
    B_gauss = np.random.normal(size=(mapping_size, 1)) * scale
    return B_gauss

class ff_net_fourier(nn.Module):
    def __init__(self,scale,region,out_dim,device):
        super(ff_net_fourier, self).__init__()
        
        mapping_size = 20  # 20
        B_gauss = get_B_gauss(mapping_size,region,scale)
        self.B_gauss = torch.from_numpy(B_gauss).type(torch.float).to(device)
        
        act_fcn = nn.Tanh()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        
        hidden_dim = 2*mapping_size
        self.net1 =  [
            nn.Linear(
                in_features=hidden_dim, out_features=hidden_dim
            ),
            act_fcn,
            nn.Linear(
                in_features=hidden_dim, out_features=hidden_dim
            ),
            act_fcn,
            nn.Linear(
                in_features=hidden_dim, out_features=hidden_dim
            ),
            act_fcn,
            nn.Linear(
                in_features=hidden_dim, out_features=out_dim
            ),
            act_fcn,
        ]
        self.net1 = nn.Sequential(*self.net1).to(device)
        self.net1.apply(init_weights)

    def forward(self, *inputs):
        inputs = torch.cat(inputs, 1)
        inputs = input_mapping(inputs,self.B_gauss)
        emb_E = self.net1(inputs)
        return emb_E


class time_pnn_fourier(nn.Module):
    def __init__(
            self,
            regions,
            scale: int=10, # values of tutorial are 1,10,100
            out_dim: int=20,
            # out_layers=None,
            device=None,
        ):
        super(time_pnn_fourier, self).__init__()


        self.time_mods = nn.ModuleDict({'time_nn_'+region: ff_net_fourier(scale,region,out_dim,device) for region in regions})

class EncoderModules(nn.Module):
    def __init__(
            self,
            regions,
            dim_seq_in: int = 5,
            device=None,
        ):
        super(EncoderModules, self).__init__()

        self.mods = nn.ModuleDict({'encoder_'+region:\
            EmbedAttenSeq(
                    dim_seq_in=dim_seq_in,
                    rnn_out=64,  # divides by 2 if bidirectional
                    dim_out=32,
                    n_layers=2,
                    bidirectional=True,
                ).to(device)
         for region in regions})

class DecoderModules(nn.Module):
    def __init__(
            self,
            regions,
            device=None,
        ):
        super(DecoderModules, self).__init__()

        self.mods = nn.ModuleDict({'decoder_'+region:\
            DecodeSeq(
                    dim_seq_in=1,
                    rnn_out=64, # divides by 2 if bidirectional
                    dim_out=20,
                    n_layers=1,
                    bidirectional=True,
                ).to(device)
         for region in regions})

class OutputModules(nn.Module):
    def __init__(
            self,
            regions,
            device=None,
            out_dim=5,
        ):
        super(OutputModules, self).__init__()

        out_layer_width = 20
        out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=2*out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=2*out_layer_width, out_features=2*out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=2*out_layer_width, out_features=out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=out_layer_width, out_features=out_dim
            ),
        ]

        self.mods = nn.ModuleDict({'output_'+region:\
            nn.Sequential(*out_layer).to(device)
            for region in regions})
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.mods.apply(init_weights)



class SEIRm_tanh(nn.Module):
    def __init__(self, population, parameter):
        super(SEIRm_tanh, self).__init__()
        # SEIRM parameters:
        self.N     = population # Population
        self.init_params(parameter)

    def get_scaled_params(self,convert_cpu=False):
        params = {}
        # these take values in the domain 0-1
        params['alpha']  = (torch.tanh(self.logalpha) + 1)*.5 + 0
        params['beta'] = (torch.tanh(self.logbeta) + 1)*.5 + 0
        # takes from 0-1
        params['gamma'] = (torch.tanh(self.loggamma) + 1)*.5 + 0
        params['mu'] = (torch.tanh(self.logmu) + 1)*.5 + 0
        
        if convert_cpu:
            for k, v in params.items():
                params[k] = v.detach().cpu().data.item()
        return params

    def get_param_vector(self):
        params = self.get_scaled_params()
        alpha = params['alpha'] 
        beta = params['beta'] 
        gamma = params['gamma'] 
        mu = params['mu'] 
        return torch.stack([alpha,beta,gamma,mu])

    def init_params(self,params,convert_cpu=False):
        """
            given:
                y = (tanh(x)+b)*a + c
            use:
                val = tanh-1( 1/a*(y-c) - b )
        """
        # these take values in the domain 0-1
        EPS = -1e-12 if params['alpha']>0.0 else 1e-12
        self.logalpha = Parameter(torch.tensor(np.arctanh(1/.5*(params['alpha']-0) - 1 + EPS), dtype=dtype),requires_grad=True) 
        EPS = -1e-12 if params['beta']>0.0 else 1e-12
        self.logbeta = Parameter(torch.tensor(np.arctanh(1/.5*(params['beta']-0) - 1 + EPS), dtype=dtype),requires_grad=True) 
        EPS = -1e-12 if params['gamma']>0.0 else 1e-12
        self.loggamma = Parameter(torch.tensor(np.arctanh(1/.5*(params['gamma']-0) - 1 + EPS), dtype=dtype),requires_grad=True) 
        EPS = -1e-12 if params['mu']>0.0 else 1e-12
        self.logmu = Parameter(torch.tensor(np.arctanh(1/.5*(params['mu']-0) - 1 + EPS), dtype=dtype),requires_grad=True) 


    def ODE(self, state, t=None):
        """
        Computes ODE states via equations       
            state is the array of state value (S,E,I,R,M)
        """
        params = self.get_scaled_params()

        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        dSE = params['beta'] * state[:,0] * state[:,2] / self.N
        dEI = params['alpha'] * state[:,1]
        dIR = params['gamma'] * state[:,2]
        dIM = params['mu'] * state[:,2]

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        dR  = dIR
        dM  = dIM

        # concat and reshape to make it rows as obs, cols as states
        dS = dS.reshape(dS.shape[0],-1)
        dE = dE.reshape(dE.shape[0],-1)
        dI = dI.reshape(dI.shape[0],-1)
        dR = dR.reshape(dR.shape[0],-1)
        dM = dM.reshape(dM.shape[0],-1)
        dstate = torch.cat([dS, dE, dI, dR, dM], 1)
        return dstate

   
    def ODE_detach(self, state, t=None):
        """
        same as ODE but detach ode parameters to avoid messing up with their gradients

        """
        params = self.get_scaled_params()

        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        dSE = params['beta'].detach() * state[:,0] * state[:,2] / self.N
        dEI = params['alpha'].detach() * state[:,1]
        dIR = params['gamma'].detach() * state[:,2]
        dIM = params['mu'].detach() * state[:,2]

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        dR  = dIR
        dM  = dIM

        # concat and reshape to make it rows as obs, cols as states
        dS = dS.reshape(dS.shape[0],-1)
        dE = dE.reshape(dE.shape[0],-1)
        dI = dI.reshape(dI.shape[0],-1)
        dR = dR.reshape(dR.shape[0],-1)
        dM = dM.reshape(dM.shape[0],-1)
        dstate = torch.cat([dS, dE, dI, dR, dM], 1)
        return dstate


class ode_modules_set(nn.Module):
    def __init__(
        self,
        regions,
        init_ode_wcalibration: bool=True,
        initial_ode_idx: int=0,  # time index for initial ode
        final_ode_idx:int=400, # time index for final ode
        device=torch.device("cpu"),
        pop: dict={},
        ) -> None:
        super(ode_modules_set, self).__init__()

        self.ode_time_array = np.arange(initial_ode_idx,final_ode_idx)
        self.device = device
        self.pop = pop
        self.ode_fcn = SEIRm_tanh 
        if init_ode_wcalibration:
            self.set_ode_params_from_json(regions)
        else:
            # have one ode for each time step
            raise Exception('not implemented')
    
    def set_ode_params_from_json(self,regions):
        """
            Function to initialize ode parameters
        """
        module_dict = {}
        for r in regions:
            path = './data/analytical/{}/'.format(r)
            param_json = os.path.join(path,'seirm-t-rmse-calibration.json')
            with open(param_json) as infile:
                init_params = json.load(infile)
            init_params['default'] = {"alpha": 0.2, "beta": 0.2, "gamma": 0.5, "mu": 0.01}
            # if parameter is not available in json, use some default parameters
            for i in self.ode_time_array:
                try:
                    module_dict[(r+' ode_'+str(i))] = \
                        self.ode_fcn(self.pop[r], init_params['ode_'+str(i)]).to(self.device)
                except:
                    module_dict[(r+' ode_'+str(i))] = \
                        self.ode_fcn(self.pop[r], init_params['default']).to(self.device)

        self.ode_mods = nn.ModuleDict(
            module_dict
            )


