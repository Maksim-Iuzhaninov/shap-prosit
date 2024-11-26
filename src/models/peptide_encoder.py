import torch as th
from torch import nn
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shap-prosit/src/models")
import model_parts as mp

def model_init(module):
    if isinstance(module, mp.SelfAttention):
        module.qkv.weight = nn.init.normal_(module.qkv.weight, 0, 0.03)
        module.qkv.bias = nn.init.zeros_(module.qkv.bias)
        module.Wo.weight = nn.init.normal_(module.Wo.weight, 0, 0.01)
        module.Wo.bias = nn.init.zeros_(module.Wo.bias)
    if isinstance(module, mp.FFN):
        module.W1.weight = nn.init.normal_(module.W1.weight, 0, 0.03)
        module.W1.bias = nn.init.zeros_(module.W1.bias)

class PeptideEncoder(nn.Module):
    def __init__(self,
                 running_units=256,
                 penult_units=512,
                 final_units=174,
                 tokens=30,
                 max_charge=8,
                 d=64,
                 h=8,
                 ffn_mult=2,
                 dropout=0,
                 alphabet=False,
                 prenorm=False,
                 norm_type='layer',
                 depth=6,
                 prec_type='preembed',
                 prec_units=256,
                 use_charge=True,
                 use_energy=False,
                 final_act='identity'
                 ):
        super(PeptideEncoder, self).__init__()
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.at_least_1 = self.use_charge or self.use_energy

        # Positional information
        arng = th.arange(0, 100, 1, dtype=th.float32)
        self.pos = nn.Parameter(
            mp.FourierFeatures(arng, 1, 500, running_units), 
            requires_grad=False
        )
        #self.pos_seq = nn.Sequential(
        #    nn.Linear(running_units, running_units),
        #    nn.SiLU(),
        #)
        self.alpha_pos = nn.Parameter(th.tensor(0.1), requires_grad=True)

        # Modified sequence embedding
        self.sequence_embedding = nn.Embedding(tokens, running_units)
        
        # Precursor information
        self.prec_units = prec_units
        if prec_type in ['pretoken', 'preembed', 'preembed_wb']:
            if use_charge:
                self.charge_embedder = nn.Embedding(max_charge, prec_units)
            if use_energy:
                self.ce_embedder = nn.Linear(prec_units, prec_units)
        self.num = sum([use_charge, use_energy])
        if self.num > 0:
            self.unite_precursors = nn.Linear(self.num*prec_units, prec_units)
        
        # Middle
        attention_dict = {
            'indim': running_units,
            'd': d,
            'h': h,
            'dropout': dropout,
            'alphabet': alphabet,
        }
        ffn_dict = {
            'indim': running_units,
            'unit_multiplier': ffn_mult,
            'dropout': dropout,
            'alphabet': alphabet,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                prenorm=prenorm, 
                norm_type=norm_type, 
                embed_type=prec_type if self.at_least_1 else None,
                embed_indim=prec_units,
                is_cross=False,
                channel_alpha=True,
            )
            for i in range(depth)
        ])

        # Last
        norm = nn.LayerNorm if norm_type == 'layer' else nn.BatchNorm1d
        self.penult = nn.Sequential(
            nn.Linear(running_units, penult_units),
            norm(penult_units),
            nn.ReLU()
        )
        if final_act.lower() in ['identity', 'linear']:
            final_act = nn.Identity()
        elif final_act.lower() == 'sigmoid':
            final_act = nn.Sigmoid()
        self.last = nn.Sequential(
            nn.Linear(penult_units, final_units),
            final_act,
        )

        self.global_step = nn.Parameter(th.tensor(0), requires_grad=False)

        self.apply(model_init)

    def Precursors(self, charge, ce):
        lst = []
        if self.use_charge:
            lst.append(self.charge_embedder(charge-1))
        if self.use_energy:
            lst.append(
                self.ce_embedder(
                    mp.FourierFeatures(ce, 0.01, 1.5, self.prec_units)
                )
            )
        joined = nn.functional.silu(th.cat(lst, dim=-1))
        precursor_embedding = self.unite_precursors(joined)

        return precursor_embedding

    def forward(self, intseq, charge=None, energy=None):
        sequence = self.sequence_embedding(intseq)
        out = sequence + self.alpha_pos*self.pos[:intseq.shape[1]][None]

        PreEmb = self.Precursors(charge, energy) if self.at_least_1 else None

        for layer in self.main:
            out = layer(out, embed_feats=PreEmb)
            out = out['out']

        penultimate = self.penult(out)

        return self.last(penultimate).mean(1)

#model = PeptideEncoder(running_units=512, tokens=28, max_charge=8)
#sequence = th.randint(0, 28, (10, 30))
#charge = th.randint(1, 8, (10,))
#out = model(sequence, charge)
#print(out.shape)