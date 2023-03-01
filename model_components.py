# Base components needed for encoder and decoder in transformer


import torch
import torch.nn as nn

class grad_skip_softmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.sm(x)

    def backward(self, grad):
        # skip gradient through the softmax on backward pass
        return grad
    
class grad_skip_logsoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lsm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        return self.lsm(x)
    
    def backward(self, grad):
        return 1/grad
        
class gru(nn.Module):
    # 'gated-recurrent-unit type gating' as seen in GTrXL paper
    def __init__(self, dim, b_g = 1) -> None:
        super().__init__()

        self.w_r = nn.Linear(dim, dim, bias = False)
        self.u_r = nn.Linear(dim, dim, bias = False)

        self.w_z = nn.Linear(dim, dim, bias = True)
        self.u_z = nn.Linear(dim, dim, bias = True)
        self.b_g = b_g # this is used to hack initial bias of the above to be below zero, such that gate is initialized close to identity
        self.w_g = nn.Linear(dim, dim, bias = False)
        self.u_g = nn.Linear(dim, dim, bias = False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):

        r = self.sigmoid(self.w_r(y) + self.u_r(x))
        z = self.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g) # when zero, gate passes identity of residual
        h_hat = self.tanh(self.w_g(y) + self.u_g(r * x))
        g = (1-z)*x + z * h_hat
        return g
        

class mlp(nn.Module):
    # 1d temporal convolution
    # no communication between tokens, uses same kernel for each token spot
    def __init__(self, embed_dim, internal_dim) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, internal_dim),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(internal_dim, embed_dim)
        ) # no second relu at output of mlp

    def forward(self, input):
        return self.block(input)


class cross_attention(nn.Module):
    def __init__(self, embed_dimension, num_heads) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dimension,
            num_heads=num_heads,
            )
    
    def forward(self, x, enc):
        x = x.squeeze()
        return self.attention(x, enc, enc)[0]

class self_attention(nn.Module):
    def __init__(self, embed_dimension, num_heads) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dimension,
            num_heads=num_heads,
            )
    
    def forward(self, x):

        return self.attention(x, x, x)[0]




class Smear_key(nn.Module):

    def __init__(self,
    sequence_length,
    heads
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, heads, sequence_length - 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, k):
        itrp = self.sigmoid(self.alpha)
        smear = k[:,:,1:,:]*itrp + k[:,:,:-1,:]*(1-itrp)
        return torch.cat([k[:,:, 0:1, :], smear], dim = 2)

class decoder_mha(nn.Module):
    #Masked smeared self attention
    def __init__(self, model_dim, sequence_length, heads) -> None:
        super().__init__()
        self.mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1) # make batch, heads, seq,seq
        self.model_dim = model_dim
        self.sequence_length = sequence_length
        self.heads = heads
        self.key_dim = model_dim // heads
        self.W_q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_k = nn.Linear(model_dim, model_dim, bias=False)
        self.W_v = nn.Linear(model_dim, model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim, bias=True)
        self.ln = nn.LayerNorm(model_dim)
        self.smear = Smear_key(sequence_length, heads)

    def forward(self,x):
        # batch, sequence, model_dim
        q = self.W_q(x).view(-1, self.sequence_length, self.heads, self.key_dim).transpose(1,2)
        k = self.W_k(x).view(-1, self.sequence_length, self.heads, self.key_dim).transpose(1,2)
        v = self.W_v(x).view(-1, self.sequence_length, self.heads, self.key_dim).transpose(1,2)
        k = self.smear(k)
        # batch, heads, sequence, dim // heads
        key_dim = k.shape[-1:][0]
        scores = q @ k.transpose(2,3) / key_dim**.5
        scores += self.mask
        attn = torch.softmax(scores, dim = 3)
        mha = attn @ v
        mha = mha.transpose(1, 2).contiguous().view(-1, self.sequence_length, self.model_dim)
        out = self.output(mha)
        # batch, sequence, model_dim
        return out

class critic_head(nn.Module):
    
    def __init__(self,
        value_points,
        model_dim,
        critic_dim
        ) -> None:
        super().__init__()

        self.value_outs = nn.Sequential(
            nn.Linear(model_dim,critic_dim),
            nn.ReLU(),
            nn.Linear(critic_dim, value_points),
            nn.ReLU()
        )

        self.gating = nn.Sequential(
            nn.Linear(model_dim, critic_dim),
            nn.ReLU(),
            nn.Linear(critic_dim, value_points),
            nn.LayerNorm(value_points),
            grad_skip_softmax()
        )

    def forward(self, x):
        values = self.value_outs(x)
        gate = self.gating(x)
        weighted_values = values * gate
        return weighted_values.sum(dim=-1)

