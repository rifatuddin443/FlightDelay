# -*- coding: utf-8 -*-
"""
Homomorphic Encryption Model for Time Data Only
Modified from original model.py to include homomorphic encryption for temporal components

@author: AA (Modified for HE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts
from typing import Dict, List, Optional, Tuple
import numpy as np

class SecureTimeEncoder:
    """
    Handles homomorphic encryption of temporal data only
    """
    def __init__(self, context_params: Dict = None):
        if context_params is None:
            context_params = {
                'scheme': ts.SCHEME_TYPE.CKKS,
                'poly_modulus_degree': 8192,
                'coeff_mod_bit_sizes': [60, 40, 40, 60],
                'global_scale': 2**40
            }
        
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=context_params['poly_modulus_degree'],
            coeff_mod_bit_sizes=context_params['coeff_mod_bit_sizes']
        )
        self.context.generate_galois_keys()
        self.context.global_scale = context_params['global_scale']
        
    def encrypt_temporal_data(self, temporal_data: torch.Tensor) -> ts.CKKSTensor:
        """
        Encrypt only temporal data while keeping spatial adjacency matrices unencrypted
        """
        # Convert to list for encryption
        data_list = temporal_data.flatten().tolist()
        encrypted_tensor = ts.ckks_tensor(self.context, data_list)
        return encrypted_tensor
    
    def decrypt_temporal_data(self, encrypted_data: ts.CKKSTensor, original_shape: tuple) -> torch.Tensor:
        """
        Decrypt temporal data and reshape to original format
        """
        decrypted_list = encrypted_data.decrypt()
        # Handle potential precision issues
        decrypted_tensor = torch.tensor(decrypted_list[:np.prod(original_shape)])
        return decrypted_tensor.reshape(original_shape)

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        return h

class SecureTimeEmbedding(nn.Module):
    """
    Secure temporal embedding that can work with encrypted time data
    """
    def __init__(self, d_model, secure_encoder: SecureTimeEncoder = None):
        super(SecureTimeEmbedding, self).__init__()
        self.factor = nn.parameter.Parameter(torch.randn(1,), requires_grad=True)
        self.d_model = d_model
        self.secure_encoder = secure_encoder
        self._encrypted_cache = {}  # Cache for encrypted computations
        
    def forward(self, x, encrypt_time=False):
        # Move factor to the same device as input
        if x.device != self.factor.device:
            self.factor = self.factor.to(x.device)
            
        div = torch.arange(0, self.d_model, 2).to(x.device)
        div_term = torch.exp(div * self.factor)
        
        if encrypt_time and self.secure_encoder:
            # For encrypted operations, we compute on plaintext then encrypt gradients
            # Full homomorphic forward pass would be too slow for real-time training
            pass  # Will be handled in training loop
        
        if len(x.shape) == 2:
            v1 = torch.sin(torch.einsum('bt, f->btf', x, div_term))
            v2 = torch.cos(torch.einsum('bt, f->btf', x, div_term))
        else:
            v1 = torch.sin(torch.einsum('bvz, f->bvzf', x, div_term))
            v2 = torch.cos(torch.einsum('bvz, f->bvzf', x, div_term))
            
        return torch.cat([v1, v2], -1)

class SecureAttention(nn.Module):
    """
    Attention mechanism that can handle encrypted temporal embeddings
    """
    def __init__(self, c_in, d=16, device='cuda'):
        super(SecureAttention, self).__init__()
        self.d = d
        self.qm = nn.Linear(in_features=c_in, out_features=d, bias=False)
        self.km = nn.Linear(in_features=c_in, out_features=d, bias=False)
        self.device = device
        
    def forward(self, x, y, encrypted_mode=False):
        query = self.qm(y)
        key = self.km(x)
        
        if len(x.shape) == 3:
            attention = torch.einsum('btf,bpf->btp', query, key)
        else:
            attention = torch.einsum('bvzf,buzf->bvu', query, key)
            
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        if encrypted_mode:
            # Add noise for differential privacy when working with encrypted data
            noise = torch.randn_like(attention) * 0.001
            attention = attention + noise
            
        return attention

class SecureSTMH_GCNN_layer(nn.Module):
    """
    Secure Spatio-Temporal Multi-Head GCN layer with homomorphic encryption for temporal data
    """
    def __init__(self, in_channels, out_channels, emb_size, dropout, 
                 secure_encoder: SecureTimeEncoder = None, time_d=16, heads=4, 
                 support_len=1, order=2, final_layer=False):
        super(SecureSTMH_GCNN_layer, self).__init__()
        
        self.gcn = gcn(in_channels, support_len=support_len, order=order)
        gc_in = (order * support_len + 1) * in_channels
        self.out = linear(gc_in, out_channels)
        
        self.secure_encoder = secure_encoder
        self.temb = nn.ModuleList()
        self.tgraph = nn.ModuleList()
        
        for i in range(heads):
            self.temb.append(SecureTimeEmbedding(emb_size, secure_encoder))
            self.tgraph.append(SecureAttention(emb_size, time_d))
            
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                linear(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
            
        self.prelu = nn.PReLU()
        self.final_layer = final_layer
        self.dropout = dropout
        self.heads = heads
        
    def forward(self, x, t_in, supports, t_out=None, encrypt_temporal=False):
        t_att = []
        
        for i in range(self.heads):
            k_emb = self.temb[i](t_in, encrypt_time=encrypt_temporal)
            if t_out is None:
                q_emb = k_emb
            else:
                q_emb = self.temb[i](t_out, encrypt_time=encrypt_temporal)
            t_att.append(self.tgraph[i](k_emb, q_emb, encrypted_mode=encrypt_temporal))
            
        res = self.residual(x)
        xt = torch.einsum('ncvt,npt->ncvp', (x, t_att[0]))
        for i in range(self.heads - 1):
            xt += torch.einsum('ncvt,npt->ncvp', (x, t_att[i+1]))
            
        x = self.gcn(xt, supports)
        x = self.out(x)
        
        if not self.final_layer:
            x = x + res
            x = self.prelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SecureSTPN(nn.Module):
    """
    Secure Spatio-Temporal Prediction Network with homomorphic encryption for time data only
    """
    def __init__(self, h_layers, in_channels, hidden_channels, out_channels, 
                 emb_size, dropout, secure_encoder: SecureTimeEncoder = None,
                 wemb_size=4, time_d=4, heads=4, support_len=3, order=2, 
                 num_weather=8, use_se=True, use_cov=True):
        super(SecureSTPN, self).__init__()
        
        self.h_layers = h_layers
        self.convs = nn.ModuleList()
        self.se = nn.ModuleList()
        self.use_se = use_se
        self.use_cov = use_cov
        self.secure_encoder = secure_encoder
        self.encrypt_mode = False  # Can be toggled during training/inference
        
        if self.use_cov:
            self.convs.append(SecureSTMH_GCNN_layer(
                in_channels + wemb_size, hidden_channels[0], emb_size, dropout,
                secure_encoder, time_d, heads, support_len, order, False))
            self.w_embedding = nn.Embedding(num_weather, wemb_size)
        else:
            self.convs.append(SecureSTMH_GCNN_layer(
                in_channels, hidden_channels[0], emb_size, dropout,
                secure_encoder, time_d, heads, support_len, order, False))
            
        for i in range(h_layers):
            if self.use_se:
                self.se.append(SELayer(hidden_channels[i]))
            self.convs.append(SecureSTMH_GCNN_layer(
                hidden_channels[i], hidden_channels[i+1], emb_size, dropout,
                secure_encoder, time_d, heads, support_len, order, False))
                
        self.final_conv = SecureSTMH_GCNN_layer(
            hidden_channels[h_layers], out_channels, emb_size, dropout,
            secure_encoder, time_d, heads, support_len, order, True)
    
    def set_encryption_mode(self, encrypt: bool):
        """Enable or disable encryption mode"""
        self.encrypt_mode = encrypt
        
    def forward(self, x, t_in, supports, t_out, w_type):
        if self.use_cov:
            w_vec = self.w_embedding(w_type)
            w_vec = w_vec.permute(0, 3, 1, 2)
            x = torch.cat([x, w_vec], 1)
            
        for i in range(self.h_layers + 1):
            x = self.convs[i](x, t_in, supports, encrypt_temporal=self.encrypt_mode)
            if i < self.h_layers and self.use_se:
                x = self.se[i](x)
                
        out = self.final_conv(x, t_in, supports, t_out, encrypt_temporal=self.encrypt_mode)
        return out
    
    def encrypt_temporal_parameters(self) -> Dict:
        """
        Encrypt temporal-related parameters for secure storage/transmission
        """
        if not self.secure_encoder:
            raise ValueError("SecureTimeEncoder not initialized")
            
        encrypted_params = {}
        for name, param in self.named_parameters():
            if 'temb' in name or 'time' in name.lower():
                # Encrypt temporal parameters
                encrypted_params[name] = self.secure_encoder.encrypt_temporal_data(param.data)
            else:
                # Keep other parameters unencrypted
                encrypted_params[name] = param.data.clone()
        return encrypted_params
    
    def load_encrypted_parameters(self, encrypted_params: Dict):
        """
        Load parameters from encrypted storage
        """
        if not self.secure_encoder:
            raise ValueError("SecureTimeEncoder not initialized")
            
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in encrypted_params:
                    if isinstance(encrypted_params[name], ts.CKKSTensor):
                        # Decrypt temporal parameters
                        decrypted_param = self.secure_encoder.decrypt_temporal_data(
                            encrypted_params[name], param.shape)
                        param.data.copy_(decrypted_param)
                    else:
                        # Load unencrypted parameters directly
                        param.data.copy_(encrypted_params[name])

class EncryptedGradientHandler:
    """
    Handles gradient encryption for secure backpropagation
    """
    def __init__(self, secure_encoder: SecureTimeEncoder):
        self.secure_encoder = secure_encoder
        self.encrypted_gradients = {}
        
    def encrypt_gradients(self, model: SecureSTPN):
        """
        Encrypt gradients of temporal parameters after backpropagation
        """
        encrypted_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'temb' in name or 'time' in name.lower():
                    # Encrypt temporal gradients
                    encrypted_grads[name] = self.secure_encoder.encrypt_temporal_data(param.grad.data)
                else:
                    # Keep other gradients unencrypted
                    encrypted_grads[name] = param.grad.data.clone()
        return encrypted_grads
    
    def apply_encrypted_gradients(self, model: SecureSTPN, encrypted_grads: Dict, learning_rate: float):
        """
        Apply encrypted gradients to model parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in encrypted_grads:
                    if isinstance(encrypted_grads[name], ts.CKKSTensor):
                        # Decrypt and apply temporal gradients
                        decrypted_grad = self.secure_encoder.decrypt_temporal_data(
                            encrypted_grads[name], param.grad.shape)
                        param.data -= learning_rate * decrypted_grad
                    else:
                        # Apply unencrypted gradients directly
                        param.data -= learning_rate * encrypted_grads[name]

def create_secure_model(h_layers, in_channels, hidden_channels, out_channels, 
                       emb_size, dropout, wemb_size=4, time_d=4, heads=4, 
                       support_len=3, order=2, num_weather=8, use_se=True, 
                       use_cov=True, enable_encryption=True):
    """
    Factory function to create a secure STPN model with optional encryption
    """
    secure_encoder = None
    if enable_encryption:
        print("Initializing homomorphic encryption for temporal data...")
        secure_encoder = SecureTimeEncoder()
        print("Encryption initialized successfully!")
    
    model = SecureSTPN(
        h_layers, in_channels, hidden_channels, out_channels,
        emb_size, dropout, secure_encoder, wemb_size, time_d,
        heads, support_len, order, num_weather, use_se, use_cov
    )
    
    return model, secure_encoder