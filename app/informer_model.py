import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask=None):
        scale = self.scale or 1. / math.sqrt(queries.size(-1))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -np.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, values)  
        return output, attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        B, L_K, _ = keys.shape
        B, L_V, _ = values.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L_Q, H, -1)
        keys = self.key_projection(keys).view(B, L_K, H, -1)
        values = self.value_projection(values).view(B, L_V, H, -1)

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_Q, -1)
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(attention, d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        y = x.permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.permute(0, 2, 1)
        
        x = x + y
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayer(self_attention, d_model, n_heads=8)
        self.cross_attention = AttentionLayer(cross_attention, d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        new_x = self.self_attention(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        new_x = self.cross_attention(x, enc_output, enc_output, attn_mask=cross_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        
        y = x.permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.permute(0, 2, 1)
        
        x = x + y
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection or nn.Linear(layers[0].norm3.normalized_shape[0], 1)
    
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        x = self.projection(x)
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=512, dropout=0.1):
        super(Informer, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(FullAttention(), d_model, d_ff, dropout) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [DecoderLayer(FullAttention(mask_flag=True), FullAttention(), d_model, d_ff, dropout) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x_enc, x_dec):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        
        dec_out = self.dec_embedding(x_dec)
        device = x_enc.device
        seq_len = x_dec.size(1)
        self_attn_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        dec_out = self.decoder(dec_out, enc_out, self_attn_mask=self_attn_mask)
        return dec_out

def load_model(weights_path, out_len, device='cpu'):
    model = Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=60,
        label_len=30,
        out_len=out_len,
        d_model=512,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.2
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_data, out_len, label_len=30, device='cpu'):
    # input_data: np.ndarray (seq_len, 1)
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0) # (1, seq_len, 1)
    x_enc = torch.FloatTensor(input_data).to(device)

    x_dec = torch.zeros((x_enc.size(0), label_len + out_len, x_enc.size(2))).to(device)
    x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]

    with torch.no_grad():
        output = model(x_enc, x_dec)  # (B, out_len, 1)
    return output.cpu().numpy()
