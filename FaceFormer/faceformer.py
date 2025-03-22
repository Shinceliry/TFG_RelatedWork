import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # mask = mask.unsqueeze(0) + alibi
    causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))
    mask = causal_mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, tgt_len, src_len):
    mask = torch.ones(tgt_len, src_len)
    if dataset == "BIWI":
        for i in range(tgt_len):
            mask[i, i*2:i*2+2] = 0
        return (mask==1).to(device=device)
    # elif dataset == "vocaset":
    #     for i in range(tgt_len):
    #         mask[i, i] = 0
    #     return (mask==1).to(device=device)
    else:
        return torch.zeros(tgt_len, src_len).to(device=device)  # shape: (tgt_len, src_len)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.n_head = 4
        self.feature_dim = args.feature_dim
        self.device = args.device

        # wav2vec2
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)

        # motion encoder/decoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

        # PPE
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)

        # temporal bias
        max_seq_len = 600
        self.biased_mask = init_biased_mask(self.n_head, max_seq_len, args.period)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim,
            nhead=self.n_head,
            dim_feedforward=2*args.feature_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # style embedding
        num_train_subjects = len(args.train_subjects)
        self.obj_vector = nn.Linear(num_train_subjects, args.feature_dim, bias=False)

    def forward(self, audio, template, vertice, one_hot, criterion,teacher_forcing=True):
        # Batch processing not supported version
        if self.srgs.batch_size == 1:
            '''
            tgt_mask: :math:`(T, T)`.
            memory_mask: :math:`(T, S)`.
            '''
            template = template.unsqueeze(1) # (1,1, V*3)
            obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
            frame_num = vertice.shape[1]
            hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
            if self.dataset == "BIWI":
                if hidden_states.shape[1]<frame_num*2:
                    vertice = vertice[:, :hidden_states.shape[1]//2]
                    frame_num = hidden_states.shape[1]//2
            hidden_states = self.audio_feature_map(hidden_states)

            if teacher_forcing:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb  
                vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
                vertice_input = vertice_input - template
                vertice_input = self.vertice_map(vertice_input)
                vertice_input = vertice_input + style_emb
                vertice_input = self.PPE(vertice_input)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
            else:
                for i in range(frame_num):
                    if i==0:
                        vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                        style_emb = vertice_emb
                        vertice_input = self.PPE(style_emb)
                    else:
                        vertice_input = self.PPE(vertice_emb)
                    tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                    memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                    vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                    vertice_out = self.vertice_map_r(vertice_out)
                    new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                    new_output = new_output + style_emb
                    vertice_emb = torch.cat((vertice_emb, new_output), 1)

            vertice_out = vertice_out + template
            loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
            loss = torch.mean(loss)
            return loss
        
        # Batch processing supported version
        else:
            """
            audio:    (B, audio_len)
            template: (B, V*3) or (B, 1, V*3) before unsqueeze
            vertice:  (B, seq_len, V*3)
            one_hot:  (B, K)   -> K = num_train_subjects (depending on train) or (B, K) / (B, K, ...) (depending on val/test)
            """
            B, T_v, _ = vertice.shape
            device = self.device
            
            hidden_states = self.audio_encoder(audio, self.dataset, frame_num=T_v).last_hidden_state
            hidden_states = self.audio_feature_map(hidden_states)  # (B, S, feature_dim)
            obj_embedding = self.obj_vector(one_hot.to(device))  # (B, feature_dim) 
            
            template = template.unsqueeze(1).to(device)
            if teacher_forcing:
                vertice_input = vertice.to(device) - template  # (B, T_v, V*3)
                vertice_input = self.vertice_map(vertice_input)  # (B, T_v, feature_dim)
                vertice_input = vertice_input + obj_embedding.unsqueeze(1)
                vertice_input = self.PPE(vertice_input)  # (B, T_v, feature_dim)
                local_mask = self.biased_mask[:, :T_v, :T_v].to(device)
                local_mask = local_mask.unsqueeze(0).repeat(B, 1, 1, 1)
                local_mask = local_mask.view(B*self.n_head, T_v, T_v)
                
                S = hidden_states.shape[1]
                base_mem_mask = enc_dec_mask(device, self.dataset, T_v, S)  # (T_v, S)
                base_mem_mask = base_mem_mask.unsqueeze(0).unsqueeze(0).repeat(B, self.n_head, 1, 1)
                base_mem_mask = base_mem_mask.view(B*self.n_head, T_v, S)
                
                vertice_out = self.transformer_decoder(
                    vertice_input,                     # (B, T_v, D)
                    hidden_states,                     # (B, S, D)
                    tgt_mask=local_mask,               # (B*n_head, T_v, T_v)
                    memory_mask=base_mem_mask,         # (B*n_head, T_v, S)
                )
                vertice_out = self.vertice_map_r(vertice_out)      # (B, T_v, V*3)
                vertice_out = vertice_out + template
                loss = criterion(vertice_out, vertice.to(device))
                return torch.mean(loss)

            else:
                outputs = []
                init_inp = torch.zeros_like(template, device=device)  # (B,1,V*3)
                init_inp = self.vertice_map(init_inp)                 # (B,1,feature_dim)
                init_inp = init_inp + obj_embedding.unsqueeze(1)      # (B,1,feature_dim)
                vertice_emb = init_inp

                for i in range(T_v):
                    cur_len = i+1
                    cur_input = self.PPE(vertice_emb)
                    local_mask = self.biased_mask[:, :cur_len, :cur_len].to(device)       # (n_head, i+1, i+1)
                    local_mask = local_mask.unsqueeze(0).repeat(B, 1, 1, 1)               # (B, n_head, i+1, i+1)
                    local_mask = local_mask.view(B*self.n_head, cur_len, cur_len)         # (B*n_head, i+1, i+1)
                    
                    S = hidden_states.shape[1]
                    base_mem_mask = enc_dec_mask(device, self.dataset, cur_len, S)        # (i+1, S)
                    base_mem_mask = base_mem_mask.unsqueeze(0).unsqueeze(0).repeat(B, self.n_head, 1, 1)
                    base_mem_mask = base_mem_mask.view(B*self.n_head, cur_len, S)
                    
                    out_seq = self.transformer_decoder(
                        cur_input,          # (B, i+1, D)
                        hidden_states,      # (B, S, D)
                        tgt_mask=local_mask,
                        memory_mask=base_mem_mask
                    )  # (B, i+1, feature_dim)

                    out_seq = self.vertice_map_r(out_seq)         # (B, i+1, V*3)
                    new_frame = out_seq[:, -1, :] + template[:, 0, :]  # (B, V*3)
                    outputs.append(new_frame.unsqueeze(1))            # (B,1,V*3)
                    
                    next_inp = new_frame - template[:, 0, :]
                    next_inp = self.vertice_map(next_inp)                   # (B, feature_dim)
                    next_inp = next_inp + obj_embedding                     # (B, feature_dim)
                    next_inp = next_inp.unsqueeze(1)                        # (B,1,feature_dim)
                    vertice_emb = torch.cat([vertice_emb, next_inp], dim=1) # (B, i+2, feature_dim)
                    
                vertice_out = torch.cat(outputs, dim=1)  # (B, T_v, V*3)
                loss = criterion(vertice_out, vertice.to(device))
                return torch.mean(loss)

    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out
