import torch 


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)


    def forward(self, src, tgt):
        encoder_output = self.encoder.forward(src)
        decoder_output = self.decoder.forward(tgt, encoder_output)
        return decoder_output


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
            )


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x



class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
            )


    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer.forward(x, encoder_output)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

    def forward(self, x):
        attention_output = self.self_attention.forward(x, x, x)
        x = self.layer_norm1.forward(x + attention_output)
        forward_output = self.feed_forward.forward(x)
        x = self.layer_norm2.forward(x + forward_output)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_masked_attention = MultiMaskedHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.layer_norm3 = LayerNormalization(d_model)

    def forward(self, x, encoder_output):
        masked_attention_output = self.self_masked_attention.forward(x, x, x)
        x = self.layer_norm1.forward(x + masked_attention_output)
        encoder_attention_output = self.encoder_attention.forward(x, encoder_output, encoder_output)
        x = self.layer_norm2.forward(x + encoder_attention_output)
        forward_output = self.feed_forward.forward(x)
        x = self.layer_norm3.forward(x + forward_output)
        return x




class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.query_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.key_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.value_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.out_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention(self.head_dim)

    def forward(self, query, key, value):
        query = [self.query_linear[i](query) for i in range(self.num_heads)]
        key = [self.key_linear[i](key) for i in range(self.num_heads)]
        value = [self.value_linear[i](value) for i in range(self.num_heads)]
        attention_outputs = [self.attention.forward(query[i], key[i], value[i]) for i in range(self.num_heads)]
        concatenated = torch.cat(attention_outputs, dim=-1)
        output = self.out_linear(concatenated)
        return output
        

class MultiMaskedHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.query_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.key_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.value_linear = torch.nn.ModuleList([torch.nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.out_linear = torch.nn.Linear(self.d_model, d_model)
        self.attention = Attention(self.head_dim)

    def forward(self, query, key, value):
        query = [self.query_linear[i](query) for i in range(self.num_heads)]
        key = [self.key_linear[i](key) for i in range(self.num_heads)]
        value = [self.value_linear[i](value) for i in range(self.num_heads)]
        attention_outputs = []
        for i in range(self.num_heads):
            scores = torch.matmul(query[i], key[i].transpose(-2, -1)) * (self.head_dim ** -0.5)
            mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1))).to(scores.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, value[i])
            attention_outputs.append(attention_output)
        concatenated = torch.cat(attention_outputs, dim=-1)
        output = self.out_linear(concatenated)
        return output



class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        p_emb = torch.zeros(seq_len, self.d_model)
        for i in range(seq_len):
            for j in range(0, self.d_model, 2):
                div_term = 1 / (10000 ** (j / self.d_model))
                p_emb[i, j] = torch.sin(position[i] * div_term)
                if j + 1 < self.d_model:
                    p_emb[i, j + 1] = torch.cos(position[i] * div_term)
        p_emb = p_emb.unsqueeze(0)
        return x + p_emb

    






class LayerNormalization(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + 1e-6)
        return self.gamma * x_normalized + self.beta


class Attention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, value)
        return output