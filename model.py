import torch
import torch.nn as nn
import torch.nn.init as init



class PainPredModel(nn.Module):
    def __init__(self, dim_input, dim_output, dim_emb=64, dim_alpha=64, dim_beta=64, dropout_context=0.2, batch_first=True):
        super(RETAIN_core, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Linear(dim_input, dim_emb, bias=False),
        )
        # init.xavier_uniform_(self.embedding[1].weight) # initialize linear layer, other choice: xavier_normal
        init.uniform_(self.embedding[0].weight, a=-0.5, b=0.5)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)
        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)

        # init.xavier_uniform_(self.alpha_fc.weight)
        init.uniform_(self.alpha_fc.weight, a=-0.5, b=0.5)
        init.zeros_(self.alpha_fc.bias)

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)
        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        # init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        # init.xavier_uniform_(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        init.uniform_(self.beta_fc.weight, a=-0.5, b=0.5)
        init.zeros_(self.beta_fc.bias)

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output),
        )
        # init.xavier_uniform_(self.output[1].weight)
        init.uniform_(self.output[1].weight, a=-1, b=1)
        init.zeros_(self.output[1].bias)

    def masked_softmax(self, batch_tensor, mask):
        exp = torch.exp(batch_tensor)
        masked_exp = exp * mask
        sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
        return masked_exp / sum_masked_exp

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]




        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        # mask -> batch_size X max_len X 1
        mask = Variable(torch.FloatTensor(
            [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
                        requires_grad=False)
        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = self.masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        beta = torch.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        att_dict = {
            'w_out': self.output[1].weight,
            'w_emb': self.embedding[0].weight,
            'alpha': alpha,
            'beta': beta
        }

        return logit, att_dict
