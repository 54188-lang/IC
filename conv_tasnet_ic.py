import torch
from torch.autograd import Variable
from config import config
from utility import models_attention_set


class TasNet(torch.nn.Module):
    def __init__(self):
        super(TasNet, self).__init__()
        # -------------- 超参 ----------------
        self.mic_num = config.model_mic_num      # 2
        self.num_spk = config.model_num_spk      # 4
        self.enc_dim = config.model_enc_dim      # 512
        self.feature_dim = config.model_feature_dim
        self.ch_dim = config.model_ch_dim

        self.win = int(config.sample_rate * config.model_win / 1000)
        self.stride = self.win // 2

        self.layer = config.model_layer
        self.stack = config.model_stack
        self.kernel = config.model_kernel
        self.causal = config.model_causal

        # -------------- 网络 ----------------
        # 编码器：2 路 → 512 维特征
        self.encoder = torch.nn.Conv1d(
            self.mic_num,
            self.enc_dim,
            self.win,
            bias=False,
            stride=self.stride
        )

        # TCN：4-D 接口 (B, C, N, L)
        self.TCN = models_attention_set.TCN(
            mic_num=self.mic_num,
            ch_dim=self.ch_dim,
            input_dim=self.enc_dim,
            output_dim=self.enc_dim * self.num_spk,
            BN_dim=self.feature_dim,
            hidden_dim=self.feature_dim * 4,
            layer=self.layer,
            stack=self.stack,
            kernel=self.kernel,
            causal=self.causal
        )
        self.receptive_field = self.TCN.receptive_field

        # 解码器：一次性出 4 路
        self.decoder = torch.nn.ConvTranspose1d(
            self.enc_dim * self.num_spk,
            self.num_spk,
            self.win,
            bias=False,
            stride=self.stride,
            groups=self.num_spk  # 关键：一次产生 4 路
        )
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    # ---------- 工具 ----------
    def pad_signal(self, input):                     
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    # ---------- 前向 ----------
    def forward(self, input):
        # print("(conv_tasnet_ic.forward)input:",input.shape)
        output, rest = self.pad_signal(input)          # (B, 2, T)

        B, C, T = output.shape                         # C=2
        # print("(conv_tasnet_ic.forward)output:",B,C,T)
        # 1. 编码器直接吃 2 路： (B, 2, T) → (B, 512, L)
        enc = self.encoder(output)
        enc_split = enc.view(B, 4, 128, -1)            # (B, 4, 128, L)

        # print("(conv_tasnet_ic.forward)enc:",enc.shapes)
        # 2. TCN 需要 (B, C, N, L)，把 512 当作 N，麦克风维 C=1
        enc_4d = enc.unsqueeze(1).repeat(1, 2, 1, 1)   # (B, 1, 512, L)
        # print("(conv_tasnet_ic.forward)enc_4d:",enc_4d.shape)
        # 3. TCN 输出： (B, 1, 4×512, L)
        masks_4d = torch.sigmoid(self.TCN(enc_4d))

        # raw_output = self.TCN(enc_4d)  # (B, 1, 4*512, L)
        # raw_view = raw_output.view(B, 1, 4, 512, -1)
        # print("Raw TCN output for source 1 (correct slice) max:", raw_view[0, 0, 0].abs().max().item())
        # print("Raw TCN output for source 2 (correct slice) max:", raw_view[0, 0, 1].abs().max().item())
        # print("Raw TCN output for source 3 (correct slice) max:", raw_view[0, 0, 2].abs().max().item())
        # print("Raw TCN output for source 4 (correct slice) max:", raw_view[0, 0, 3].abs().max().item())
        # sigmoid_view = torch.sigmoid(raw_view)
        # print("Sigmoid output for source 1 (correct) max:", sigmoid_view[0, 0, 0].max().item())
        # print("Sigmoid output for source 2 (correct) max:", sigmoid_view[0, 0, 1].max().item())
        # print("Sigmoid output for source 3 (correct) max:", sigmoid_view[0, 0, 2].max().item())
        # print("Sigmoid output for source 4 (correct) max:", sigmoid_view[0, 0, 3].max().item())
        
        # 4. 取参考麦克风（第0路）的掩码
        masks = masks_4d.squeeze(1)                    # (B, 4×512, L)
        masks = masks.view(B, self.num_spk, self.enc_dim, -1)
        ref_enc = enc.unsqueeze(1)                     # (B, 1, 512, L)
        masked = ref_enc * masks                       # (B, 4, 512, L)
        # decoder weight shape: (4, 512, kernel)
        # 5. 解码：一次性 4 路 → (B, 4, T_out)
        wav = self.decoder(masked.view(B, 4 * self.enc_dim, -1))
        wav = wav[:, :, self.stride:-(rest + self.stride)].contiguous()
        return wav.view(B, 4, -1)