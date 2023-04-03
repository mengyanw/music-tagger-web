import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from attention_modules import BertConfig, BertEncoder, BertEmbeddings, BertPooler, PositionalEncoding


class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=1, padding=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2, padding=2, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Conv_V(nn.Module):
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, filter_shape,
                              padding=(0, filter_shape[1]//2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out

class Conv_H(nn.Module):
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, filter_length,
                              padding=filter_length//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out

class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''
    def __init__(self, num_classes, config=None):
        super(CRNN, self).__init__()
        # self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'],
        #                                                  n_fft=config['n_fft'],
        #                                                  f_min=config['fmin'],
        #                                                  f_max=config['fmax'],
        #                                                  n_mels=config['n_mels'])
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, 64)
        self.layer2 = Conv_2d(64, 128)
        self.layer3 = Conv_2d(128, 128)
        self.layer4 = Conv_2d(128, 128)

        # RNN
        self.layer5 = nn.GRU(128, 64, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(64, num_classes)

    def forward(self, x):
        # Spectrogram
        # x = self.spec(x)
        # x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x

class Musicnn(nn.Module):
    '''
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    '''

    def __init__(self, num_classes, config=None):
        super(Musicnn, self).__init__()

        # Spectrogram
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons front-end
        m1 = Conv_V(1, 204, (int(0.7 * 96), 7))
        m2 = Conv_V(1, 204, (int(0.4 * 96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel = 512
        self.layer1 = Conv_1d(561, backend_channel, kernel_size=7, stride=1, padding=3, pooling=1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, kernel_size=7, stride=1, padding=3, pooling=1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, kernel_size=7, stride=1, padding=3, pooling=1)

        # Dense
        dense_channel = 500
        self.dense1 = nn.Linear((561 + (backend_channel * 3)) * 2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, num_classes)

    def forward(self, x):
        # Spectrogram
        # x = self.spec(x)
        # x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)

        return out

class FCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self, num_classes, config=None):
        super(FCN, self).__init__()

        # Spectrogram
        # self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'],
        #                                                     n_fft=config['n_fft'],
        #                                                     f_min=config['fmin'],
        #                                                     f_max=config['fmax'],
        #                                                     n_mels=config['n_mels'])
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # FCN
        self.layer1 = Conv_2d(1, 64, kernel_size=3, stride=2, padding=2, pooling=3)
        self.layer2 = Conv_2d(64, 128, kernel_size=3, stride=2, padding=2, pooling=2)
        self.layer3 = Conv_2d(128, 128, kernel_size=3, stride=2, padding=2, pooling=2)
        self.layer4 = Conv_2d(128, 128, kernel_size=4, stride=2, padding=4, pooling=2)
        self.layer5 = Conv_2d(128, 64, kernel_size=3, stride=2, padding=3, pooling=2)

        # Dense
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spectrogram
        # print(x.shape)
        # x = self.spec(x)
        # x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # FCN
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        # torch.Size([4, 1, 128, 10240])
        # torch.Size([4, 64, 32, 2560])
        # torch.Size([4, 128, 8, 640])
        # torch.Size([4, 128, 2, 160])
        # torch.Size([4, 128, 1, 40])
        # torch.Size([4, 64, 1, 10])

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class SampleCNN(nn.Module):
    def __init__(self, n_classes, config=None):
        super(SampleCNN, self).__init__()
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        # 128 x 10240
        self.conv1 = nn.Sequential(
            nn.Conv1d(config['n_mels'], 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 128 x 5120
        self.conv2 = Conv_1d(128, 128)
        # 128 x 2560
        self.conv3 = Conv_1d(128, 128)
        # 128 x 1280
        self.conv4 = Conv_1d(128, 256)
        # 256 x 640
        self.conv5 = Conv_1d(256, 256)
        # 256 x 320
        self.conv6 = Conv_1d(256, 256)
        # 256 x 160
        self.conv7 = Conv_1d(256, 256, kernel_size=2, stride=2, padding=1)
        # 256 x 40
        self.conv8 = Conv_1d(256, 256, kernel_size=2, stride=2, padding=1)
        # 256 x 10
        self.conv9 = Conv_1d(256, 256, kernel_size=2, stride=2, padding=2)
        # 256 x 3
        self.conv10 = Conv_1d(256, 512, kernel_size=2, stride=2, padding=2)
        # 512 x 1
        self.conv11 = Conv_1d(512, 512)
        # 512 x 1
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        # x = self.to_db(x)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        # print(out.shape)
        out = self.conv5(out)
        # print(out.shape)
        out = self.conv6(out)
        # print(out.shape)
        out = self.conv7(out)
        # print(out.shape)
        out = self.conv8(out)
        # print(out.shape)
        out = self.conv9(out)
        # print(out.shape)
        out = self.conv10(out)
        # print(out.shape)
        out = self.conv11(out)
        # print(out.shape)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        # print(out.shape)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        logit = self.activation(out)
        return logit

class ShortChunkCNN_Res(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self, n_class=50, config=None):
        super(ShortChunkCNN_Res, self).__init__()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        n_channels = 128
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x

class CNNSA(nn.Module):
    '''
    Won et al. 2019
    Toward interpretable music tagging with self-attention.
    Feature extraction with CNN + temporal summary with Transformer encoder.
    '''

    def __init__(self, n_class=50, config=None):
        super(CNNSA, self).__init__()

        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        n_channels = 128
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer7 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))

        # Transformer encoder
        bert_config = BertConfig(vocab_size=256,
                                 hidden_size=256,
                                 num_hidden_layers=2,
                                 num_attention_heads=8,
                                 intermediate_size=1024,
                                 hidden_act="gelu",
                                 hidden_dropout_prob=0.4,
                                 max_position_embeddings=700,
                                 attention_probs_dropout_prob=0.5)
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.vec_cls = self.get_cls(256)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_class)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def forward(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Get [CLS] token
        x = x.permute(0, 2, 1)
        x = self.append_cls(x)

        # Transformer encoder
        x = self.encoder(x)
        x = x[-1]
        x = self.pooler(x)

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x
