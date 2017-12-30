import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, out_size)

    def forward(self, feats):
        h_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        c_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        h_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)
        c_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)

        if feats.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
            h_t2 = h_t2.cuda()
            c_t2 = c_t2.cuda()

        for _, feat_t in enumerate(feats.chunk(feats.size(1), dim=1)):
            h_t, c_t = self.lstm1(feat_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

        # aggregated feature
        feat = h_t2
        return feat

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 512, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)



# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)

# Test of theory
class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.__name__='Test1'
        self.vag=nn.Sequential(
            nn.Conv2d(1,64,(1024,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            )
        self.aag=nn.Sequential(
            nn.Conv2d(1,64,(128,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            )
        self.fc=nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU(True),
        )
        self.rnn=nn.RNNCell(128, 32)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeats=self.vag(vfeat.unsqueeze(1))
        afeats=self.aag(afeat.unsqueeze(1))
        vafeats=torch.cat((vfeats,afeats), 1)
        #vafeats=torch.transpose(torch.cat((vfeat,afeat), 1),1,3)
        #feats=self.fag(vafeats)
        state = Variable(torch.zeros(vafeats.size(0), 32).float(), requires_grad=False).cuda()
        for _, feat in enumerate(vafeats.chunk(vafeats.size(3), dim=3)):
            state = self.rnn(feat.squeeze(), state)
        res=self.fc(state)
        return res.squeeze()


# class Test2(nn.Module):
#     def __init__(self):
#         super(Test2, self).__init__()
#         self.__name__='Test2'
#         self.vag=nn.Sequential(
#             nn.Conv2d(1, 256, (1024, 4)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             )
#         self.aag=nn.Sequential(
#             nn.Conv2d(1, 256, (128, 4)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             )
#         self.fc=nn.Sequential(
#             #nn.Dropout2d(),
#             nn.Linear(512, 256),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(256, 2),
#         )
#         self.softmax=nn.Softmax2d()
#         self.init_params()

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform(m.weight)
#                 nn.init.constant(m.bias, 0)

#     def forward(self, vfeat, afeat):
#         vfeat=self.vag(vfeat.unsqueeze(1))
#         afeat=self.aag(afeat.unsqueeze(1))
#         vafeats=torch.transpose(torch.cat((vfeat,afeat), 1),1,3)
#         feats=torch.transpose(self.fc(vafeats), 1, 3)
#         conf=self.softmax(feats)
#         res=torch.mean(conf[:, 0, :, :], 2)
#         return res.squeeze()

class Test2(nn.Module):
    def __init__(self):
        super(Test3, self).__init__()
        self.__name__='Test3'
        self.sim=nn.Sequential(
            nn.Linear(1024+128, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        self.pred_sim=nn.Sequential(
            nn.Linear(1024+128, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        self.softmax=nn.Softmax2d()
        self.fc=nn.Linear(2, 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat=torch.transpose(vfeat, 1, 2)
        afeat=torch.transpose(afeat, 1, 2)
        vafeats=torch.cat((vfeat[:,:-1,:], afeat[:,:-1,:]), 2)
        pred_feats=torch.cat((vfeat[:,1:,:], afeat[:,:-1,:]), 2)
        sim=self.sim(vafeats)
        pred_sim=self.pred_sim(pred_feats)
        res=torch.mean(self.fc(torch.cat((sim, pred_sim), 2)), 1)
        return res.squeeze()

# class attention(nn.Module):
#     def __init__(self, dim):
#         super(attention, self).__init__()
#         self.attn = nn.Linear(dim, 1)
#     def forward(self, inputs):
#         score = self.attn(inputs):
#         score = F.softmax(score)
#         return score

# def count_params(net):
#     count = 0
#     for name, param in net.named_parameters():
#         print(name, param.numel())
#         count += param.numel()
#     print(net.__class__.__name__, count)


class LSTMAttention(nn.Module):
    def __init__(self, seq, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.seq = seq
        self.olstm = nn.LSTM(input_size, hidden_size,
                             batch_first=True, num_layers=num_layers, dropout=dropout)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, self.seq),
            nn.Softmax()
        )

    def forward(self, x):
        _, (h_t, c_t) = self.olstm(x.detach()[:, 0].unsqueeze(1) * 0)
        output = []
        for i in range(self.seq):
            attn = self.attention(c_t[0]).unsqueeze(1)
            attn_applied = torch.bmm(attn, x)
            _, (h_t, c_t) = self.olstm(attn_applied, (h_t, c_t))
            output.append(h_t[-1])

        return torch.stack(output, dim=1)

class Test3(nn.Module):
    def __init__(self):
        super().__init__()
        # 120x1024/128
        self.vfeature, self.afeature, self.feature, self.seq = 64, 64, 64, 120
        self.vlstm = nn.GRU(1024, self.vfeature, batch_first=True)
        self.alstm = nn.GRU(128, self.afeature, batch_first=True)
        self.olstm = LSTMAttention(self.seq, self.vfeature + self.afeature, self.feature)

        self.dropout = nn.Dropout(p=0.5)  # OK to reuse one

        self.conv = nn.Sequential(
            # 1x120x64
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2, bias=False),
            # 32x60x32
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 32x30x16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 2), stride=(3, 2), bias=False, groups=32),
            # 64x10x8
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 64x5x4
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 4, 64, bias=False),
            nn.BatchNorm1d(64, affine=False),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.init_params()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'conv' in name:
                nn.init.kaiming_normal(param)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
            elif 'bias' in name:
                nn.init.constant(param, 0.01)

    def forward(self, vfeat, afeat, logger=None, epoch=None):
        vfeat = self.dropout(vfeat)
        afeat = self.dropout(afeat)
        vfeat, _ = self.vlstm(vfeat)
        afeat, _ = self.alstm(afeat)
        # 120x64
        if logger:
            logger.add_histogram('vlstm', vfeat.data.cpu().numpy(), epoch, 'auto')
            logger.add_histogram('alstm', afeat.data.cpu().numpy(), epoch, 'auto')

        x = torch.cat((vfeat, afeat), 2)
        # 120x128
        x = self.dropout(x)
        x = self.olstm(x)
        if logger:
            logger.add_histogram('olstm', x.data.cpu().numpy(), epoch, 'auto')
        # 120x64

        x = x.unsqueeze(1)
        x = self.dropout(x)
        x = self.conv(x)
        if logger:
            logger.add_histogram('conv', x.data.cpu().numpy(), epoch, 'auto')
        # 64*5*4

        x = x.view(-1, 64 * 5 * 4)
        x = self.dropout(x)
        x = self.fc(x)
        if logger:
            logger.add_histogram('fc', x.data.cpu().numpy(), epoch, 'auto')

        return x.squeeze()


class Test4(nn.Module):
    def __init__(self):
        super(Test4, self).__init__()
        self.__name__='Test4'
        self.sim=nn.Sequential(
            nn.Linear(1024+128, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        self.pred_sim=nn.Sequential(
            nn.Linear(1024+128, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        self.delayed_sim=nn.Sequential(
            nn.Linear(1024+128, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        self.softmax=nn.Softmax2d()
        self.fc=nn.Linear(3, 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat=torch.transpose(vfeat, 1, 2)
        afeat=torch.transpose(afeat, 1, 2)
        vafeats=torch.cat((vfeat[:,:-1,:], afeat[:,:-1,:]), 2)
        pred_feats=torch.cat((vfeat[:,1:,:], afeat[:,:-1,:]), 2)
        delayed_feats=torch.cat((vfeat[:,:-1,:], afeat[:,1:,:]), 2)
        sim=self.sim(vafeats)
        pred_sim=self.pred_sim(pred_feats)
        delayed_sim=self.delayed_sim(delayed_feats)

        # use attention


        res=torch.mean(self.fc(torch.cat((sim, pred_sim, delayed_sim), 2)), 1)
        return res.squeeze()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        # loss=torch.mean(torch.pow(label-dist, 2))
        return loss
