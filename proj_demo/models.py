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
        self.vag=nn.Sequential(
            nn.Conv2d(1,256,(1024,1)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,64,1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,8,1),
            nn.BatchNorm2d(8),
            nn.Softmax2d(),
            )
        self.aag=nn.Sequential(
            nn.Conv2d(1,64,(128,1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,8,1),
            nn.BatchNorm2d(8),
            nn.Softmax2d(),
            )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        #feats=torch.cat((vfeat, afeat),1)
        vag=self.vag(vfeat.unsqueeze(1))
        aag=self.aag(afeat.unsqueeze(1))
        res=torch.pow(torch.sum(torch.pow(vag-aag,2),1),0.5)
        res=torch.sum(res.squeeze(),1)
        return res.squeeze()


class Test2(nn.Module):
    def __init__(self):
        super(Test2, self).__init__()
        self.vag=nn.Sequential(
            nn.Conv2d(1,64,(1024,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            )
        self.aag=nn.Sequential(
            nn.Conv2d(1,64,(128,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            )
        self.fc=nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vag=self.vag(vfeat.unsqueeze(1)).view(-1,120,64)
        aag=self.aag(afeat.unsqueeze(1)).view(-1,120,64)
        #res=torch.pow(torch.sum(torch.pow(vag-aag,2),1),0.5)
        #res=torch.sum(res.squeeze(),1)
        feats=torch.cat((vag,aag),2)
        res=torch.mean(self.fc(feats), 1)
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
        return loss
