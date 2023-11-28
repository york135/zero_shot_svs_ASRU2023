import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ResNetBlocks import *
import librosa
import sys, time, pickle
from tqdm import tqdm

# This amazing code is from clovaai/voxceleb_trainer
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
                )

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def pre_net(self, x):
        x = self.torchfb(x)+1e-6
        if self.log_input: x = x.log()
        out = self.instancenorm(x).unsqueeze(1)
        return out

    def forward(self, x):

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(enabled=False):

        # x = self.torchfb(x)+1e-6
        # if self.log_input: x = x.log()
        # out = self.instancenorm(x).unsqueeze(1)


        # Porting to the svs model !

        # x = self.conv1(out)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0],-1,x.size()[-1])
        # print (x.shape)

        w = self.attention(x)

        if self.encoder_type == "SAP":
            output = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            output = torch.cat((mu,sg),1)

        output = output.view(output.size()[0], -1)
        # print (output.shape)
        output = self.fc(output)

        local_outputs = []
        for i in range(0, x.shape[2], 8):
            start = i
            end = min(i + 8, x.shape[2])

            w = self.attention(x[:,:,start:end])
            mu = torch.sum(x[:,:,start:end] * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x[:,:,start:end]**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            local_output = torch.cat((mu,sg),1)

            local_output = local_output.view(local_output.size()[0], -1)
            local_output = self.fc(local_output)
            local_outputs.append(local_output.unsqueeze(1))

        local_outputs = torch.cat(local_outputs, dim=1)

        # print (output.shape)
        # return x, out.squeeze(1)
        return output, local_outputs

class DummySpeakerNet(nn.Module):

    def __init__(self):
        super(DummySpeakerNet, self).__init__()

    def forward(self, data, label=None):
        return

def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = DummySpeakerNet()
    model.__S__ = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)

    checkpoint = torch.load(kwargs["initial_model"], map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    return model.__S__

def load_pkl(dataset_prefix):
    print ('Start loading dataset,', time.time(), 'from', dataset_prefix)
    
    waveform_data = []

    singer_ids = ['f1', 'f2', 'm1', 'm2']

    for singer_id in singer_ids:
        cur_path = dataset_prefix + '_' + singer_id + '.pkl'
        with open(cur_path, 'rb') as f:
            _, _, _, waveforms = pickle.load(f)
            waveform_data.append(waveforms)

    print ('Dataset loaded,', time.time())
    return waveform_data

if __name__ == "__main__":
    dataset_prefix = sys.argv[1]
    output_prefix = sys.argv[2]

    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    model = MainModel(nOut=512, log_input=True, n_mels=64, encoder_type="ASP", initial_model="baseline_v2_ap.model").to(device)
    model.eval()
    # print (model.training)
    waveform_data = load_pkl(dataset_prefix)

    singer_ids = ['f1', 'f2', 'm1', 'm2']

    with torch.no_grad():
        for i in range(len(waveform_data)):
            cur_embedding = []
            cur_local_embedding = []
            for j in tqdm(range(len(waveform_data[i]))):
                y = librosa.resample(waveform_data[i][j], 24000, 16000)
                # print (y.shape)
                # y = F.interpolate(torch.tensor(waveform_data[i][j]).unsqueeze(0).unsqueeze(1), scale_factor=2.0/3.0, mode='linear').squeeze(1)
                y = torch.tensor(y)

                data = y.reshape(-1,y.size()[-1]).to(device)
                # print (data)
                outp = model.pre_net(data)
                outp, local_outputs = model(outp)
                outp, local_outputs = outp.cpu(), local_outputs.cpu()
                # print (outp.shape, local_outputs.shape)
                # print (outp[0,:10])
                # print (y.shape, data.shape, outp.shape)
                cur_embedding.append(outp)
                cur_local_embedding.append(local_outputs)

            cur_path = output_prefix + '_' + singer_ids[i] + '.pkl'
            print ("Output embedding at", cur_path)
            with open(cur_path, 'wb') as f:
                pickle.dump([cur_embedding, cur_local_embedding], f)

