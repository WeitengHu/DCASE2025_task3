"""
参考 AN EXPERIMENTAL STUDY ON SOUND EVENT LOCALIZATION AND DETECTION
UNDER REALISTIC TESTING CONDITIONS的ResNet-Conformer的模型图
"""

"""
    对DCASE2025:
    Forward pass for the SELD model.
    audio_feat: Tensor of shape (batch_size, n_feat_channel, 251, 64) (stereo spectrogram input).
    vid_feat: Optional tensor of shape (batch_size, 50, 7, 7) (visual feature map).
    Returns:  Tensor of shape
                (batch_size, 50, 117) - audio - multiACCDOA.
                (batch_size, 50, 39)  - audio - singleACCDOA.
                (batch_size, 50, 156) - audio_visual - multiACCDOA.
                (batch_size, 50, 52) - audio_visual - singleACCDOA.

"""
import torch
import torch.nn as nn
# from models.ResNet.resnet import ResNet, resnet18
# from models.conformer.encoder import ConformerBlocks
# from models.AvgMaxPooling import AvgMaxPooling2d
from models.ResNet.resnet import ResNet, resnet18
from models.conformer.encoder import ConformerBlocks
from models.AvgMaxPooling import AvgMaxPooling2d
class ResNetConformer(nn.Module):
    def __init__(
            self,params, in_feat_shape,
            encoder_dim=256,
            num_layers=8,
            # time_pool_size=2,
            dropout_p=0.1,
            num_classes=13,
            out_dim=117, # multiACCDOA格式的输出维度
             # 输入通道数，立体声为2
    ):
        super(ResNetConformer, self).__init__()
        # 初始化ResNet部分
        self.resnet = resnet18(include_top=False, base_channels=24,in_channels=in_feat_shape[1])  # 输入通道数为2（立体声）
        self.resnet_out_dim = self.resnet.out_dim  # resnet18的输出维度
        # resnet与Conformer之间的Conv2d层
        # 注意：ResNet输出后需要先进行频率维度的压缩，然后调整通道数
        # 假设经过4次freq_pool后，频率维度已经很小，我们进行adaptive pooling
        self.freq_adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, T, 1)
        self.encoder_dim = encoder_dim
        self.conv2d = nn.Conv2d(self.resnet_out_dim, encoder_dim, kernel_size=1,bias=False)
        # 初始化Conformer部分
        self.conformer_blocks = ConformerBlocks(encoder_dim=encoder_dim,num_layers=num_layers)
        # self.time_pooling = AvgMaxPooling2d(kernel_size=(time_pool_size,1), stride=(time_pool_size,1))
        self.dnorm = params['dnorm']
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(encoder_dim, out_dim),
            # nn.Tanh()
        )
        self.doa_act = nn.Tanh()
        self.dist_act = nn.Tanh() if self.dnorm == True else nn.ReLU()
    def activation(self, x):
        num_tracks = 3
        num_elements = 3
        num_classes = 13
        B, T, D = x.shape
        x_accdoa = x.reshape(B, T, num_tracks, num_elements, num_classes)
        x_doa = self.doa_act(x_accdoa[:, :, :, 0:2, :])  # act*x, act*y
        x_dist = self.dist_act(x_accdoa[:, :, :, 2:3, :])  # dist
        x_activated = torch.cat([x_doa, x_dist], dim=3)  # concat along element dimension   
        x_activated = x_activated.reshape(B, T, D)
        return x_activated

    def forward(self, x):
        x = self.resnet(x)  # (B,C,T,F')ResNet部分提取特征，每个layer之后只进行frequency维度的pooling
        # x = x.permute(0, 2, 1)
        x = self.freq_adaptive_pool(x)  # (B,C,T,1)
        x = self.conv2d(x) # (B,C,T,1)
        x = x.squeeze(-1)  # (B,C,T)
        
        # 转换为 Conformer 输入格式: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T, C)

        x = self.conformer_blocks(x)  # Conformer部分处理序列数据
        # # 转换回 (B, C, T) 格式以使用 2D pooling
        # x = x.permute(0, 2, 1)  # (B, encoder_dim, T)
        # x = x.unsqueeze(-1)  # (B, encoder_dim, T, 1) - 添加频率维度以使用2D pooling
        
        # # 使用 AvgMaxPooling 进行时间维度池化
        # x = self.time_pooling(x)  # (B, encoder_dim, T', 1)
        
        # x = x.squeeze(-1)  # (B, encoder_dim, T')
        # x = x.permute(0, 2, 1)  # (B, T', encoder_dim)
        x = self.fc(x)  # 最终分类层
        x = self.activation(x)  # 特殊激活函数处理
        return x
    

if __name__ == '__main__':

    test_audio_feat = torch.rand([8, 2, 400, 64])
    test_video_feat = torch.rand([8, 50, 7, 7])
    # test_video_feat = None  # set to none for audio modality

    test_model = ResNetConformer()
    doa = test_model(test_audio_feat)
    print(doa.size())
    # print(test_model)
