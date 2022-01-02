import cv2
from kornia.geometry import resize, center_crop
from einops import rearrange
import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, PReLU, Sigmoid, Dropout, Sequential, Module, \
    MaxPool2d, AdaptiveAvgPool2d
from collections import namedtuple
from matplotlib import pyplot as plt


# https://github.com/orpatashnik/StyleCLIP
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        # print(*modules)
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


class IDLoss(nn.Module):
    def __init__(self, pre_trained_model_ir_se50):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # if torch.cuda.is_available():
        #     map_location = lambda storage, loc: storage.cuda()
        # else:
        #     map_location = 'cpu'
        self.facenet.load_state_dict(torch.load(pre_trained_model_ir_se50, map_location='cpu'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        # self.facenet.cuda()
        # self.opts = pre_trained_model_ir_se50

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        # breakpoint()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count


def load_img(
        path: str,
        height: int = 256,
        width: int = 256,
        bits: int = 8,
        plot: bool = False,
        crop_mode: str = "centre-crop",
        save_gt: bool = False,
        **kwargs,
) -> torch.Tensor:
    img = cv2.imread(str(path), -1)[:, :, ::-1] / (2 ** bits - 1)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)

    if crop_mode == "resize-crop":
        # Resize such that shorter side matches corresponding target side
        smaller_side = min(height, width)
        img = resize(img.unsqueeze(0), smaller_side, align_corners=False).squeeze(0)

    img = center_crop(img.unsqueeze(0), (height, width), align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0)

    if plot:
        plt.imshow(img)
        plt.show()

    if save_gt:
        cv2.imwrite("gt.png", img.numpy()[:, :, ::-1] * 255.0)

    # H x W x 3 to NHWC
    img = rearrange(img, "h w c -> 1 c h w")
    return img


if __name__ == "__main__":
    # weight at https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing
    ir_se50_weights_path = './outputs/ckpt/model_ir_se50.pth'
    id_loss = IDLoss(ir_se50_weights_path)
    img_gen = load_img(path='./output_900_0.png')
    img_orig = load_img(path='./output_150_0.png')
    i_loss = id_loss(img_gen, img_orig)[0].detach().numpy()
    print("id_loss for different images", i_loss)

    i_loss = id_loss(img_orig, img_orig)[0].detach().numpy()
    print("id_loss for same images", i_loss)
