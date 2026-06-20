import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class KeypointHead(nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,  # when not given, use input_feature_dim//2
    ):
        """
        Args:
            input_feature_dim: Dimension of input feature
            output_dim: Dimension of output regression
            hidden_dims: [h1, h2] List of 2 hidden dimensions for intermediate. default is [input_feature_dim//2] * 2
        """

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [input_feature_dim // 2] * 2
        else:
            assert isinstance(hidden_dims, List) and len(hidden_dims) == 2

        self.conv1 = nn.Conv2d(input_feature_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        # interpolate is dependent on target output size
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU(False)
        self.conv3 = nn.Conv2d(hidden_dims[1], output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if hasattr(x, 'features_upsampled_8x'):
            x = x.features_upsampled_8x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)

        return x

class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, in_w).float(), torch.linspace(-1, 1, in_h).float(), indexing="ij")

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        keypoint_y = (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        keypoints = torch.stack([keypoint_x, keypoint_y], dim=-1)

        return keypoints


from torch.autograd import Variable
class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=8, learned_beta=True, initial_beta=25.0):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        if learned_beta:
            self.beta = torch.nn.Parameter(torch.ones(n_keypoints) * initial_beta)
        else:
            self.beta = (torch.ones(n_keypoints) * initial_beta).cuda()

    def forward(self, heatmaps, size_mult=1.0):
        epsilon = 1e-8
        bch, ch, n_row, n_col = heatmaps.size()
        n_kpts = ch

        beta = self.beta

        # input has the shape (#bch, n_kpts+1, img_sz[0], img_sz[1])
        # +1 is for the Zrel
        heatmaps2d = heatmaps[:, :n_kpts, :, :]
        heatmaps2d = self.avgpool(heatmaps2d)

        # heatmaps2d has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d.contiguous().view(bch, n_kpts, -1)

        # getting the max value of the maps across each 2D matrix
        map_max = torch.max(heatmaps2d, dim=2, keepdim=True)[0]

        # reducing the max from each map
        # this will make the max value zero and all other values
        # will be negative.
        # max_reduced_maps has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d - map_max

        beta_ = beta.view(1, n_kpts, 1).repeat(1, 1, n_row * n_col)
        # due to applying the beta value, the non-max values will be further
        # pushed towards zero after applying the exp function
        exp_maps = torch.exp(beta_ * heatmaps2d)
        # normalizing the exp_maps by diving it to the sum of elements
        # exp_maps_sum has the shape (#bch, n_kpts, 1)
        exp_maps_sum = torch.sum(exp_maps, dim=2, keepdim=True)
        exp_maps_sum = exp_maps_sum.view(bch, n_kpts, 1, 1)
        normalized_maps = exp_maps.view(bch, n_kpts, n_row, n_col) / (
            exp_maps_sum + epsilon
        )

        col_vals = torch.arange(0, n_col) * size_mult
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col).cuda()
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        row_vals = torch.arange(0, n_row).view(n_row, -1) * size_mult
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col).cuda()
        # row_mat gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_idx = Variable(col_idx, requires_grad=False)
        weighted_x = normalized_maps * col_idx.float()
        weighted_x = weighted_x.view(bch, n_kpts, -1)
        x_vals = torch.sum(weighted_x, dim=2)

        row_idx = Variable(row_idx, requires_grad=False)
        weighted_y = normalized_maps * row_idx.float()
        weighted_y = weighted_y.view(bch, n_kpts, -1)
        y_vals = torch.sum(weighted_y, dim=2)

        out = torch.stack((x_vals, y_vals), dim=2)

        return out


if __name__ == "__main__":
    # [(630,476), (616,476), (630, 462), (616,462)]
    # H, W = 462, 616
    H, W = int(462/14*8), int(616/14*8)

    image_feature = torch.randn(1, 256, H, W).cuda()

    # keypoint_head = KeypointHead(input_feature_dim=256, output_dim=8).cuda()
    # belief_map = keypoint_head(image_feature)
    # print(belief_map.shape)

    # spatial_softmax = SpatialSoftmax(in_c=256, in_h=H, in_w=W, num_kp=8).cuda()
    # keypoints = spatial_softmax(image_feature)
    # print(keypoints.shape)

    spatial_conv = nn.Conv2d(256, 8, kernel_size=1).cuda()
    soft_argmax_pavlo = SoftArgmaxPavlo(n_keypoints=8).cuda()
    keypoints = soft_argmax_pavlo(spatial_conv(image_feature))
    print(keypoints.shape)
