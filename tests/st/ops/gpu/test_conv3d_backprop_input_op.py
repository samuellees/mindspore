# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P

context.set_context(device_target='GPU')


class Conv3dInput(nn.Cell):
    def __init__(self):
        super(Conv3dInput, self).__init__()
        out_channel = 1
        kernel_size = 2
        self.conv_input = P.Conv3DBackpropInput(out_channel,
                                                kernel_size,
                                                pad_mode="valid",
                                                pad=0,
                                                mode=1,
                                                stride=1,
                                                dilation=1,
                                                group=1)

        self.get_shape = P.Shape()

    @ms_function
    def construct(self, out, w, x):
        return self.conv_input(w, out, self.get_shape(x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_backprop_input():
    w = Tensor(np.array(
        [[[[[-0.2361,  0.3280],
           [-0.1224, -0.2836]],

          [[-0.0180,  0.3383],
           [ 0.0081,  0.1408]]]]]).astype(np.float32))
    x = Tensor(np.array(
        [[[[[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]],

          [[ 9., 10., 11.],
           [12., 13., 14.],
           [15., 16., 17.]],
           
          [[18., 19., 20.],
           [21., 22., 23.],
           [24., 25., 26.]]]]]).astype(np.float32))
    grad_out = Tensor(np.array(
        [[[[[0.5414, 0.1442],
           [0.0283, 0.2984]],

          [[0.7242, 0.2607],
           [0.9309, 0.9152]]]]]).astype(np.float32))
    conv3d_input = Conv3dInput()
    grad_input = conv3d_input(grad_out, w, x)
    expect = np.array(
        [[[[[-0.1278,  0.1435,  0.0473],
           [-0.0730, -0.2323,  0.0570],
           [-0.0035, -0.0446, -0.0846]],

          [[-0.1807,  0.3565,  0.1343],
           [-0.3045, -0.0664,  0.3475],
           [-0.1137, -0.3696, -0.2175]],

          [[-0.0131,  0.2403,  0.0882],
           [-0.0109,  0.4025,  0.3463],
           [ 0.0076,  0.1385,  0.1289]]]]]).astype(np.float32)
    assert (abs(grad_input.asnumpy() - expect) < np.ones(shape=[1, 1, 3, 3, 3]) * 1.0e-4).all()