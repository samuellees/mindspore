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
from mindspore.ops import operations as P
import numbers
from mindspore.common.initializer import One


class NetConv3dTranspose(nn.Cell):
    def __init__(self):
        super(NetConv3dTranspose, self).__init__()
        in_channel = 3
        out_channel = 2
        kernel_size = 2
        self.conv_trans = nn.Conv3dTranspose(in_channel, out_channel,
                                        kernel_size,
                                        pad_mode="valid",
                                        padding=0,
                                        stride=1,
                                        dilation=1,
                                        group=1,
                                        weight_init=One())

    def construct(self, x):
        return self.conv_trans(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_transpose():
    x = Tensor(np.arange(1 * 3 * 2 * 2 * 2).reshape(1, 3, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[[ 24.,  51.,  27.],
                        [ 54., 114.,  60.],
                        [ 30.,  63.,  33.]],

                        [[ 60., 126.,  66.],
                        [132., 276., 144.],
                        [ 72., 150.,  78.]],

                        [[ 36.,  75.,  39.],
                        [ 78., 162.,  84.],
                        [ 42.,  87.,  45.]]],


                        [[[ 24.,  51.,  27.],
                        [ 54., 114.,  60.],
                        [ 30.,  63.,  33.]],

                        [[ 60., 126.,  66.],
                        [132., 276., 144.],
                        [ 72., 150.,  78.]],

                        [[ 36.,  75.,  39.],
                        [ 78., 162.,  84.],
                        [ 42.,  87.,  45.]]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", max_device_memory="0.2GB")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x)
    assert (output.asnumpy() == expect).all()