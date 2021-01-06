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
from mindspore.common.tensor import Tensor
from mindspore.nn import BatchNorm3d
from mindspore.nn import Cell
from mindspore.ops import composite as C


class Batchnorm_Net(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init, use_batch_statistics=None):
        super(Batchnorm_Net, self).__init__()
        self.bn = BatchNorm3d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init,
                              use_batch_statistics=use_batch_statistics)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_forward():
    x = np.array([[[[[5.3401e-01, 7.6590e-01, 7.4739e-01],
           [4.8129e-01, 6.2816e-01, 2.6645e-01],
           [5.6883e-01, 6.9016e-01, 6.6287e-01]],

          [[1.0535e-01, 2.7238e-01, 9.8831e-01],
           [9.1905e-01, 9.4756e-01, 3.8076e-01],
           [2.8471e-01, 6.2435e-01, 2.6670e-01]]],


         [[[2.2445e-01, 7.0691e-05, 8.3474e-01],
           [9.7002e-01, 3.9454e-01, 6.5432e-01],
           [1.3920e-01, 5.9494e-01, 8.0298e-01]],

          [[6.7785e-02, 7.6266e-01, 4.8218e-01],
           [1.0485e-01, 6.9664e-01, 6.0554e-01],
           [6.0264e-02, 1.6288e-01, 7.0866e-02]]]]]).astype(np.float32)
    expect_output = np.array([[[[[-0.11460,  0.80167,  0.7285],
                            [-0.3229,  0.2574, -1.1718],
                            [ 0.0230,  0.5024,  0.3946]],

                            [[-1.8084, -1.1484,  1.6805],
                            [ 1.4068,  1.5195, -0.7201],
                            [-1.0997,  0.2424, -1.1708]]],


                            [[[-0.6351, -1.3499,  1.3090],
                            [ 1.7399, -0.0933,  0.7343],
                            [-0.9067,  0.5451,  1.2078]],

                            [[-1.1342,  1.0794,  0.1859],
                            [-1.0161,  0.8691,  0.5788],
                            [-1.1582, -0.8313, -1.1244]]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    moving_mean = np.zeros(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32)
    # PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-03, atol=1e-4)
    # GRAPH_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-03, atol=1e-4)
    # GRAPH_MODE inference
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), x, rtol=1e-03, atol=1e-4)
    # PYNATIVE_MODE inference
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), x, rtol=1e-03, atol=1e-4)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_backward():
    x = np.array([[[[[5.3401e-01, 7.6590e-01, 7.4739e-01],
                    [4.8129e-01, 6.2816e-01, 2.6645e-01],
                    [5.6883e-01, 6.9016e-01, 6.6287e-01]],

                    [[1.0535e-01, 2.7238e-01, 9.8831e-01],
                    [9.1905e-01, 9.4756e-01, 3.8076e-01],
                    [2.8471e-01, 6.2435e-01, 2.6670e-01]]],


                    [[[2.2445e-01, 7.0691e-05, 8.3474e-01],
                    [9.7002e-01, 3.9454e-01, 6.5432e-01],
                    [1.3920e-01, 5.9494e-01, 8.0298e-01]],

                    [[6.7785e-02, 7.6266e-01, 4.8218e-01],
                    [1.0485e-01, 6.9664e-01, 6.0554e-01],
                    [6.0264e-02, 1.6288e-01, 7.0866e-02]]]]]).astype(np.float32)
    grad = np.array([[[[[0.5121, 0.7513, 0.4234],
                    [0.2849, 0.5122, 0.8911],
                    [0.2382, 0.0440, 0.4348]],

                    [[0.7826, 0.1405, 0.5640],
                    [0.2250, 0.6701, 0.5360],
                    [0.5227, 0.8489, 0.8967]]],


                    [[[0.0749, 0.1094, 0.2155],
                    [0.0576, 0.0279, 0.4442],
                    [0.0315, 0.6138, 0.3153]],

                    [[0.0647, 0.6600, 0.9817],
                    [0.2720, 0.6074, 0.5651],
                    [0.2664, 0.4275, 0.3583]]]]]).astype(np.float32)
    expect_output = np.array([[[[[-0.0398,  1.1169, -0.1954],
                            [-0.9856,  0.0465,  1.2136],
                            [-1.0903, -1.7468, -0.2276]],

                            [[ 0.6376, -1.7468,  0.5797],
                            [-0.8228,  0.9620, -0.0852],
                            [-0.2255,  1.3733,  1.2361]]],


                            [[[-0.6677, -0.3642, -0.7465],
                            [-1.3662, -0.9641,  0.1377],
                            [-0.7323,  0.7293, -0.4012]],

                            [[-0.5650,  0.7318,  1.9985],
                            [ 0.0634,  0.6211,  0.5650],
                            [ 0.0841,  0.5087,  0.3676]]]]]).astype(np.float32)

    weight = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.zeros(2).astype(np.float32))
    moving_mean = Tensor(np.zeros(2).astype(np.float32))
    moving_var_init = Tensor(np.ones(2).astype(np.float32))
    # GRAPH_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, weight, bias, moving_mean, moving_var_init)
    bn_net.set_train()
    bn_grad = Grad(bn_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    np.testing.assert_allclose(output[0].asnumpy(), expect_output, rtol=1e-03, atol=1e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_stats_false_forward():
    x = np.array([[[[[5.3401e-01, 7.6590e-01, 7.4739e-01],
                    [4.8129e-01, 6.2816e-01, 2.6645e-01],
                    [5.6883e-01, 6.9016e-01, 6.6287e-01]],

                    [[1.0535e-01, 2.7238e-01, 9.8831e-01],
                    [9.1905e-01, 9.4756e-01, 3.8076e-01],
                    [2.8471e-01, 6.2435e-01, 2.6670e-01]]],


                    [[[2.2445e-01, 7.0691e-05, 8.3474e-01],
                    [9.7002e-01, 3.9454e-01, 6.5432e-01],
                    [1.3920e-01, 5.9494e-01, 8.0298e-01]],

                    [[6.7785e-02, 7.6266e-01, 4.8218e-01],
                    [1.0485e-01, 6.9664e-01, 6.0554e-01],
                    [6.0264e-02, 1.6288e-01, 7.0866e-02]]]]]).astype(np.float32)

    expect_output = np.array([[[[[-0.1146,  0.8017,  0.7285],
                                [-0.3229,  0.2574, -1.1718],
                                [ 0.0230,  0.5024,  0.3946]],

                                [[-1.8084, -1.1484,  1.6805],
                                [ 1.4068,  1.5195, -0.7201],
                                [-1.0997,  0.2424, -1.1708]]],


                                [[[-0.6351, -1.3499,  1.3090],
                                [ 1.7399, -0.0933,  0.7343],
                                [-0.9067,  0.5451,  1.2078]],

                                [[-1.1342,  1.0794,  0.1859],
                                [-1.0161,  0.8691,  0.5788],
                                [-1.1582, -0.8313, -1.1244]]]]]).astype(np.float32)

    # weight = np.ones(2).astype(np.float32)
    # bias = np.ones(2).astype(np.float32) * 3
    # moving_mean = np.zeros(2).astype(np.float32)
    # moving_var_init = np.ones(2).astype(np.float32) * 2
    
    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    moving_mean = np.zeros(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32)
    use_batch_statistics = False
    # PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                           Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), x, rtol=1e-03, atol=1e-4)
    # GRAPH_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                           Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), x, rtol=1e-03, atol=1e-4)
