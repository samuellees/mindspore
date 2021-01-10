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
from mindspore.nn import Bn4Xception
from mindspore.nn import Cell
from mindspore.ops import composite as C


class Batchnorm_Net(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init, use_batch_statistics=None):
        super(Batchnorm_Net, self).__init__()
        self.bn = Bn4Xception(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
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
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.6059, 0.3118, 0.3118, 1.2294],
                                [-0.1471, 0.7706, 1.6882, 2.6059],
                                [0.3118, 1.6882, 2.1471, 2.1471],
                                [0.7706, 0.3118, 2.6059, -0.1471]],

                               [[0.9119, 1.8518, 1.3819, -0.0281],
                                [-0.0281, 0.9119, 1.3819, 1.8518],
                                [2.7918, 0.4419, -0.4981, 0.9119],
                                [1.8518, 0.9119, 2.3218, -0.9680]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32)
    moving_mean = np.ones(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias),
                           Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_backward():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    grad = np.array([[
        [[1, 2, 7, 1], [4, 2, 1, 3], [1, 6, 5, 2], [2, 4, 3, 2]],
        [[9, 4, 3, 5], [1, 3, 7, 6], [5, 7, 9, 9], [1, 4, 6, 8]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.69126546, -0.32903028, 1.9651246, -0.88445705],
                                [0.6369296, -0.37732816, -0.93275493, -0.11168876],
                                [-0.7878612, 1.3614, 0.8542711, -0.52222186],
                                [-0.37732816, 0.5886317, -0.11168876, -0.28073236]],

                               [[1.6447213, -0.38968924, -1.0174079, -0.55067265],
                                [-2.4305856, -1.1751484, 0.86250514, 0.5502673],
                                [0.39576983, 0.5470243, 1.1715001, 1.6447213],
                                [-1.7996241, -0.7051701, 0.7080077, 0.5437813]]]]).astype(np.float32)

    weight = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    moving_mean = Tensor(np.ones(2).astype(np.float32))
    moving_var_init = Tensor(np.ones(2).astype(np.float32))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, weight, bias, moving_mean, moving_var_init)
    bn_net.set_train()
    bn_grad = Grad(bn_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_stats_false_forward():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)

    expect_output = np.array([[[[3.707105, 5.121315, 5.121315, 6.535525],
                                [4.41421, 5.8284197, 7.24263, 8.656839],
                                [5.121315, 7.24263, 7.9497347, 7.9497347],
                                [5.8284197, 5.121315, 8.656839, 4.41421]],

                               [[6.535525, 7.9497347, 7.24263, 5.121315],
                                [5.121315, 6.535525, 7.24263, 7.9497347],
                                [9.363945, 5.8284197, 4.41421, 6.535525],
                                [7.9497347, 6.535525, 8.656839, 3.707105]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32) * 3
    moving_mean = np.zeros(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32) * 2
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4
    use_batch_statistics = False

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                           Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                           Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

# FIXME: This case can not pass
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_infer_backward():
    expect_output = np.array([[[[-0.3224156, -0.3840524], [1.1337637, -1.0998858]],
                               [[-0.1724273, -0.877854], [0.0422135, 0.5828123]],
                               [[-1.1006137, 1.1447179], [0.9015862, 0.5024918]]]]).astype(np.float32)
    np.random.seed(1)
    x_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    input_grad_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    ms_input = Tensor(x_np)
    weight = Tensor(np.ones(3).astype(np.float32))
    bias = Tensor(np.zeros(3).astype(np.float32))
    moving_mean = Tensor(np.zeros(3).astype(np.float32))
    moving_var_init = Tensor(np.ones(3).astype(np.float32))
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms_net = Batchnorm_Net(3, weight, bias, moving_mean, moving_var_init)
    ms_net.set_train(False)
    ms_grad = Grad(ms_net)
    ms_out_grad_np = ms_grad(ms_input, Tensor(input_grad_np))
    print(ms_out_grad_np)
    assert np.allclose(ms_out_grad_np[0].asnumpy(), expect_output)

test_train_forward()
test_train_backward()
test_train_stats_false_forward()
test_infer_backward()