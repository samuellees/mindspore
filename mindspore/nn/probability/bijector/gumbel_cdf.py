# Copyright 2020 Huawei Technologies Co., Ltd
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
"""GumbelCDF Bijector"""
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator
from mindspore.ops import operations as P
from ..distribution._utils.utils import check_greater_zero, set_param_type
from ..distribution._utils.custom_ops import exp_generic, log_generic
from .bijector import Bijector


class GumbelCDF(Bijector):
    r"""
    GumbelCDF Bijector.
    This Bijector performs the operation:

    .. math::
        Y = \exp(-\exp(\frac{-(X - loc)}{scale}))

    Note:
        For `reverse` and `reverse_log_jacobian`, input should be in range of (0, 1).

    Args:
        loc (int, float, list, numpy.ndarray, Tensor): The location. Default: 0..
        scale (int, float, list, numpy.ndarray, Tensor): The scale. Default: 1.0.
        dtype (mindspore.dtype): Type of the distribution which the bijector operates on. Default: float32.
        name (str): The name of the Bijector. Default: 'Gumbel_CDF'.

    Examples:
        >>> # To initialize a GumbelCDF bijector of loc 0.0, and scale 1.0.
        >>> import mindspore.nn.probability.bijector as msb
        >>> gum = msb.GumbelCDF(0.0, 1.0)
        >>>
        >>> # To use GumbelCDF bijector in a network.
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.gum = msb.GumbelCDF(0.0, 1.0)
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other functions
        >>>         # by replacing 'forward' by the name of the function.
        >>>         ans1 = self.gum.forward(value)
        >>>         ans2 = self.gum.inverse(value)
        >>>         ans3 = self.gum.forward_log_jacobian(value)
        >>>         ans4 = self.gum.inverse_log_jacobian(value)
    """

    def __init__(self,
                 loc=0.0,
                 scale=1.0,
                 dtype=mstype.float32,
                 name='GumbelCDF'):
        """
        Constructor of GumbelCDF Bijector.
        """
        param = dict(locals())
        valid_dtype = mstype.float_type + mstype.int_type + mstype.uint_type
        Validator.check_type(type(self).__name__, dtype, valid_dtype)
        parameter_type = set_param_type({'loc': loc, "scale": scale}, dtype)
        super(GumbelCDF, self).__init__(name=name, dtype=dtype, param=param)

        self._parameter_type = parameter_type
        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')
        check_greater_zero(self._scale, "scale")
        self._event_shape = self._calc_event_shape()

        self.cast = P.Cast()
        self.exp = exp_generic
        self.log = log_generic


    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def parameter_type(self):
        return self._parameter_type

    def extend_repr(self):
        return f'loc = {self.loc}, scale = {self.scale}'

    def shape_mapping(self, shape):
        return shape

    def _forward(self, x):
        x = self._check_value(x, 'value')
        x = self.cast(x, self.parameter_type)
        z = (x - self.loc) / self.scale
        return self.exp(-self.exp(-z))

    def _inverse(self, y):
        y = self._check_value(y, 'value')
        y = self.cast(y, self.parameter_type)
        return self.loc - self.scale * self.log(-self.log(y))

    def _forward_log_jacobian(self, x):
        x = self._check_value(x, 'value')
        x = self.cast(x, self.parameter_type)
        z = (x - self.loc) / self.scale
        return -z - self.exp(-z) - self.log(self.scale)

    def _inverse_log_jacobian(self, y):
        y = self._check_value(y, 'value')
        y = self.cast(y, self.parameter_type)
        return self.log(self.scale / (-1. * y * self.log(y)))