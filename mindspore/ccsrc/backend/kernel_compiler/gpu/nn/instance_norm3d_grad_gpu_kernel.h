/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM3D_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM3D_GRAD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class InstanceNorm3dGradGpuKernel : public GpuKernel {
 public:
  InstanceNorm3dGradGpuKernel()
      : mode_(CUDNN_BATCHNORM_SPATIAL),
        epsilon_(10e-5),
        is_null_input_(false),
        x_desc_(nullptr),
        dy_desc_(nullptr),
        dx_desc_(nullptr),
        scale_bias_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~InstanceNorm3dGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto x = GetDeviceAddress<T>(inputs, 1);
    auto scale = GetDeviceAddress<float>(inputs, 2);
    auto save_mean = GetDeviceAddress<float>(inputs, 3);
    auto save_variance = GetDeviceAddress<float>(inputs, 4);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    auto bn_scale = GetDeviceAddress<float>(outputs, 1);
    auto bn_bias = GetDeviceAddress<float>(outputs, 2);

    const float alpha_data_diff = 1;
    const float beta_data_diff = 0;
    const float alpha_param_diff = 1;
    const float beta_param_diff = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnBatchNormalizationBackward(handle_, mode_, &alpha_data_diff, &beta_data_diff, &alpha_param_diff,
                                      &beta_param_diff, x_desc_, x, dy_desc_, dy, dx_desc_, dx, scale_bias_desc_,
                                      scale, bn_scale, bn_bias, epsilon_, save_mean, save_variance),
      "Kernel Launch Failed.");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", InstanceNorm3dGradGpuKernel should be 5";
    }

    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (shape.size() != 5) {
      MS_LOG(EXCEPTION) << "tensor shape is " << shape.size() << ", InstanceNorm3dGradGpuKernel should be 5";
      return false;
    }
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "InstanceNorm3dGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "data_format");
    if (format_attr == kOpFormat_NDHWC) {
      MS_LOG(ERROR) << "NDHWC is not supported in InstanceNorm3DGradGpuKernel";
      data_format_ = kOpFormat_NDHWC;
    }

    mode_ = CUDNN_BATCHNORM_SPATIAL;
    is_training_ = GetAttr<bool>(kernel_node, "is_training");
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    auto output_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto scale_bias_shape_2d = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
    std::vector<size_t> scale_bias_shape = {scale_bias_shape_2d[0], scale_bias_shape_2d[1], 1, 1, 1};

    // convert shape(N, C, ...) to shape(1, N*C, ...) to use batchnorm.
    output_shape[1] *= output_shape[0];
    output_shape[0] = 1;
    in_shape[1] *= in_shape[0];
    in_shape[0] = 1;
    scale_bias_shape[1] *= scale_bias_shape[0];
    scale_bias_shape[0] = 1;
    Set5DDesc(in_shape, output_shape, scale_bias_shape);

    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(scale_bias_desc_),
                               "Destroy para desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&scale_bias_desc_),
                                "Create para desc failed");
  }

  void InitSizeLists() override {
    size_t input_size = 0;
    size_t para_size = 0;
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_desc_, &input_size),
                                  "Get input size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(scale_bias_desc_, &para_size),
                                  "Get input size failed");
    }

    input_size_list_.push_back(input_size);
    input_size_list_.push_back(input_size);
    input_size_list_.push_back(para_size);
    input_size_list_.push_back(para_size);
    input_size_list_.push_back(para_size);

    output_size_list_.push_back(input_size);
    output_size_list_.push_back(para_size);
    output_size_list_.push_back(para_size);
    output_size_list_.push_back(input_size);
    output_size_list_.push_back(input_size);
  }

 private:
  void Set5DDesc(const std::vector<size_t> &in_shape, const std::vector<size_t> &output_shape,
                 const std::vector<size_t> &scale_bias_shape) {
    const int nbDims = 5;
    int dimAin[5];
    int strideAin[5];
    int dimAout[5];
    int strideAout[5];
    int dimAscale_bias[5];
    int strideAscale_bias[5];
    SetDimA(in_shape, dimAin, 5, data_format_);
    SetStrideA(in_shape, strideAin, 5, data_format_);
    SetDimA(output_shape, dimAout, 5, data_format_);
    SetStrideA(output_shape, strideAout, 5, data_format_);
    SetDimA(scale_bias_shape, dimAscale_bias, 5, data_format_);
    SetStrideA(scale_bias_shape, strideAscale_bias, 5, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
      cudnnSetTensorNdDescriptor(x_desc_, cudnn_data_type_, nbDims, dimAin, strideAin),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
      cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, nbDims, dimAout, strideAout),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
      cudnnSetTensorNdDescriptor(dx_desc_, cudnn_data_type_, nbDims, dimAin, strideAin),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
      cudnnSetTensorNdDescriptor(scale_bias_desc_, CUDNN_DATA_FLOAT, nbDims, dimAscale_bias, strideAscale_bias),
      "cudnnSetTensorNdDescriptor failed");
  }
  cudnnBatchNormMode_t mode_;
  std::string data_format_;
  bool is_training_;
  double epsilon_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t scale_bias_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM3D_GRAD_GPU_KERNEL_H_
