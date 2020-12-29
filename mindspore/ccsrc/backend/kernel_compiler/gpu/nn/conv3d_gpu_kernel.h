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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3DGPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3DGPUKERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class Conv3dGpuFwdKernel : public GpuKernel {
 public:
  Conv3dGpuFwdKernel()
      : cudnn_handle_(nullptr),
        input_desc_(nullptr),
        output_desc_(nullptr),
        filter_desc_(nullptr),
        conv_desc_(nullptr),
        padded_desc_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_height_(0),
        old_width_(0),
        old_depth_(0),
        pad_depth_(0),
        pad_height_(0),
        pad_width_(0),
        pad_head_(0),
        pad_tail_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        group_(1),
        is_null_input_(false),
        input_size_(0),
        filter_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(false) {}
  ~Conv3dGpuFwdKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *filter_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
      workspace_addr = GetDeviceAddress<T>(workspace, 0);
    }

    const float alpha = 1;
    const float beta = 0;
    if (use_pad_) {
      T *padded_addr = GetDeviceAddress<T>(workspace, 1);
      if (data_format_ == kOpFormat_NDHWC) {
        // CalPadNDHWC unimplemented.
        MS_LOG(ERROR) << "CalPadNDHWC unimplemented.";
      } else {
        // CalPad(5d) unimplemented.
        MS_LOG(ERROR) << "CalPad(5d) unimplemented.";
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    }
    return true;
  }
  
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }

    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "data_format");

    if (format_attr == kOpFormat_NDHWC) {
      data_format_ = kOpFormat_NDHWC;
    }
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(in_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "Conv3dGpuFwdKernel input is null.";
      InitSizeLists();
      return true;
    }
    SetNCDHW(in_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format_);
    if (data_format_ == kOpFormat_NDHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
    }
    Set5DDesc(in_shape, filter_shape, output_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "groups"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pads");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    pad_depth_ = pad_list[0];
    pad_height_ = pad_list[2];
    pad_width_ = pad_list[4];
    use_pad_ = !((pad_depth_ == pad_list[1]) && (pad_height_ == pad_list[3]) && (pad_width_ == pad_list[5]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    int padA[3];
    int strideA[3] = {stride_[2], stride_[3], stride_[4]};
    int dilaA[3] = {dilation_[2], dilation_[3], dilation_[4]};
    if (use_pad_) {
      pad_depth_ = pad_list[0] + pad_list[1];
      pad_height_ = pad_list[2] + pad_list[3];
      pad_width_ = pad_list[4] + pad_list[5];
      pad_head_ = pad_list[0];
      pad_top_ = pad_list[2];
      pad_left_ = pad_list[4];
      int dimA[5];
      int strideApadded[5];
      if (data_format_ == kOpFormat_NCDHW) {
        auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_depth_ + pad_depth_),
                             IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_)};
        SetDimA(padded_shape, dimA, 5, data_format_);
        SetStrideA(padded_shape, strideApadded, 5, data_format_);
      } else if (data_format_ == kOpFormat_NDHWC) {
        auto padded_shape = {IntToSize(n_), IntToSize(old_depth_ + pad_depth_), IntToSize(old_height_ + pad_height_),
                            IntToSize(old_width_ + pad_width_), IntToSize(c_)};
        SetDimA(padded_shape, dimA, 5, data_format_);
        SetStrideA(padded_shape, strideApadded, 5, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, 5, dimA, strideApadded),
                                  "cudnnSetTensor5dDescriptor failed");
      padA[0] = 0;
      padA[1] = 0;
      padA[2] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 3, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor failed");
      input_descriptor_real = padded_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_depth_;
      padA[1] = pad_height_;
      padA[2] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 3, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution3dDescriptor failed");
      input_descriptor_real = input_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(input_descriptor_real);
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(filter_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&filter_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(filter_size_);
    output_size_list_.push_back(output_size_);
    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, output_desc_,
                                                conv_algorithm_, &workspace_size_),
        "cudnnGetConvolutionForwardWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                  conv_algorithm_, &workspace_size_),
          "cudnnGetConvolutionForwardWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);

    return;
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but conv3d needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but conv3d needs 1 output.";
      return false;
    }
    return true;
  }

  void Set5DDesc(const std::vector<size_t> &in_shape, const std::vector<size_t> &filter_shape,
                 const std::vector<size_t> &output_shape) {
    const int nbDims = 5;
    int dimA[5];
    int strideAin[5];
    int dimAout[5];
    int strideAout[5];
    SetDimA(in_shape, dimA, 5, data_format_);
    SetStrideA(in_shape, strideAin, 5, data_format_);
    SetDimA(output_shape, dimAout, 5, data_format_);
    SetStrideA(output_shape, strideAout, 5, data_format_);

    int filterDimA[5];
    // ODHWI for NDHWC; OIDHW for NCDHW
    SetDimA(filter_shape, filterDimA, 5, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, nbDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, nbDims, dimAout, strideAout),
                                "cudnnSetTensor5dDescriptor failed");
  }

  void SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
    if (group_ > 1 || CUDNN_MAJOR < 7) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetConvolutionForwardAlgorithm(
                                    cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm_),
                                  "cudnnGetConvolutionForwardAlgorithm failed");
    } else {
      constexpr int requested_algo_count = 1;
      int returned_algo_count;
      cudnnConvolutionFwdAlgoPerf_t perf_results;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_,
                                               output_desc_, requested_algo_count, &returned_algo_count, &perf_results),
        "cudnnGetConvolutionForwardAlgorithm_v7 failed");
      conv_algorithm_ = perf_results.algo;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != 5) {
      MS_LOG(EXCEPTION) << "Conv3d's' stride must be 5d!";
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "Conv3d stride only support 1 in N axis and C axis!";
    }
    if (dilation_.size() != 5) {
      MS_LOG(EXCEPTION) << "Conv3d's dilation must be 5d!";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "Conv3d dilation only support 1 in N axis and C axis!";
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_algorithm_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t padded_desc_;
  std::string pad_mode_;
  std::string data_format_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_height_;
  int old_width_;
  int old_depth_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_head_;
  int pad_tail_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  size_t input_size_;
  size_t filter_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3DGPUKERNEL_H_
