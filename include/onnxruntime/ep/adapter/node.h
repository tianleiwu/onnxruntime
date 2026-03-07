// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>
#include <vector>

namespace onnxruntime {
namespace ep {
namespace adapter {

struct NodeArg {
  std::string Name() const noexcept {
    return name_;
  }

  const ONNX_NAMESPACE::TensorShapeProto* Shape() const noexcept {
    return has_shape_ ? &shape_proto_ : nullptr;
  }

  bool Exists() const noexcept {
    return exists_;
  }

  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept {
    return has_type_proto_ ? &type_proto_ : nullptr;
  }

 private:
  friend struct Node;

  std::string name_;
  bool exists_{false};
  bool has_shape_{false};
  bool has_type_proto_{false};
  ONNX_NAMESPACE::TensorShapeProto shape_proto_;
  ONNX_NAMESPACE::TypeProto type_proto_;
};

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::Node`.
/// </summary>
struct Node {
  struct Cache {
    explicit Cache(const OrtKernelInfo* kernel_info) : kernel_info_{kernel_info} {
      const size_t input_count = kernel_info_.GetInputCount();
      input_defs_.reserve(input_count);
      for (size_t i = 0; i < input_count; ++i) {
        input_defs_.push_back(CreateNodeArg(i, true));
      }

      const size_t output_count = kernel_info_.GetOutputCount();
      output_defs_.reserve(output_count);
      for (size_t i = 0; i < output_count; ++i) {
        output_defs_.push_back(CreateNodeArg(i, false));
      }
    }

    NodeArg CreateNodeArg(size_t index, bool is_input) const noexcept {
      NodeArg node_arg;

      try {
        node_arg.name_ = is_input ? kernel_info_.GetInputName(index) : kernel_info_.GetOutputName(index);
        node_arg.exists_ = !node_arg.name_.empty();
        if (!node_arg.exists_) {
          return node_arg;
        }

        auto type_info = is_input ? kernel_info_.GetInputTypeInfo(index) : kernel_info_.GetOutputTypeInfo(index);
        node_arg.has_type_proto_ = true;

        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
          auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
          auto* tensor_type = node_arg.type_proto_.mutable_tensor_type();
          tensor_type->set_elem_type(static_cast<int>(tensor_info.GetElementType()));

          if (tensor_info.HasShape()) {
            node_arg.has_shape_ = true;
            auto shape = tensor_info.GetShape();
            for (const auto dim_value : shape) {
              node_arg.shape_proto_.add_dim()->set_dim_value(dim_value);
            }
            *tensor_type->mutable_shape() = node_arg.shape_proto_;
          }
        }
      } catch (...) {
        // Keep a best-effort empty NodeArg so plugin builds can query optional presence
        // without depending on the full framework Node/NodeArg implementation.
      }

      return node_arg;
    }

    Ort::ConstKernelInfo kernel_info_;
    std::vector<NodeArg> input_defs_;
    std::vector<NodeArg> output_defs_;
  };

  explicit Node(const OrtKernelInfo* kernel_info) : cache_{std::make_shared<Cache>(kernel_info)} {}

  struct ValueInfoList {
    size_t size() const noexcept {
      return defs_ != nullptr ? defs_->size() : 0;
    }

    const NodeArg* operator[](size_t index) const noexcept {
      return defs_ != nullptr && index < defs_->size() ? &(*defs_)[index] : nullptr;
    }

    const std::vector<NodeArg>* defs_;
  };

  struct ArgCountList {
    int front() const noexcept {
      return count_;
    }

    int count_;
  };

  /** Gets the Node's name. */
  std::string Name() const noexcept {
    return cache_->kernel_info_.GetNodeName();
  }

  /** Gets the Node's operator type. */
  std::string OpType() const noexcept {
    return cache_->kernel_info_.GetOperatorType();
  }

  /** Gets the Node's domain. */
  std::string Domain() const {
    return cache_->kernel_info_.GetOperatorDomain();
  }

  /** Gets the since version of the operator. */
  int SinceVersion() const noexcept {
    return cache_->kernel_info_.GetOperatorSinceVersion();
  }

  ValueInfoList InputDefs() const noexcept {
    return ValueInfoList{&cache_->input_defs_};
  }

  ValueInfoList OutputDefs() const noexcept {
    return ValueInfoList{&cache_->output_defs_};
  }

  ArgCountList InputArgCount() const noexcept {
    return ArgCountList{static_cast<int>(cache_->kernel_info_.GetInputCount())};
  }

 private:
  const std::shared_ptr<Cache> cache_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
