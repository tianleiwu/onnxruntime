// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep.h"
#include "cuda_ep_factory.h"

namespace onnxruntime {
namespace cuda_plugin {

CudaEp::CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger)
    : OrtEp{},
      factory_(factory),
      name_(factory.GetEpName()),
      config_(config),
      logger_(logger) {
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers for kernel-registry-based EP
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;

  // Not a compile-based EP
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  const OrtApi& ort_api = factory_.GetOrtApi();
  Ort::Status log_status(ort_api.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                                   "CUDA Plugin EP created",
                                                   ORT_FILE, __LINE__, __FUNCTION__));
}

CudaEp::~CudaEp() = default;

/*static*/
const char* ORT_API_CALL CudaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const CudaEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* ort_graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  Ort::ConstGraph graph{ort_graph};
  std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;
  }

  // For each node, check if we have a registered kernel
  for (const auto& node : all_nodes) {
    std::string op_type_str = node.GetOperatorType();
    std::string domain_str = node.GetDomain();
    const char* op_type = op_type_str.c_str();
    const char* domain = domain_str.c_str();

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
        graph_support_info, node, &kernel_def));

    size_t input_count = node.GetInputs().size();
    printf("GetCapability: Node op_type=%s, domain=%s, input_count=%zu, kernel_def=%p\n", op_type, domain, input_count, (void*)kernel_def);
    fflush(stdout);

    if (kernel_def != nullptr) {
      RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_AddSingleNode(
          graph_support_info, node));
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetKernelRegistryImpl(
    OrtEp* this_ptr,
    const OrtKernelRegistry** kernel_registry) noexcept {
  auto* ep = static_cast<CudaEp*>(this_ptr);
  *kernel_registry = nullptr;

  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetPreferredDataLayoutImpl(
    OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  *preferred_data_layout = ep->config_.prefer_nhwc ? OrtEpDataLayout_NHWC : OrtEpDataLayout_NCHW;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunStartImpl(
    OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/) noexcept {
  // Stub: will later manage CUDA Graph capture state
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunEndImpl(
    OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/, bool /*sync_stream*/) noexcept {
  // Stub: will later manage CUDA Graph replay state
  return nullptr;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
