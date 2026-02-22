// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_stream_plugin.h"
#include "cuda_ep_factory.h"
#include <mutex>

namespace onnxruntime {
namespace cuda_plugin {

namespace {
static std::unordered_map<cudaStream_t, CudaSyncStream*> g_stream_map;
static std::mutex g_stream_map_mutex;
}  // namespace

// ---------------------------------------------------------------------------
// CudaSyncStream
// ---------------------------------------------------------------------------

CudaSyncStream::CudaSyncStream(CudaEpFactory& factory, int device_id,
                               const OrtEp* /*ep*/)
    : OrtSyncStreamImpl{},
      factory_(factory),
      device_id_(device_id) {
  ort_version_supported = ORT_API_VERSION;
  GetHandle = GetHandleImpl;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  Release = ReleaseImpl;
}

CudaSyncStream::~CudaSyncStream() {
  CleanupDeferredCPUBuffers();

  if (cuda_stream_) UnregisterStream(cuda_stream_);

  if (cublas_handle_) cublasDestroy(cublas_handle_);
  if (cudnn_handle_) cudnnDestroy(cudnn_handle_);
  if (cublas_lt_handle_) cublasLtDestroy(cublas_lt_handle_);
  if (cuda_stream_) cudaStreamDestroy(cuda_stream_);
}

OrtStatus* CudaSyncStream::InitHandles() {
  cudaSetDevice(device_id_);

  CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
  RegisterStream(cuda_stream_, this);

  CUBLAS_RETURN_IF_ERROR(cublasCreate(&cublas_handle_));
  CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas_handle_, cuda_stream_));

  CUDNN_RETURN_IF_ERROR(cudnnCreate(&cudnn_handle_));
  CUDNN_RETURN_IF_ERROR(cudnnSetStream(cudnn_handle_, cuda_stream_));

  CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&cublas_lt_handle_));

  return nullptr;
}

void CudaSyncStream::EnqueueDeferredCPUBuffer(void* cpu_buffer) {
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

void CudaSyncStream::CleanupDeferredCPUBuffers() {
  for (void* buf : deferred_cpu_buffers_) {
    cudaFreeHost(buf);
  }
  deferred_cpu_buffers_.clear();
}

/*static*/ void* ORT_API_CALL CudaSyncStream::GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  printf("CudaSyncStream::GetHandleImpl called with this_ptr=%p\n", (void*)this_ptr);
  fflush(stdout);
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  printf("CudaSyncStream::GetHandleImpl returning stream->cuda_stream_=%p\n", (void*)stream->cuda_stream_);
  fflush(stdout);
  return stream->cuda_stream_;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::CreateNotificationImpl(
    OrtSyncStreamImpl* this_ptr, OrtSyncNotificationImpl** notification) noexcept {
  printf("CudaSyncStream::CreateNotificationImpl called\n");
  fflush(stdout);
  EXCEPTION_TO_STATUS_BEGIN
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  auto notif = std::make_unique<CudaSyncNotification>(*stream);
  *notification = notif.release();
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  printf("CudaSyncStream::FlushImpl called\n");
  fflush(stdout);
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream->cuda_stream_));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  // Synchronize before releasing deferred CPU buffers to ensure
  // all async copies using those buffers have completed.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream->cuda_stream_));
  stream->CleanupDeferredCPUBuffers();
  return nullptr;
}

/*static*/ void ORT_API_CALL CudaSyncStream::ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<CudaSyncStream*>(this_ptr);
}

/*static*/ CudaSyncStream* CudaSyncStream::FromCudaStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_stream_map_mutex);
  auto it = g_stream_map.find(stream);
  if (it != g_stream_map.end()) {
    return it->second;
  }
  printf("CudaSyncStream::FromCudaStream: Stream %p NOT FOUND in map (size=%zu)\n", stream, g_stream_map.size());
  fflush(stdout);
  return nullptr;
}

/*static*/ void CudaSyncStream::RegisterStream(cudaStream_t stream, CudaSyncStream* sync_stream) {
  std::lock_guard<std::mutex> lock(g_stream_map_mutex);
  g_stream_map[stream] = sync_stream;
  printf("CudaSyncStream::RegisterStream: %p -> %p (map size=%zu)\n", stream, sync_stream, g_stream_map.size());
  fflush(stdout);
}

/*static*/ void CudaSyncStream::UnregisterStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_stream_map_mutex);
  g_stream_map.erase(stream);
  printf("CudaSyncStream::UnregisterStream: %p (map size=%zu)\n", stream, g_stream_map.size());
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// CudaSyncNotification
// ---------------------------------------------------------------------------

CudaSyncNotification::CudaSyncNotification(CudaSyncStream& stream)
    : OrtSyncNotificationImpl{},
      stream_(stream) {
  ort_version_supported = ORT_API_VERSION;
  Activate = ActivateImpl;
  WaitOnDevice = WaitOnDeviceImpl;
  WaitOnHost = WaitOnHostImpl;
  Release = ReleaseImpl;

  // Create a CUDA event for synchronization (disable timing for performance)
  cudaSetDevice(stream_.GetDeviceId());
  cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}

CudaSyncNotification::~CudaSyncNotification() {
  if (event_) {
    cudaEventDestroy(event_);
  }
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::ActivateImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventRecord(notif->event_, notif->stream_.GetCudaStream()));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::WaitOnDeviceImpl(
    OrtSyncNotificationImpl* this_ptr, OrtSyncStream* stream) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  // SyncStream_GetHandle is in the main ORT API
  cudaStream_t wait_stream = static_cast<cudaStream_t>(Ort::GetApi().SyncStream_GetHandle(stream));
  CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(wait_stream, notif->event_, 0));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::WaitOnHostImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(notif->event_));
  return nullptr;
}

/*static*/ void ORT_API_CALL CudaSyncNotification::ReleaseImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<CudaSyncNotification*>(this_ptr);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
