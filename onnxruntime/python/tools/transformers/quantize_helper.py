# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os

import torch
# No longer need to import Conv1D or onnx (unless used elsewhere)

logger = logging.getLogger(__name__)


def _get_size_of_pytorch_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size


class QuantizeHelper:
    @staticmethod
    def quantize_torch_model(model, dtype=torch.qint8):
        """
        Usage: model = quantize_model(model)
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype
        )

        logger.info(f"Size of full precision Torch model(MB): {_get_size_of_pytorch_model(model)}")
        logger.info(f"Size of quantized Torch model(MB): {_get_size_of_pytorch_model(quantized_model)}")
        return quantized_model

    @staticmethod
    def quantize_onnx_model(onnx_model_path, quantized_model_path, use_external_data_format=False):
        from pathlib import Path  # noqa: PLC0415

        from onnxruntime.quantization import quantize_dynamic  # noqa: PLC0415

        Path(quantized_model_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Size of full precision ONNX model(MB):{os.path.getsize(onnx_model_path) / (1024 * 1024)}")
        quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            use_external_data_format=use_external_data_format,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        logger.info(f"quantized model saved to:{quantized_model_path}")
        # TODO: inlcude external data in total model size.
        logger.info(f"Size of quantized ONNX model(MB):{os.path.getsize(quantized_model_path) / (1024 * 1024)}")
