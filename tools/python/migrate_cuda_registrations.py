#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Generate CUDA plugin kernel registration entries from the internal CUDA EP
function table in `cuda_execution_provider.cc`.

The output is a C++ include file with entries in this form:
  {"Add", 7, 12, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TYPE_TO_ORT_ENUM = {
    "bool": "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
    "float": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
    "double": "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
    "double_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
    "MLFloat16": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16",
    "BFloat16": "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16",
    "int8_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
    "int16_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
    "int32_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
    "int64_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
    "uint8_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8",
    "uint16_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16",
    "uint32_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32",
    "uint64_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64",
    "Float8E4M3FN": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN",
    "Float8E5M2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2",
    "Float8E4M3FNUZ": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ",
    "Float8E5M2FNUZ": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ",
    "Float4E2M1x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
    "UInt4x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4",
    "Int4x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4",
}

MACRO_PATTERN = re.compile(r"BuildKernelCreateInfo\s*<\s*([^>]+)\s*>")


@dataclass(frozen=True, order=True)
class Entry:
    op_type: str
    since_version_start: int
    since_version_end: int
    type_enum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("onnxruntime/core/providers/cuda/cuda_execution_provider.cc"),
        help="Path to cuda_execution_provider.cc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc"),
        help="Output .inc file path",
    )
    parser.add_argument(
        "--ops",
        nargs="*",
        default=["Add", "Relu", "MatMul", "Gemm", "Conv"],
        help="Optional op-type allowlist. Empty means no op filter.",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=["float"],
        help="Optional type-token allowlist. Empty means no type filter.",
    )
    parser.add_argument(
        "--domain",
        default="kOnnxDomain",
        help="Only include registrations with this domain token.",
    )
    parser.add_argument(
        "--max-opset",
        type=int,
        default=23,
        help="Upper opset bound used for non-versioned registrations.",
    )
    return parser.parse_args()


def split_args(arg_blob: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in arg_blob:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_macro(macro_expr: str, max_opset: int) -> tuple[str, int, int, str] | None:
    m = re.match(r"([A-Z0-9_]+)\s*\((.*)\)$", macro_expr.strip())
    if not m:
        return None

    macro_name = m.group(1)
    args = split_args(m.group(2))

    if macro_name == "ONNX_OPERATOR_KERNEL_CLASS_NAME" and len(args) == 4:
        _, _, since, op = args
        return op, int(since), max_opset, "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED"

    if macro_name == "ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME" and len(args) == 5:
        _, _, start, end, op = args
        return op, int(start), int(end), "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED"

    if macro_name == "ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME" and len(args) == 5:
        _, _, since, t, op = args
        type_enum = TYPE_TO_ORT_ENUM.get(t)
        if type_enum is None:
            return None
        return op, int(since), max_opset, type_enum

    if macro_name == "ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME" and len(args) == 6:
        _, _, start, end, t, op = args
        type_enum = TYPE_TO_ORT_ENUM.get(t)
        if type_enum is None:
            return None
        return op, int(start), int(end), type_enum

    return None


def extract_function_table(source_text: str) -> str:
    start = source_text.find("static const BuildKernelCreateInfoFn function_table[] = {")
    if start < 0:
        raise RuntimeError("Failed to locate CUDA function_table start")

    end = source_text.find("};", start)
    if end < 0:
        raise RuntimeError("Failed to locate CUDA function_table end")

    return source_text[start:end]


def iter_entries(
    source_text: str,
    domain_filter: str,
    max_opset: int,
    op_filter: set[str],
    type_filter: set[str],
) -> Iterable[Entry]:
    table_text = extract_function_table(source_text)

    for line in table_text.splitlines():
        if "BuildKernelCreateInfo<" not in line:
            continue

        macro_match = MACRO_PATTERN.search(line)
        if not macro_match:
            continue

        macro_expr = macro_match.group(1).strip()
        parsed = parse_macro(macro_expr, max_opset)
        if parsed is None:
            continue

        op, start, end, type_enum = parsed
        macro_name, arg_blob = re.match(r"([A-Z0-9_]+)\s*\((.*)\)$", macro_expr).groups()
        args = split_args(arg_blob)
        if len(args) < 2:
            continue
        domain = args[1]
        if domain != domain_filter:
            continue

        if op_filter and op not in op_filter:
            continue

        if type_filter and macro_name in {
            "ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME",
            "ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME",
        }:
            type_token = args[-2]
            if type_token not in type_filter:
                continue

        if start > end:
            continue

        yield Entry(op, start, end, type_enum)


def write_output(output_path: Path, entries: list[Entry], argv: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "// This file is generated by tools/python/migrate_cuda_registrations.py",
        "// Do not edit by hand.",
        f"// Command: {' '.join(argv)}",
        "",
    ]

    for e in entries:
        lines.append(
            f'{{"{e.op_type}", {e.since_version_start}, {e.since_version_end}, {e.type_enum}}},'
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    source_text = args.input.read_text(encoding="utf-8")
    op_filter = set(args.ops) if args.ops else set()
    type_filter = set(args.types) if args.types else set()

    entries = sorted(
        set(
            iter_entries(
                source_text,
                domain_filter=args.domain,
                max_opset=args.max_opset,
                op_filter=op_filter,
                type_filter=type_filter,
            )
        )
    )

    write_output(
        args.output,
        entries,
        argv=["python", "tools/python/migrate_cuda_registrations.py", *sys.argv[1:]],
    )

    print(
        f"Generated {len(entries)} registrations to {args.output} "
        f"(ops={','.join(args.ops) if args.ops else 'ALL'}, "
        f"types={','.join(args.types) if args.types else 'ALL'})"
    )


if __name__ == "__main__":
    main()
