#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
CUDA EP Plugin Registration Parity Report

Compares kernel registrations between the bundled CUDA EP and the plugin CUDA EP
by statically parsing source files. Produces a report showing which ops are in
both builds, only in bundled, or only in plugin.

Usage:
    python tools/ci_build/cuda_plugin_parity_report.py [--repo-root /path/to/onnxruntime]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Regex patterns for kernel registration macros
# These macros define kernel classes and are the source of truth for op registrations.
KERNEL_EX_PATTERNS = [
    # ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # version
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_TYPED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # version
        r"(\w+)\s*,\s*"  # type
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, start_ver, end_ver, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_VERSIONED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # start_version
        r"(\d+)\s*,\s*"  # end_version
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, start_ver, end_ver, type, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # start_version
        r"(\d+)\s*,\s*"  # end_version
        r"(\w+)\s*,\s*"  # type
        r"(\w+)\s*,"  # provider
    ),
]

# Patterns for contrib ops (CUDA_MS_OP macros expand to ONNX_OPERATOR macros internally)
# Just match the ONNX_OPERATOR_*_KERNEL_EX patterns since the CUDA_MS_OP macros are wrappers.


def extract_kernel_registrations(file_path):
    """Extract (op_name, domain, since_version) tuples from kernel registration macros in a file."""
    registrations = []
    try:
        content = Path(file_path).read_text(errors="replace")
    except OSError:
        return registrations

    # Remove C/C++ comments to avoid false matches
    content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    # Join lines ending with \ (macro continuations) so multi-line macros become single-line
    content = re.sub(r"\\\s*\n\s*", " ", content)

    for pattern in KERNEL_EX_PATTERNS:
        for m in pattern.finditer(content):
            groups = m.groups()
            op_name = groups[0]
            domain = groups[1]
            since_version = int(groups[2])
            registrations.append((op_name, domain, since_version, str(file_path)))

    return registrations


def parse_registration_table(file_path, table_func_name):
    """Parse the registration table function to extract op names referenced in BuildKernelCreateInfo calls."""
    registrations = set()
    try:
        content = Path(file_path).read_text(errors="replace")
    except OSError:
        return registrations

    # Find the function
    func_start = content.find(f"{table_func_name}")
    if func_start < 0:
        return registrations

    # Extract class names from BuildKernelCreateInfo<CLASS_NAME> entries
    # Pattern: ONNX_OPERATOR_*_KERNEL_CLASS_NAME(provider, domain, ver, [type,] name)
    class_name_patterns = [
        # ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)
        re.compile(r"ONNX_OPERATOR_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)"),
        # ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)
        re.compile(
            r"ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\w+\s*,\s*(\w+)\s*\)"
        ),
        # ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, start, end, name)
        re.compile(
            r"ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\d+\s*,\s*(\w+)\s*\)"
        ),
        # ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, start, end, type, name)
        re.compile(
            r"ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\w+\s*,\s*(\w+)\s*\)"
        ),
    ]

    # Also handle CUDA_MS_OP_CLASS_NAME and CUDA_MS_OP_TYPED_CLASS_NAME
    ms_op_patterns = [
        # CUDA_MS_OP_CLASS_NAME(ver, name)  -> domain is kMSDomain
        re.compile(r"CUDA_MS_OP_CLASS_NAME\s*\(\s*(\d+)\s*,\s*(\w+)\s*\)"),
        # CUDA_MS_OP_TYPED_CLASS_NAME(ver, type, name)
        re.compile(r"CUDA_MS_OP_TYPED_CLASS_NAME\s*\(\s*(\d+)\s*,\s*\w+\s*,\s*(\w+)\s*\)"),
    ]

    # Scan from function start to end of file (conservative)
    search_region = content[func_start:]

    for pattern in class_name_patterns:
        for m in pattern.finditer(search_region):
            domain, version, name = m.group(1), int(m.group(2)), m.group(3)
            registrations.add((name, domain, version))

    for pattern in ms_op_patterns:
        for m in pattern.finditer(search_region):
            version, name = int(m.group(1)), m.group(2)
            registrations.add((name, "kMSDomain", version))

    return registrations


def get_excluded_files(cmake_path, repo_root):
    """Parse the plugin CMake file to get regex exclusion patterns."""
    exclusion_patterns = []
    try:
        content = Path(cmake_path).read_text()
    except OSError:
        return exclusion_patterns

    # Match: list(FILTER CC_SRCS EXCLUDE REGEX "pattern")
    # or:    list(FILTER CU_SRCS EXCLUDE REGEX "pattern")
    for m in re.finditer(r'list\s*\(\s*FILTER\s+\w+\s+EXCLUDE\s+REGEX\s+"([^"]+)"\s*\)', content):
        pat = m.group(1)
        # Only keep non-commented lines
        line_start = content.rfind("\n", 0, m.start()) + 1
        line = content[line_start : m.start()]
        if "#" not in line:
            exclusion_patterns.append(pat)

    return exclusion_patterns


def find_kernel_files(base_dirs, extensions=(".cc",)):
    """Find all kernel source files in the given directories."""
    files = []
    for base_dir in base_dirs:
        for ext in extensions:
            for path in Path(base_dir).rglob(f"*{ext}"):
                files.append(str(path))
    return sorted(files)


def is_excluded(file_path, exclusion_patterns):
    """Check if a file path matches any exclusion pattern."""
    for pat in exclusion_patterns:
        if re.search(pat, file_path):
            return True
    return False


def generate_report(repo_root):
    """Generate the full parity report."""
    repo_root = Path(repo_root)

    # Paths
    cuda_ep_cc = repo_root / "onnxruntime/core/providers/cuda/cuda_execution_provider.cc"
    cuda_nhwc_cc = repo_root / "onnxruntime/core/providers/cuda/cuda_nhwc_kernels.cc"
    contrib_cc = repo_root / "onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"
    plugin_cmake = repo_root / "cmake/onnxruntime_providers_cuda_plugin.cmake"

    # 1. Parse bundled EP registration tables
    bundled_standard = parse_registration_table(cuda_ep_cc, "RegisterCudaKernels")
    bundled_nhwc = parse_registration_table(cuda_nhwc_cc, "RegisterCudaKernels")  # NHWC uses same function name pattern
    bundled_contrib = parse_registration_table(contrib_cc, "RegisterCudaContribKernels")

    # 2. Get exclusion patterns from plugin CMake
    exclusion_patterns = get_excluded_files(plugin_cmake, repo_root)

    # 3. Scan all CUDA kernel source files
    core_cuda_dir = repo_root / "onnxruntime/core/providers/cuda"
    contrib_cuda_dir = repo_root / "onnxruntime/contrib_ops/cuda"

    all_cc_files = find_kernel_files([core_cuda_dir, contrib_cuda_dir])

    # 4. Categorize files and extract registrations
    plugin_registrations = []  # (op, domain, ver, file) tuples - compiled into plugin
    excluded_registrations = []  # (op, domain, ver, file) tuples - excluded from plugin

    for f in all_cc_files:
        regs = extract_kernel_registrations(f)
        if not regs:
            continue
        if is_excluded(f, exclusion_patterns):
            excluded_registrations.extend(regs)
        else:
            plugin_registrations.extend(regs)

    # 5. Build op sets for comparison
    plugin_ops = set()
    for op, domain, ver, _ in plugin_registrations:
        plugin_ops.add((op, domain, ver))

    excluded_ops = set()
    for op, domain, ver, _ in excluded_registrations:
        excluded_ops.add((op, domain, ver))

    # Unique op names (ignoring version/type variants)
    plugin_op_names = set((op, domain) for op, domain, _ in plugin_ops)
    excluded_op_names = set((op, domain) for op, domain, _ in excluded_ops)
    bundled_op_names = set()
    for ops_set in [bundled_standard, bundled_nhwc, bundled_contrib]:
        for op, domain, ver in ops_set:
            bundled_op_names.add((op, domain))

    # 6. Generate report
    report = []
    report.append("=" * 70)
    report.append("CUDA EP Plugin — Kernel Registration Parity Report")
    report.append("=" * 70)
    report.append("")

    report.append("## Summary")
    report.append("  NOTE: Plugin macro counts may undercount due to nested macro")
    report.append("  expansion (e.g., BINARY_OP_VERSIONED_UZILHFD wraps multiple")
    report.append("  ONNX_OPERATOR_TYPED_KERNEL_EX calls). Bundled table counts")
    report.append("  from RegisterCudaKernels/RegisterCudaContribKernels are accurate.")
    report.append("")
    report.append("  Bundled EP registration table entries:")
    report.append(f"    Standard ops:  {len(bundled_standard)}")
    report.append(f"    NHWC ops:      {len(bundled_nhwc)}")
    report.append(f"    Contrib ops:   {len(bundled_contrib)}")
    report.append(f"    Total:         {len(bundled_standard) + len(bundled_nhwc) + len(bundled_contrib)}")
    report.append("")
    report.append("  Plugin kernel macro invocations (in compiled .cc files):")
    report.append(f"    Total:         {len(plugin_registrations)}")
    report.append("  Excluded kernel macro invocations:")
    report.append(f"    Total:         {len(excluded_registrations)}")
    report.append("")
    report.append("  Unique op names (op, domain):")
    report.append(f"    In plugin:     {len(plugin_op_names)}")
    report.append(f"    Excluded:      {len(excluded_op_names)}")
    report.append(f"    In bundled:    {len(bundled_op_names)}")
    report.append("")

    # Plugin-only ops (in plugin but not in bundled table — likely already handled)
    plugin_only = plugin_op_names - bundled_op_names
    if plugin_only:
        report.append(f"  Plugin-only op names (not in bundled table): {len(plugin_only)}")
        for op, domain in sorted(plugin_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    # Bundled-only ops (in bundled but not in plugin+excluded)
    all_source_ops = plugin_op_names | excluded_op_names
    bundled_only = bundled_op_names - all_source_ops
    if bundled_only:
        report.append(f"  Bundled-only op names (not in any .cc file KERNEL_EX): {len(bundled_only)}")
        for op, domain in sorted(bundled_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    # Coverage ratio
    if bundled_op_names:
        coverage = len(plugin_op_names & bundled_op_names) / len(bundled_op_names) * 100
        report.append(f"  Plugin coverage: {coverage:.1f}% of bundled unique op names")
    report.append("")

    # 7. Excluded ops detail
    report.append("## Excluded Ops by Category")
    report.append("")

    # Group excluded by file/directory
    excluded_by_dir = defaultdict(list)
    for op, domain, ver, filepath in excluded_registrations:
        # Extract a short category from the path
        rel_path = str(filepath).replace(str(repo_root) + "/", "")
        parts = rel_path.split("/")
        # Find the most descriptive sub-directory
        if "contrib_ops" in rel_path:
            idx = parts.index("cuda") if "cuda" in parts else 0
            category = "/".join(parts[idx + 1 : -1]) or parts[-1]
        elif "core/providers/cuda" in rel_path:
            idx = [i for i, p in enumerate(parts) if p == "cuda"][-1]
            category = "/".join(parts[idx + 1 : -1]) or parts[-1]
        else:
            category = "other"
        excluded_by_dir[category].append((op, domain, ver, rel_path))

    for category in sorted(excluded_by_dir):
        entries = excluded_by_dir[category]
        unique_ops = set((op, domain) for op, domain, _, _ in entries)
        report.append(f"  [{category}] ({len(entries)} registrations, {len(unique_ops)} unique ops)")
        for op, domain in sorted(unique_ops):
            report.append(f"    - {op} ({domain})")
        report.append("")

    report.append("## Active CMake Exclusion Patterns")
    for i, pat in enumerate(exclusion_patterns, 1):
        report.append(f"  {i:2d}. {pat}")
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="CUDA EP Plugin Registration Parity Report")
    parser.add_argument("--repo-root", default=None, help="Path to onnxruntime repo root")
    args = parser.parse_args()

    if args.repo_root:
        repo_root = args.repo_root
    else:
        # Try to detect from script location
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        if not (Path(repo_root) / "onnxruntime").exists():
            print("Error: Could not find repo root. Use --repo-root flag.", file=sys.stderr)
            sys.exit(1)

    report = generate_report(repo_root)
    print(report)


if __name__ == "__main__":
    main()
