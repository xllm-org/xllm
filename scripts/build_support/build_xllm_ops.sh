#!/usr/bin/env bash
set -euo pipefail

source_dir=$1
patch_dir=$2
applied=()

restore_patches() {
	local status=$?
	trap - EXIT
	local i
	for ((i = ${#applied[@]} - 1; i >= 0; --i)); do
		git -C "${source_dir}" apply --reverse "${applied[i]}" || status=1
	done
	exit "${status}"
}
trap restore_patches EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for patch in "${patch_dir}"/*.patch; do
	git -C "${source_dir}" apply "${patch}"
	applied+=("${patch}")
done

bash "${source_dir}/build.sh" \
	--build-dir="${XLLM_OPS_BUILD_DIR:-${source_dir}/build}"
