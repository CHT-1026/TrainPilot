#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export TRAINPILOT_GRPO_CONFIG="${TRAINPILOT_GRPO_CONFIG:-model/config/grpo_v2.json}"
export TRAINPILOT_GRPO_DATA="${TRAINPILOT_GRPO_DATA:-model/dataset/datasets/trainpilot_agent/grpo_v2}"
export TRAINPILOT_GRPO_RUN="${TRAINPILOT_GRPO_RUN:-runs/trainpilot_grpo_v2}"
export TRAINPILOT_GRPO_EXTRA_TEST=test.test_code.test_grpo_v2
exec bash model/trainer/run_trainpilot_grpo.sh "$@"
