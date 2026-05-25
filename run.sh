#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

BENCHMARKS=("AMC23" "MATH")
NUM_SAMPLES=8
MAX_TOKENS=8192
TEMPERATURE=1
RUN_TAG="qwen25_7b_temp${TEMPERATURE}_sample${NUM_SAMPLES}_chat_${MAX_TOKENS}"

for BENCHMARK in "${BENCHMARKS[@]}"; do
  case "${BENCHMARK}" in
    AIME24)
      INPUT="src/datasets/AIME24/test.jsonl"
      ;;
    AMC23)
      INPUT="src/datasets/AMC23/test.jsonl"
      ;;
    MATH)
      INPUT="src/datasets/MATH/Math-OAI.jsonl"
      ;;
    Minerva-MATH)
      INPUT="src/datasets/Minerva-MATH/minerva-math.jsonl"
      ;;
    College_Math)
      INPUT="src/datasets/College_Math/college_math_200.jsonl"
      ;;
    OlympiadBench)
      INPUT="src/datasets/OlympiadBench/olympiadbench_200.jsonl"
      ;;
    *)
      echo "Unknown BENCHMARK: ${BENCHMARK}" >&2
      exit 1
      ;;
  esac

  OUTPUT="outputs/rollouts/${BENCHMARK}/${RUN_TAG}.jsonl"
  READABLE_OUTPUT="outputs/rollouts_readable/${BENCHMARK}/${RUN_TAG}.readable.md"

  printf 'Benchmark: %s\n' "${BENCHMARK}"
  printf 'Input: %s\n' "${INPUT}"
  printf 'JSONL output: %s\n' "${OUTPUT}"
  printf 'Readable output: %s\n\n' "${READABLE_OUTPUT}"

  # Generate rollouts.
  time python scripts/local_rollout/generate_rollouts_mlx.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --benchmark "${BENCHMARK}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p 0.95

  # Convert JSONL rollouts into a readable Markdown file.
  python scripts/local_rollout/make_readable_rollouts.py \
    --input "${OUTPUT}" \
    --output "${READABLE_OUTPUT}" \
    --response-field response

  printf '\nJSONL output: %s\n' "${OUTPUT}"
  printf 'Readable output: %s\n\n' "${READABLE_OUTPUT}"
done
