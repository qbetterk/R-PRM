import argparse
import json
import os
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_problem(row):
    for key in ["query", "problem", "question"]:
        if key in row and row[key]:
            return row[key]
    raise ValueError(f"No problem text found in row keys: {row.keys()}")


def get_id(row, fallback_idx):
    for key in ["id", "unique_id", "idx", "question_number"]:
        if key in row:
            return row[key]
    return fallback_idx


def make_user_prompt(problem):
    return (
        "Solve the following problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem:\n{problem}"
    )


def make_chat_prompt(tokenizer, problem):
    messages = [
        {"role": "user", "content": make_user_prompt(problem)}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def strip_after_stop_strings(text):
    stop_strings = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
    ]

    earliest = None
    for stop_string in stop_strings:
        idx = text.find(stop_string)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx

    if earliest is None:
        return text.strip()

    return text[:earliest].strip()


def truncate_after_repeated_final(text):
    markers = [
        "Final boxed answer:",
        "Final answer:",
        "Answer:",
    ]

    for marker in markers:
        positions = []
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(marker)

        # Keep through the first final-answer region; cut before it repeats.
        if len(positions) >= 2:
            return text[:positions[1]].strip()

    return text.strip()


def load_existing_keys(output_path):
    done = set()
    invalid_lines = 0
    if not os.path.exists(output_path):
        return done, invalid_lines

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((str(row["id"]), int(row["candidate_id"])))
            except Exception:
                invalid_lines += 1
    return done, invalid_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output file before generation starts.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report existing/pending generations without running inference.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and os.path.exists(args.output):
        print(f"Removing existing output file: {args.output}")
        os.remove(args.output)

    done, invalid_lines = load_existing_keys(args.output)
    expected_total = len(rows) * args.num_samples
    print(f"Output file: {args.output}")
    print(f"Expected generations: {expected_total}")
    print(f"Completed generations found: {len(done)}")
    print(f"Pending generations: {expected_total - len(done)}")
    if invalid_lines:
        print(f"Warning: ignored {invalid_lines} invalid/incomplete JSONL lines in output file.")

    if args.check_only:
        return

    if len(done) >= expected_total:
        print("All generations already completed. Nothing to do.")
        return

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    def write_one_generation(row, problem_id, candidate_id, fout):
        problem = get_problem(row)
        user_prompt = make_user_prompt(problem)
        prompt = make_chat_prompt(tokenizer, problem)

        raw_response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        response = strip_after_stop_strings(raw_response)
        clean_response = truncate_after_repeated_final(response)

        out = {
            "benchmark": args.benchmark,
            "id": problem_id,
            "candidate_id": candidate_id,
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "response": response,
            "raw_response": raw_response,
            "clean_response": clean_response,
            "source_row": row,
        }

        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
        done.add((str(problem_id), int(candidate_id)))

    pending = []
    for idx, row in enumerate(rows):
        problem_id = get_id(row, idx)
        for candidate_id in range(args.num_samples):
            key = (str(problem_id), int(candidate_id))
            if key not in done:
                pending.append((row, problem_id, candidate_id))

    with open(args.output, "a", encoding="utf-8") as fout:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task(args.benchmark, total=len(pending))
            for row, problem_id, candidate_id in pending:
                progress.update(
                    task,
                    description=(
                        f"{args.benchmark} problem={problem_id} "
                        f"candidate={candidate_id}"
                    ),
                )
                write_one_generation(row, problem_id, candidate_id, fout)
                progress.update(task, advance=1)


if __name__ == "__main__":
    main()
