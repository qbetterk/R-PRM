import argparse
import json
from pathlib import Path


def pick_problem_text(ex):
    source_row = ex.get("source_row", {}) or {}
    for key in ["problem", "question", "query"]:
        if source_row.get(key):
            return source_row[key]
    return ex.get("user_prompt") or ex.get("prompt") or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input rollout JSONL file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file. Defaults to <input>.readable.md.",
    )
    parser.add_argument(
        "--response-field",
        default="response",
        choices=["response", "raw_response", "clean_response"],
        help="Which response field to render in markdown.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if args.output is None:
        out = inp.with_suffix(".readable.md")
    else:
        out = Path(args.output)

    lines = inp.read_text(encoding="utf-8").splitlines()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Readable Rollouts: {inp.name}\n\n")
        f.write(f"Source JSONL: `{inp}`\n\n")
        f.write(f"Rendered response field: `{args.response_field}`\n\n")

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            ex = json.loads(line)
            source_row = ex.get("source_row", {}) or {}

            f.write("---\n\n")
            f.write(
                f"## Row {i}: Problem {ex.get('id')} / "
                f"Candidate {ex.get('candidate_id')}\n\n"
            )
            f.write(f"**Benchmark:** {ex.get('benchmark')}\n\n")
            f.write(f"**Model:** `{ex.get('model')}`\n\n")
            f.write(f"**Temperature:** {ex.get('temperature')}  \n")
            f.write(f"**Top-p:** {ex.get('top_p')}  \n")
            f.write(f"**Max tokens:** {ex.get('max_tokens')}\n\n")

            f.write("### Problem\n\n")
            f.write(pick_problem_text(ex))
            f.write("\n\n")

            f.write("### Gold Answer\n\n")
            f.write(str(source_row.get("answer", "")))
            f.write("\n\n")

            f.write("### Gold Solution\n\n")
            f.write(source_row.get("solution", ""))
            f.write("\n\n")

            f.write(f"### Model Response (`{args.response_field}`)\n\n")
            f.write(ex.get(args.response_field, ""))
            f.write("\n\n")

            if args.response_field != "clean_response" and ex.get("clean_response"):
                f.write("### Clean Response\n\n")
                f.write(ex.get("clean_response", ""))
                f.write("\n\n")

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
