from pathlib import Path
import re


KEEP_SEEDS = {"1018", "1428", "2026", "2138", "3053", "3521", "4083", "4396", "5035", "5472"}
RESULT_EXTENSIONS = {".txt", ".log", ".pt", ".png", ".csv"}
SEED_PATTERN = re.compile(r"(?<!\d)\d{3,6}(?!\d)")


def should_delete(path):
    if path.suffix.lower() not in RESULT_EXTENSIONS:
        return False

    matches = SEED_PATTERN.findall(path.name)
    if matches and not any(seed in KEEP_SEEDS for seed in matches):
        return True

    return path.parent.name == "C_LoReW_hparam_search" and path.suffix.lower() in RESULT_EXTENSIONS


def main():
    base_dir = Path(__file__).resolve().parent
    deleted = []

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue

        resolved = path.resolve()
        if base_dir not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside {base_dir}: {resolved}")

        if should_delete(path):
            path.unlink()
            deleted.append(resolved)

    remaining_bad = []
    for path in base_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RESULT_EXTENSIONS:
            continue
        matches = SEED_PATTERN.findall(path.name)
        if matches and not any(seed in KEEP_SEEDS for seed in matches):
            remaining_bad.append(path.resolve())

    print(f"Deleted files: {len(deleted)}")
    print(f"Remaining non-selected seeded result files: {len(remaining_bad)}")
    for path in remaining_bad[:20]:
        print(path)


if __name__ == "__main__":
    main()
