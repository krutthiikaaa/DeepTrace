"""Phase 3 Evaluation -- run all test images through the full pipeline."""
import os
import pprint

from pipeline import run

TEST_DIR = "test_images"
TEST_IMAGES = [
    "1_baseline_sharp.jpg",
    "2_reject_blurry.jpg",
    "3_synthetic_fake.jpg",
]

DIVIDER = "=" * 70


def main():
    print(DIVIDER)
    print("  PHASE 3 EVALUATION -- full pipeline end-to-end")
    print(DIVIDER)

    for filename in TEST_IMAGES:
        path = os.path.join(TEST_DIR, filename)
        print(f"\n{'- ' * 35}")
        print(f"  IMAGE: {filename}")
        print(f"{'- ' * 35}")

        if not os.path.isfile(path):
            print(f"  [SKIP] file not found: {path}")
            continue

        result = run(path)

        # -- verdict summary --
        print(f"  Verdict    : {result.get('verdict', 'N/A')}")
        print(f"  Confidence : {result.get('confidence', 'N/A')}")

        # -- quality gate --
        qg = result.get("quality_gate", {})
        if qg:
            print(f"  Quality Gate: applicable={qg.get('applicable')}, "
                  f"score={qg.get('score')}, note={qg.get('note')}")

        # -- category scores --
        cats = result.get("category_scores", {})
        if cats:
            print("  Category Scores:")
            for cat, score in cats.items():
                print(f"    {cat:30s}: {score:.4f}")

        # -- reasons --
        reasons = result.get("reasons", [])
        if reasons:
            print("  Reasons:")
            for r in reasons:
                print(f"    - {r}")

        # -- individual signals --
        signals = result.get("signals", [])
        if signals:
            print("  Signals:")
            for s in signals:
                app = s.get("applicable", "?")
                sc = s.get("score", 0)
                nm = s.get("signal_name", "?")
                nt = s.get("note", "")
                print(f"    {nm:35s}  score={sc:<8.4f}  applicable={str(app):5s}  {nt}")

    print(f"\n{DIVIDER}")
    print("  EVALUATION COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
