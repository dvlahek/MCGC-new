from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from field_only_cgec import FieldOnlyCGEC


def main():
    parser = argparse.ArgumentParser(description="Generate a field-only CGEC query explanation as JSON.")
    parser.add_argument("--dataset", choices=["digits"], default="digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=122)
    parser.add_argument("--output", type=Path, default=Path("query_explanation.json"))
    args = parser.parse_args()

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=7000 + args.seed,
        stratify=data.target,
    )
    model = FieldOnlyCGEC().fit(X_train, y_train)
    explanation = model.explain_query(X_test[args.query_index], top_support=5)
    explanation["dataset"] = args.dataset
    explanation["seed"] = args.seed
    explanation["query_index"] = args.query_index
    explanation["true_label"] = int(y_test[args.query_index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(explanation, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
