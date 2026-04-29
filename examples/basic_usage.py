"""Minimal example for the semi-analytic population package."""

import fastropop


def main() -> None:
    print("fastropop version:", fastropop.__version__)
    print("primary model:", "fastropop.SemiAnalyticPopulation")
    print("next step: instantiate the model once JAX dependencies are installed")


if __name__ == "__main__":
    main()
