# Evolution Science

[![codecov](https://codecov.io/gh/maycuatroi/evo-science/branch/main/graph/badge.svg?token=evo-science_token_here)](https://codecov.io/gh/maycuatroi/evo-science)
[![CI](https://github.com/maycuatroi/evo-science/actions/workflows/main.yml/badge.svg)](https://github.com/maycuatroi/evo-science/actions/workflows/main.yml)

Awesome evo_science created by maycuatroi

## Install it from PyPI

```bash
pip install evo-science
```

## Example
```python
    model = LinearRegressionModel()

    x = FeatureSet(features=[PClass, Sex, Age, SibSp, Parch, Fare])
    y = FeatureSet(features=[Survived])

    (x + y).build(
        csv_path="https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    model.fit(x=x, y=y)
    model.evaluate(x=x, y=y, metrics=[Slope, ErrorStd])
    model.calculate_coefficients(x=x)
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
