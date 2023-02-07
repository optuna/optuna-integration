# Contribution Guidelines

It’s an honor to have you on board!

We are proud of this project and have been working to make it great since day one.
We believe you will love it, and we know there’s room for improvement.
We want to
- implement features that make what you want to do possible and/or easy.
- write more tutorials and [examples](https://github.com/optuna/optuna-examples) that help you get familiar with Optuna.
- make issues and pull requests on GitHub fruitful.
- have more conversations and discussions on [GitHub Discussions](https://github.com/optuna/optuna-integration/discussions).

We need your help and everything about Optuna you have in your mind pushes this project forward.
Join Us!

If you feel like giving a hand, here are some ways:
- Implement a feature
    - If you have some cool idea, please open an issue first to discuss design to make your idea in a better shape.
- Send a patch
    - Dirty your hands by tackling [issues with `contribution-welcome` label](https://github.com/optuna/optuna-integration/issues?q=is%3Aissue+is%3Aopen+label%3Acontribution-welcome)
- Report a bug
    - If you find a bug, please report it! Your reports are important.
- Fix/Improve documentation
    - Documentation gets outdated easily and can always be better, so feel free to fix and improve
- Let us and the Optuna community know your ideas and thoughts.
    - __Contribution to Optuna includes not only sending pull requests, but also writing down your comments on issues and pull requests by others, and joining conversations/discussions on [GitHub Discussions](https://github.com/optuna/optuna-integration/discussions).__
    - Also, sharing how you enjoy Optuna is a huge contribution! If you write a blog, let us know about it!


## Pull Request Guidelines

If you make a pull request, please follow the guidelines below:

- [Setup Optuna Integration](#setup-optuna-integration)
- [Checking the Format, Coding Style, and Type Hints](#checking-the-format-coding-style-and-type-hints)
- [Documentation](#documentation)
- [Unit Tests](#unit-tests)
- [Continuous Integration and Local Verification](#continuous-integration-and-local-verification)
- [Creating a Pull Request](#creating-a-pull-request)

Detailed conventions and policies to write, test, and maintain Optuna code are described in the [Optuna Wiki](https://github.com/optuna/optuna/wiki).

- [Coding Style Conventions](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions)
- [Deprecation Policy](https://github.com/optuna/optuna/wiki/Deprecation-policy)
- [Test Policy](https://github.com/optuna/optuna/wiki/Test-Policy)

### Setup Optuna-Integration

Currently, optuna-integration does not work with editable installation.

```bash
git clone git@github.com:YOUR_NAME/optuna-integration.git
cd optuna-integration
pip install .
```

### Checking the Format, Coding Style, and Type Hints

Code is formatted with [black](https://github.com/psf/black),
and docstrings are formatted with [blackdoc](https://github.com/keewis/blackdoc).
Coding style is checked with [flake8](http://flake8.pycqa.org) and [isort](https://pycqa.github.io/isort/),
and additional conventions are described in the [Wiki](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions).
Type hints, [PEP484](https://www.python.org/dev/peps/pep-0484/), are checked with [mypy](http://mypy-lang.org/).

You can check the format, coding style, and type hints at the same time just by executing a script `formats.sh`.
If your environment is missing some dependencies such as black, blackdoc, flake8, isort or mypy,
you will be asked to install them.
The following commands automatically fix format errors by auto-formatters.

```bash
# Install auto-formatters.
$ pip install ".[checking]"

$ ./formats.sh
```

### Documentation

When adding a new feature to the framework, you also need to document it in the reference.

The Optuna-Integration document is still in the Optuna document, if you want to
add to it, please see the
[documentation section of](https://github.com/optuna/optuna/blob/master/CONTRIBUTING.md#documentation)
the Optuna repository.

### Unit Tests

When adding a new feature or fixing a bug, you also need to write sufficient test code.
We use [pytest](https://pytest.org/) as the testing framework and
unit tests are stored under the [tests directory](./tests).

Please install some required packages at first.
```bash
# Install required packages to test.
pip install ".[test]"

# Install required packages on which integration modules depend.
pip install ".[all]"
```

You can run your tests as follows:

```bash
# Run all the unit tests.
pytest

# Run all the unit tests defined in the specified test file.
pytest tests/${TARGET_TEST_FILE_NAME}

# Run the unit test function with the specified name defined in the specified test file.
pytest tests/${TARGET_TEST_FILE_NAME} -k ${TARGET_TEST_FUNCTION_NAME}
```

See also the [Optuna Test Policy](https://github.com/optuna/optuna/wiki/Test-Policy), which describes the principles to write and maintain Optuna tests to meet certain quality requirements.

### Continuous Integration and Local Verification

Optuna repository uses GitHub Actions.

### Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

First, the **title** of your pull request should:

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

*The title will be directly visible in the release notes.*

For example:

- Introduces Tree-structured Parzen Estimator to `optuna.samplers`

Second, the **description** of your pull request should:

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks

## Learning Optuna's Implementation

With Optuna actively being developed and the amount of code growing,
it has become difficult to get a hold of the overall flow from reading the code.
So we created a tiny program called [Minituna](https://github.com/CyberAgentAILab/minituna).
Once you get a good understanding of how Minituna is designed, it will not be too difficult to read the Optuna code.
We encourage you to practice reading the Minituna code with the following article.

[An Introduction to the Implementation of Optuna, a Hyperparameter Optimization Framework](https://medium.com/optuna/an-introduction-to-the-implementation-of-optuna-a-hyperparameter-optimization-framework-33995d9ec354)

