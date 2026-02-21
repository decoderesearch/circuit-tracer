# Contributing to circuit-tracer

Thank you for your interest in contributing to circuit-tracer! We appreciate the community involvement we've seen so far and welcome contributions.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers the project.

## Maintenance Bandwidth

We maintain this project on a **best-effort basis**.

- PR reviews may take time and we cannot guarantee timely responses or merges
- Issues may not receive immediate attention

### API Stability

⚠️ **Warning**: This library is under active development and **breaking changes are possible**. The API is not stable and breaking changes may occur in any release.

## How to Contribute

We encourage contributions! Here's how you can help:

1. **Make your changes** with clear, descriptive commits
2. **Test your changes** - see Testing section below
3. **Submit a Pull Request** with a clear description of your changes

## Installing with dev dependencies

```bash
pip install -e ".[dev]"
```

## Linting / formatting

We use `ruff check` and `ruff format` for code quality. When contributing:

- Run `ruff check` and `ruff format` on your changes
- Run existing tests to ensure nothing breaks
- Check that relevant demo notebooks still execute correctly, particularly:
  - `demos/circuit_tracing_tutorial.ipynb`
  - `demos/attribute_demo.ipynb`
  - `demos/intervention_demo.ipynb`

## Tests

We use `pytest` for testing. When contributing:

- Add tests for new functionality where possible
- Run `pytest` before submitting a PR

## Type checking

We use `pyright` for type checking. When contributing:

- Run `pyright` before submitting a PR

## What We're Looking For

- Bug fixes
- Performance enhancements
- New features that align with the project's goals
- In future: updates to support new models or transcoders (currently blocked on our pipeline for generating feature activation examples)

## Before Contributing

- Check existing issues and PRs to avoid duplicate work
- For major changes, consider opening an issue first to discuss the approach
- Understand that we cannot commit to reviewing all changes

## Code of Conduct

Please be respectful and constructive in all interactions.
