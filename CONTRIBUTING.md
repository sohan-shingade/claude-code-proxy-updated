# Contributing

Thank you for contributing to claude-code-proxy.

This repository is a maintained fork of [raine/claude-code-proxy](https://github.com/raine/claude-code-proxy). Please open issues and pull requests against this repository for changes specific to this fork.

## Development

Install a stable Rust toolchain, then run:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
```

Keep changes focused, add regression tests for behavior changes, and do not include credentials or unredacted traffic captures. Update public documentation and the changelog when user-visible behavior changes.

## Pull requests

Describe the problem, the chosen behavior, and the commands used to verify the change. By contributing, you agree that your contribution is licensed under the repository's MIT License.
