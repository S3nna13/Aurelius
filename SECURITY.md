# Security Practices for Aurelius

## Container Images
- All Docker images now run as non‑root users (`aurelius`, `app`, or `aurelius` for the Rust gateway).
- Base images are pinned to specific digests to ensure reproducible builds:
  - `python@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3`
  - `node@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f`
  - `rust@sha256:4333721398de61f53ccbe53b0b855bcc4bb49e55828e8f652d7a8ac33dd0c118`
  - `alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc`

## CI / CD Hardening
- Added **Trivy** container‑image scanning in the CI workflow (`trivy-scan` job). The scan fails on any **HIGH** or **CRITICAL** findings.
- Concurrency groups with `cancel-in-progress` prevent duplicate pipeline runs.
- npm audit steps already present for Node.js layers; they now run with `continue‑on‑error` to surface issues without breaking the run.

## Helm Chart
- Helm `values.yaml` still uses version tags for development, but a comment reminds to replace them with digests (e.g. `tag: "@sha256:<digest>"`) before production deployments.

## CORS
- Updated `src/serving/cors_middleware.py` to disallow credentials when `allowed_origins="*"`, complying with the CORS specification.

## Further Work
- After publishing images to the registry, replace Helm `tag` entries with the concrete digest (e.g. `tag: "@sha256:..."`).
- Periodically review Trivy scan results and address any new vulnerabilities.
