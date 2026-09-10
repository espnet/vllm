# Docker Hub automation

[espnet-docker.yml](../../../.github/workflows/espnet-docker.yml) builds the
current checkout on native GitHub-hosted x86_64 and ARM64 runners. It checks
the complete images before pushing them to **docker.io/espnet/vllm**. Only
after both architecture jobs succeed does it update the shared `latest` tag.
GitHub holds the source and runs the builds; Docker Hub holds the images.

For the two tasks requiring organization access, forward
[OWNER_SETUP.md](OWNER_SETUP.md). The workflow follows ESPnet's existing
`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` convention. No self-hosted runner,
Docker Hub Automated Builds subscription or GitHub app is needed.

## Automatic updates

Relevant changes on `main`, a weekly Monday 03:23 UTC schedule, or **Run
workflow** trigger builds. Scheduled runs rebuild the pinned base with the
current `main` source; updating vLLM itself requires a reviewed base-digest
and compatibility update. A rebuild does not automatically migrate this fork
to a newer upstream vLLM release.

Without Docker Hub secrets, builds and validation still run but publication
is skipped. Once both secrets are present, automatic runs publish without an
additional enable switch. Manual runs default to `publish=true`; deselect it
for a build-only run. Set repository variable `ESPNET_DOCKER_PUBLISH=false` to
pause publication while retaining build validation. `DOCKERHUB_IMAGE` can
optionally override `espnet/vllm` with another authorized `namespace/name`.

Only `espnet/vllm` on `main` can publish. Pull requests do not receive Docker
Hub credentials. The `docker` GitHub environment is used by the jobs; any
required-reviewer rule added to it will also require approval for scheduled
runs. For unattended publication, do not add a required-reviewer rule.

Each run uses `ubuntu-24.04` for amd64 and `ubuntu-24.04-arm` for arm64. The
workflow removes unused preinstalled SDKs on those disposable hosted VMs and
checks for at least 40 GiB of free Docker storage before building. It uses
the default Docker builder's image store to avoid a second full CUDA image
inside a separate BuildKit container. Runner image changes can affect free
space; a failed disk check stops the run before publication.

## Tags and failure behavior

| Tag | Meaning |
| --- | --- |
| `sha-<commit>-<run>-<attempt>-amd64` | Checked x86_64 image from one build |
| `sha-<commit>-<run>-<attempt>-arm64` | Checked ARM64 image from one build |
| `sha-<commit>-<run>-<attempt>` | Manifest combining both successful builds |
| `main`, `latest` | Latest complete build of the current `main` revision |
| `main-amd64`, `main-arm64` | Architecture-specific moving tags |

The run ID distinguishes weekly rebuilds of the same commit with different
resolved dependencies. Deploy by unique tag or registry digest when the exact
image matters. Builds, offline validation and architecture pushes must all
succeed before the promotion job runs. A single-architecture failure leaves
`latest` unchanged, though a successful sibling may already have pushed its
unique tag. If `main` advances during a build, that run publishes only unique
tags. Registry updates to different moving tags are separate operations.

After the first public run, verify:

```bash
docker buildx imagetools inspect espnet/vllm:latest
docker pull espnet/vllm:latest
```

The manifest must list both `linux/amd64` and `linux/arm64`. The public tag is
not created merely by committing this workflow or adding secrets.

## Validation

The Dockerfile installs the explicitly versioned ESPnet serving package
explained in [COMPATIBILITY.md](COMPATIBILITY.md), retaining the official
vLLM CUDA/PyTorch stack. `check_dependencies.py` runs `pip check`, allowing
only the verified NCCL override and ARM cuSPARSELt wheel-tag discrepancy
present in the pinned upstream image, as documented in COMPATIBILITY.md.
Any other dependency error fails the build.

The image must also import all three models, both OpusLM tokenizers, ESPnet
codec/SSL components, torchaudio, torchvision and Xcodec, and start the vLLM
CLI help parser. That help-only command selects the CPU platform because the
builder has no NVIDIA driver; the image's serving default remains CUDA.
GitHub repeats these checks in containers with networking disabled.
These checks do not load LM checkpoints or measure GPU execution, audio
quality, CFG, preemption or throughput. GPU validation results must be
recorded separately; see [README.md](README.md).

## Maintainer commands

The GitHub CLI prompts for secret values; no token belongs in a shell argument
or source file. Repository secrets are sufficient; organization secrets with
access to this repository or secrets in its `docker` environment also work.

```bash
gh secret set DOCKERHUB_USERNAME --repo espnet/vllm
gh secret set DOCKERHUB_TOKEN --repo espnet/vllm
gh workflow run espnet-docker.yml --repo espnet/vllm --ref main -f publish=true
```

If existing organization secrets can be shared, the owner only needs to change
their repository access lists instead of running the first two commands.
Tokens stored only in another repository's environment cannot be read back
or automatically inherited.

## References

- [ESPnet's existing Docker workflow](https://github.com/espnet/espnet/blob/master/.github/workflows/publish_docker_image.yml)
- [Docker Hub repository creation](https://docs.docker.com/docker-hub/repos/create/)
- [Docker personal access tokens](https://docs.docker.com/security/access-tokens/personal-access-tokens/)
- [Docker organization access tokens](https://docs.docker.com/security/access-tokens/organization-access-tokens/)
- [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
