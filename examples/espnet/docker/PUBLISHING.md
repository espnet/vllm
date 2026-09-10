# Docker Hub automation

The workflow in [espnet-docker.yml](../../../.github/workflows/espnet-docker.yml)
builds the current `main` checkout, validates the image locally, and optionally
pushes that exact image to Docker Hub. It follows ESPnet's use of a `docker`
GitHub environment and `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets.
The workflow is prepared for setup; no automated image publication has been
verified from this repository.

The intended public image repository is **`docker.io/espnet/vllm`**. The
workflow defaults to this name; `DOCKERHUB_IMAGE` is an optional override.
GitHub stores and builds the source; Docker Hub stores the resulting images.
This uses GitHub Actions, so Docker Hub's separate **Automated Builds** service
and a Docker Hub connection to GitHub are not required.

```text
espnet/vllm main update / weekly timer / manual run
  -> GitHub Actions -> native Docker builder -> dependency and import checks
  -> docker.io/espnet/vllm:sha-<commit>-<run>-<attempt>-amd64
  -> promote to main-amd64 and latest if main has not advanced
```

## Triggers and image tags

- Changes on `main` affecting the image or Python runtime trigger a build once
  `ESPNET_DOCKER_ENABLED=true` is configured.
- A weekly build runs Monday at 03:23 UTC after the same enable switch is set.
- **Run workflow** works independently of that switch. Its `publish` input
  defaults to `false`, allowing build validation before enabling publication.
- Automatic runs publish only when `ESPNET_DOCKER_PUBLISH=true`; manual runs
  publish only when their explicit `publish` input is selected.
- Only `espnet/vllm` on `main` can execute this job. Pull requests do not run
  code on the configured builder or access Docker Hub credentials.

Each successful publication creates
`sha-<full-source-sha>-<run-id>-<attempt>-amd64` and then advances `main-amd64`
and `latest`. The run ID distinguishes weekly rebuilds of the same source with
different dependency resolutions. A failed build or validation never reaches
the push steps. Tag updates are serialized; individual registry tag updates
are not atomic as a group. Deploy by the unique tag or registry digest when
reproducibility matters. If `main` has advanced during a build, that run only
publishes its unique tag and leaves the moving tags unchanged.

Automatic publication currently covers **linux/amd64 only**. Local `build.sh`
continues to support arm64 and both architectures, with digest-pinned bases.
Before adding an arm64 publication job and a shared multi-platform manifest,
validate the complete runtime on a native arm64 GPU machine.

## One-time setup

1. In [Docker Hub](https://hub.docker.com/), open **My Hub > Repositories >
   Create repository**. Select namespace **espnet**, name **vllm**, and
   visibility **Public**. Creating it requires owner or editor access in the
   Docker Hub organization. The image namespace is independent of the GitHub
   repository name; membership in the ESPnet GitHub organization alone does
   not grant Docker Hub access.
   Obtain a personal access token with Read/Write permission from a Docker ID
   that can push to this repository. Set `DOCKERHUB_USERNAME` to that Docker
   ID, **not** to `espnet` merely because it is the image namespace.
   Alternatively, if the organization has Docker Team/Business, an owner can
   create an organization access token with image push access to this
   repository; for that token type, the login username is `espnet`.
2. Create the GitHub environment `docker`. Add environment secrets
   `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`. Store the token only in GitHub's
   secret settings. If this environment requires review, its review rules also
   apply to scheduled publication.
   An environment named `docker` in `espnet/espnet` is separate from the one in
   `espnet/vllm`. Its secrets are not automatically inherited. An organization
   secret can instead be shared with this repository by the organization
   administrator, provided its Docker Hub identity has the required access.
3. Keep the default image repository `espnet/vllm`, or set repository variable
   `DOCKERHUB_IMAGE` to another authorized `namespace/repository` without a
   registry hostname or tag.
4. Set **repository** variable `ESPNET_DOCKER_RUNNER` to a JSON array of runner
   labels, for example `["self-hosted", "linux", "x64", "espnet-vllm-builder"]`.
   Use a dedicated native amd64 Docker builder with at least **100 GiB free in
   Docker's data directory**, Buildx support and network access to Docker Hub,
   PyPI and the codec download endpoint. A larger GitHub-hosted runner can also
   be selected using its configured label. The fallback `ubuntu-24.04` often
   has insufficient free space for this CUDA image; the workflow checks before
   building. The builder must use a local Docker daemon so the disk check
   measures the daemon's actual storage.
5. Resolve the runtime dependency issue below, then run the workflow manually
   with `publish=false`. Record a GPU smoke run using the resulting image and
   converted checkpoints before the first public release.
6. Run once with `publish=true`, inspect the Docker Hub tag and digest, then set
   **repository** variables `ESPNET_DOCKER_ENABLED=true` and
   `ESPNET_DOCKER_PUBLISH=true` to enable automatic builds and publication.

Repository scope is required for the job enable switch and runner selection
because environment variables are not available when GitHub schedules the job.
To pause publication while retaining automatic validation, set
`ESPNET_DOCKER_PUBLISH=false`. No credential or runner has been configured by
adding these files.

### Exact GitHub setup commands

After creating the Docker Hub repository and registering the native builder,
the following commands configure `espnet/vllm`. Replace the runner label with
the label of the machine actually registered in **Settings > Actions >
Runners**. Merely setting this variable does not provision a machine.

```bash
gh api --method PUT repos/espnet/vllm/environments/docker
gh secret set DOCKERHUB_USERNAME --env docker --repo espnet/vllm
gh secret set DOCKERHUB_TOKEN --env docker --repo espnet/vllm
gh variable set ESPNET_DOCKER_RUNNER --repo espnet/vllm \
  --body '["self-hosted","linux","x64","espnet-vllm-builder"]'
gh workflow run espnet-docker.yml --repo espnet/vllm --ref main \
  -f publish=false
```

The secret commands prompt for values; do not put the token into a commit,
chat message or command argument. Resolve the dependency conflict below and
validate the built image before running the publication commands:

```bash
gh workflow run espnet-docker.yml --repo espnet/vllm --ref main \
  -f publish=true
# After the first successful push and GPU validation:
gh variable set ESPNET_DOCKER_PUBLISH --repo espnet/vllm --body true
gh variable set ESPNET_DOCKER_ENABLED --repo espnet/vllm --body true
docker buildx imagetools inspect espnet/vllm:latest
docker pull espnet/vllm:latest
```

The last two commands work only after an image has actually been published.
The current automated `latest` tag is amd64, not a promise of ARM support.
Both architectures can live in this same Docker Hub repository. When native
ARM build and GPU validation are available, add an ARM build job and promote
a shared manifest only after both architecture jobs succeed; a pull then
selects the matching image automatically. Do not have independent architecture
jobs overwrite the same `latest` tag.

## Runtime issue to resolve before enabling publication

As checked on 2026-09-09, the Dockerfile installs unpinned `espnet` and then
force-reinstalls `torch==2.13.0`. PyPI's current `espnet==202609.post1` metadata
requires `torch>=2.9.1,<2.12`. Sequential pip installations can succeed while
leaving this declared dependency unsatisfied. Reinstalling torch can also
replace CUDA dependencies inherited from the base image.

The workflow deliberately runs `python3 -m pip check` before publication, so
this conflict must be resolved first. Choose and pin a compatible ESPnet
release or reviewed ESPnet revision together with the vLLM runtime, using the
maintainer's working local image as evidence. Older releases should not be
selected on their torch constraint alone: for example, ESPnet 202511 requires
`setuptools<74`, conflicting with this vLLM build's `setuptools>=77.0.3`.
Do not bypass the check or assume successful installation proves compatibility.

The subsequent offline smoke check verifies the vLLM import path, package
versions, a CUDA-enabled torch build, the native extension's presence, all three model
imports, both OpusLM tokenizers, codec/SSL dependencies and CLI startup.
It does **not** validate audio quality, GPU kernel execution, model loading,
CFG, preemption or throughput. Run the documented clients against converted
checkpoints on a GPU for those checks.

## Using an existing local image

First record its tag, architecture, package versions, source revision and
successful model requests. For example, on the machine holding the image:

```bash
IMAGE=your-local-image:tag
docker image inspect "$IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
docker run --rm --network none --entrypoint python3 "$IMAGE" -m pip check
docker run --rm --network none --entrypoint python3 "$IMAGE" -m pip freeze
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/examples/espnet/docker/smoke_test.py:/tmp/espnet-smoke.py:ro" \
  "$IMAGE" /tmp/espnet-smoke.py
```

The smoke script expects the fork's documented `/workspace/vllm-fork` layout.
If the existing image uses another layout, inspect that difference before
adapting the assertion. A local image can supply build cache on the same
builder, or be published once under a distinct bootstrap tag after validation.
It is not accessible from another GitHub runner until transferred or pushed to
a registry. Publishing it once does not establish automatic updates: future
images still need a reproducible Dockerfile and the workflow above.

## References

- [Docker Hub: create a repository](https://docs.docker.com/docker-hub/repos/create/)
- [Docker: personal access tokens](https://docs.docker.com/security/access-tokens/personal-access-tokens/)
- [Docker: organization access tokens](https://docs.docker.com/security/access-tokens/organization-access-tokens/)
- [ESPnet's Docker publication workflow](https://github.com/espnet/espnet/blob/master/.github/workflows/publish_docker_image.yml)
  runs weekly on `master` and supports manual dispatch.
- [Docker: test before push](https://docs.docker.com/build/ci/github-actions/test-before-push/)
  describes local validation before publication.
- [GitHub: larger runners](https://docs.github.com/en/actions/reference/runners/larger-runners)
  lists larger storage options.
- [ESPnet 202609.post1 package metadata](https://pypi.org/pypi/espnet/202609.post1/json)
  records the declared torch dependency.
