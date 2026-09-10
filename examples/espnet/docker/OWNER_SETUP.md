# One-time owner setup for espnet/vllm

The repository contains the Dockerfile and GitHub Actions workflow. GitHub
builds x86_64 and ARM64 images and updates Docker Hub after both builds pass.
Please complete these two setup steps:

1. **Docker Hub:** create a **Public** repository named **vllm** under the
   **espnet** organization. Give the Docker identity already used by ESPnet's
   CI permission to push to it. [Create repository](https://hub.docker.com/repository/create)
2. **GitHub:** make the following Actions secrets available to
   **espnet/vllm**: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
   [Repository secrets](https://github.com/espnet/vllm/settings/secrets/actions)

If these are already organization secrets, simply include `espnet/vllm` in
both secrets' repository access lists. Otherwise add them as repository
secrets using the existing CI identity's Docker Hub Read/Write access token,
or a new token for that identity. For a personal access token, the username
is the token owner's Docker ID; **espnet** is the image namespace, not
necessarily the login username. Paste the token directly into GitHub's secret
form; it does not need to be shared with Haoran or sent in a message.

Secrets stored only in the `docker` environment of `espnet/espnet` do not
carry over to `espnet/vllm`, and GitHub cannot reveal their original values.
Use a retained token or create a new one if necessary. An organization access
token is optional for organizations that already have the required Docker
subscription; a personal access token works without that extra setup.

That completes the owner's setup. No runner machine, Docker Hub Automated
Build, GitHub app installation, code change, or enable variable is required.
The next relevant `main` update or weekly run publishes automatically.
Haoran can also start the [ESPnet Docker image workflow](https://github.com/espnet/vllm/actions/workflows/espnet-docker.yml)
using **Run workflow** with `publish=true` and verify the first public image.
The owner does not need to build or push an image manually.

The resulting image name is `espnet/vllm:latest`; Docker selects x86_64 or
ARM64 automatically. Until the first successful publication, this tag may
not exist. See [PUBLISHING.md](PUBLISHING.md) for workflow details.
