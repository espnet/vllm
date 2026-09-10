# Docker Hub credentials

The Dockerfile and GitHub Actions workflow are configured to build amd64 and
arm64 images and publish them to `espnet/vllm`. Publication requires a Docker
Hub repository and credentials accessible to GitHub Actions.

## Docker Hub access

Create a public repository named `vllm` under the `espnet` organization if it
does not already exist, and grant the CI account push access to it.

Alternatively, the first `docker push` can create the repository automatically
when the token's account has repository-creation permission in the namespace
(Owner or Editor for an organization). Automatic creation uses the namespace's
default visibility; verify that `espnet/vllm` is public after the first push.

Use a retained Read & Write personal access token for that account, or create
one in Docker Hub under **Account settings → Personal access tokens → Generate
new token**. The login username is the token owner's Docker ID; the image
namespace `espnet` may differ from that username.

## GitHub repository secrets

Organization-level secrets are optional. To configure this repository directly:

1. Open [Actions secrets for espnet/vllm](https://github.com/espnet/vllm/settings/secrets/actions).
2. Select **New repository secret**.
3. Enter `DOCKERHUB_USERNAME` as the name and the CI account's Docker ID as
   the value, then select **Add secret**.
4. Select **New repository secret** again. Enter `DOCKERHUB_TOKEN` as the
   name and the account's Read & Write token as the value, then save.

Enter the token directly into GitHub. Existing secret values cannot be read
back from GitHub; use a retained token or create a new one if needed.

If organization secrets with these names already exist, granting `vllm`
access to both is an alternative to creating repository secrets. Credentials
stored only in `espnet/espnet`, including its `docker` environment, are not
inherited by `espnet/vllm`.

## Initial publication

Open the [ESPnet Docker image workflow](https://github.com/espnet/vllm/actions/workflows/espnet-docker.yml),
select **Run workflow**, choose `main`, and leave `publish` enabled. Adding
secrets does not itself trigger a build.

After successful publication, `espnet/vllm:latest` resolves to the image for
the host architecture. Subsequent relevant updates to `main` and the weekly
schedule publish automatically. See [PUBLISHING.md](PUBLISHING.md) for tags,
publication controls, and verification commands.

## References

- [GitHub Actions secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)
- [Docker personal access tokens](https://docs.docker.com/security/access-tokens/personal-access-tokens/)
- [Docker Hub default repository visibility](https://docs.docker.com/docker-hub/settings/)
- [Docker organization roles](https://docs.docker.com/security/roles-and-permissions/core-roles/)
