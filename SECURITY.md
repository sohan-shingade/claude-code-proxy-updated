# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub's security advisory form](https://github.com/sohan-shingade/claude-code-proxy-updated/security/advisories/new). Do not open a public issue with exploit details, credentials, or unredacted request captures.

Include affected versions, reproduction steps, impact, and any suggested mitigation. You should receive an acknowledgment within seven days.

## Supported versions

Security fixes are applied to the latest release. Users should upgrade to the newest published version before reporting an issue already fixed there.

## Deployment

The proxy binds to loopback by default. A non-loopback bind requires `CCP_PROXY_AUTH_TOKEN`; `/v1/*` clients must send that token as `Authorization: Bearer <token>` or `x-api-key: <token>`. `/healthz` remains public. Use TLS or a trusted private network when traffic leaves the host, because the built-in listener provides authentication but not encryption.
