# Third-party notices

The Python SDK includes third-party code distributed under the MIT license.
Its original copyright and permission notice is preserved in
[LICENSES/sdk-MIT.txt](LICENSES/sdk-MIT.txt) and included in built wheels under
`reliquary_sandbox/licenses/`.

The self-hosted controller, runtime integration, deployment configuration,
and evaluation examples are licensed under the root [LICENSE](LICENSE).
The client package does not include a hosted provider's backend or infrastructure.

Docker, gVisor, Python, and Python dependencies retain their own licenses.

The `reliquary-harness` distribution (`harness/`) reproduces texts and logic of
Harbor's Terminus 2 agent (harbor-framework/harbor 06d0e86a, Apache License 2.0)
in `reliquary_harness/terminus.py`; the license is shipped beside it as
`HARBOR_LICENSE.md`, and the upstream files it follows are vendored under
`harness/tests/upstream/terminus/`.
