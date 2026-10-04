"""Run a small evaluation, exchange files, and clean up the guest."""

import json
import tempfile
from pathlib import Path

from reliquary_sandbox import APIClient, CreateSandboxRequest, SandboxClient


def main() -> None:
    api = APIClient()
    client = SandboxClient(api)
    sandbox = client.create(
        CreateSandboxRequest(
            name="python-eval",
            docker_image="reliquary-sandbox-guest:0.1.0",
            cpu_cores=1,
            memory_gb=1,
            timeout_minutes=10,
        )
    )
    try:
        client.wait_for_creation(sandbox.id)
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            source = local / "eval.py"
            source.write_text(
                "import json\n"
                "from pathlib import Path\n"
                "score = sum(n * n for n in range(10))\n"
                "Path('/workspace/result.json').write_text(json.dumps({'score': score}))\n"
                "print('evaluation complete')\n"
            )
            client.upload_file(sandbox.id, "/workspace/eval.py", str(source))
            result = client.execute_command(sandbox.id, "python /workspace/eval.py", timeout=30)
            if result.exit_code != 0:
                raise RuntimeError(f"Evaluation exited {result.exit_code}: {result.stderr}")
            destination = local / "result.json"
            client.download_file(sandbox.id, "/workspace/result.json", str(destination))
            print(json.dumps(json.loads(destination.read_text()), indent=2))
    finally:
        client.delete(sandbox.id)
        api.client.close()


if __name__ == "__main__":
    main()
