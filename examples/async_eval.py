"""Execute a bounded evaluation with the asynchronous client."""

import asyncio

from reliquary_sandbox import AsyncSandboxClient, CreateSandboxRequest


async def main() -> None:
    async with AsyncSandboxClient() as client:
        sandbox = await client.create(
            CreateSandboxRequest(
                name="async-eval",
                docker_image="reliquary-sandbox-guest:0.1.0",
                cpu_cores=1,
                memory_gb=1,
                timeout_minutes=10,
            )
        )
        try:
            await client.wait_for_creation(sandbox.id)
            result = await client.execute_command(sandbox.id, "python --version", timeout=30)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr)
            print(result.stdout.strip())
        finally:
            await client.delete(sandbox.id)


if __name__ == "__main__":
    asyncio.run(main())
