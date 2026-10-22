ARG NODE_BASE
ARG PYTHON_BASE
FROM ${NODE_BASE} AS node
FROM ${PYTHON_BASE}
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
WORKDIR /opt/signed-cli/node
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund --omit=dev && \
    node -e 'if (!process.report.getReport().header.glibcVersionRuntime || process.arch !== "x64") process.exit(1)' && \
    rm -rf node_modules/@anthropic-ai/claude-code-linux-x64-musl \
      node_modules/@anthropic-ai/claude-agent-sdk-linux-x64-musl && \
    npm cache clean --force
COPY requirements.txt /opt/signed-cli/requirements.txt
RUN python -m venv /opt/signed-cli/runtime && \
    /opt/signed-cli/runtime/bin/python -m pip install --no-cache-dir --only-binary=:all: \
      --require-hashes -r /opt/signed-cli/requirements.txt
COPY acp-runner.py /opt/signed-cli/acp-runner.py
RUN mkdir -p /var/tmp/vf-node/bin /var/tmp/vf-codex-0.147.0-1.2.0 \
      /var/tmp/vf-claude-agent-acp-2.1.232-0.67.0 /workspace /home/runner && \
    ln -s /usr/local/bin/node /var/tmp/vf-node/bin/node && \
    ln -s /opt/signed-cli/node /var/tmp/vf-codex-0.147.0-1.2.0/acp && \
    ln -s /opt/signed-cli/node /var/tmp/vf-claude-agent-acp-2.1.232-0.67.0/packages && \
    groupadd --gid 10001 runner && \
    useradd --uid 10001 --gid 10001 --home-dir /home/runner --no-create-home runner && \
    chown 10001:10001 /workspace /home/runner
ENV HOME=/home/runner
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
