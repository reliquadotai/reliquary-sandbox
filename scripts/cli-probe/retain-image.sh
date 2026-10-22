#!/usr/bin/env bash
# Retain exact image bytes as ciphertext; no plaintext archive touches disk.
set -euo pipefail
umask 077
image=$1
out=$2
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ ! -e "$out" ]]
mkdir -m 700 "$out"
cert="$(dirname "$0")/recipient.crt"
fingerprint=$(openssl x509 -in "$cert" -noout -fingerprint -sha256)
fingerprint=${fingerprint#*=}
fingerprint=${fingerprint//:/}
[[ "$fingerprint" == C8C959D5DD13314320C084011D04DB576543BBE000D0CC5C778B89A202EFC689 ]]
# Non-streaming GCM buffers input: cap image size before export, after probes stop.
image_bytes=$(docker inspect --format '{{.Size}}' "$image")
[[ "$image_bytes" =~ ^[0-9]+$ ]]
(( image_bytes < 1024 * 1024 * 1024 - 16 * 1024 * 1024 ))
fifo="$out/archive-hash.fifo"
mkfifo -m 600 "$fifo"
hasher=
cleanup() {
  if [[ -n "$hasher" ]]; then
    kill "$hasher" 2>/dev/null || true
    wait "$hasher" 2>/dev/null || true
  fi
  rm -f "$fifo"
}
trap cleanup EXIT
sha256sum < "$fifo" > "$out/archive.sha256" &
hasher=$!
# GNU bash uses KiB for this bound; stat below independently rejects oversized output.
ulimit -f 2097152
docker save "$image" | tee "$fifo" | \
  openssl cms -encrypt -aes-256-gcm -binary -outform DER -recip "$cert" \
    -out "$out/runtime.cms"
wait "$hasher"
hasher=
cipher_bytes=$(stat -c '%s' "$out/runtime.cms")
(( cipher_bytes > 0 && cipher_bytes <= 2 * 1024 * 1024 * 1024 ))
sha256sum "$out/runtime.cms" > "$out/cipher.sha256"
python3 - "$image" "$out" "$fingerprint" <<'PY'
import json
import pathlib
import sys
image, directory, fingerprint = sys.argv[1:]
root = pathlib.Path(directory)
summary = {"image_id": image, "archive_sha256": (root / "archive.sha256").read_text().split()[0],
           "ciphertext_sha256": (root / "cipher.sha256").read_text().split()[0],
           "ciphertext_bytes": (root / "runtime.cms").stat().st_size,
           "maximum_image_bytes": 1024 * 1024 * 1024 - 16 * 1024 * 1024,
           "recipient_certificate_sha256": fingerprint,
           "encryption": "OpenSSL CMS AES-256-GCM; recipient private key never present on CI"}
(root / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
PY
