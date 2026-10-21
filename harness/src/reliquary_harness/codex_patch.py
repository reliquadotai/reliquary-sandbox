"""OpenAI Codex CLI's `apply_patch` engine as one self-contained Python program (`PROGRAM`).

Ported from OpenAI Codex CLI (codex-rs/apply-patch at c7234c1), Apache License 2.0, Copyright 2025 OpenAI.
Sources: `codex-rs/apply-patch/src/{parser.rs, streaming_parser.rs, seek_sequence.rs, file_update.rs,
text_file.rs, lib.rs, invocation.rs, standalone_executable.rs}`, the freeform tool handler
`codex-rs/core/src/tools/handlers/apply_patch.rs` (what the model is answered), `core/src/safety.rs` (the
empty-patch rejection), `core/src/tools/mod.rs` (`format_exec_output_for_model`) and the local filesystem
of `codex-rs/exec-server` (read/write/remove semantics and their error texts).

`PROGRAM` is run in the box as `python3 -c PROGRAM <base64 of the UTF-8 patch> [--standalone]
[--model-output]`, with cwd = the directory relative patch paths resolve against. It is stdlib-only and
parses on Python 3.6+.

Default mode = what the Codex CLI `apply_patch` tool does with one call:
  1. lenient parse (`parse_patch`); a parse error answers
     `apply_patch verification failed: invalid patch: ...` / `... invalid hunk at line N, ...`;
  2. a `*** Environment ID:` preamble answers `apply_patch environment selection is unavailable for this
     turn` (single-environment turn);
  3. verification of the whole patch before anything is written (`verify_apply_patch_args`): duplicate
     targets, Delete targets readable as UTF-8 text, every Update computable; a failure answers
     `apply_patch verification failed: <error>` and touches nothing;
  4. an empty patch answers `patch rejected: empty patch`;
  5. the patch is applied hunk by hunk (`apply_patch_with_options`): stdout gets
     `Success. Updated the following files:\\nA ..\\nM ..\\nD ..\\n` (added, then modified, then deleted, each
     in patch order, paths spelled as in the patch), or the error line of the hunk that failed (anyhow's
     outermost context, e.g. `Failed to write file /abs/path\\n`); earlier hunks stay applied, as in Codex.
  Everything the model would read is printed on stdout; the exit status is 0 on success, 1 otherwise.
  With `--model-output`, results of step 5 are wrapped as Codex wraps exec output for the model:
  `Exit code: N\\nWall time: S seconds\\nOutput:\\n<text>` (steps 1-4 are never wrapped in Codex).

`--standalone` = the standalone `apply_patch` executable instead (no verification step, parse errors on
stderr as `Invalid patch: ...` / `Invalid patch hunk on line N: ...`, apply errors on stderr, summary on
stdout); it is what the upstream fixtures in `apply-patch/tests` exercise.

Deliberate differences / limits (everything else is meant to be byte-identical):
* Only POSIX path semantics (the box is Linux): paths are joined to the cwd lexically as Codex's `PathUri`
  does (`.`/empty segments dropped, `..` pops without escaping `/`, absolute paths replace the cwd);
  Windows drive/UNC handling and percent-encoding internals are not ported.
* Codex's sandbox and approval layers are not reproduced (no `patch rejected: writing outside of the
  project`, no sandbox-denial retries): files are read and written as the unsandboxed local filesystem
  does, following symlinks.
* The wall time printed with `--model-output` is this program's own; Codex's output truncation policy is
  not applied (patch outputs are far below it).
* The unified diffs and change events Codex builds for its UI are not produced (the model never sees them).
"""

PROGRAM = r'''
import sys
if __name__ == "__main__":      # run as the program: never the importing process's path
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]

import base64
import errno
import math
import os
import re
import stat
import time

BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"
ENVIRONMENT_ID_MARKER = "*** Environment ID:"
MAX_READ_FILE_BYTES = 512 * 1024 * 1024

# Rust's char::is_whitespace (Unicode White_Space), which str.strip() does not match exactly.
RUST_WHITESPACE = (
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
    "\u200a\u2028\u2029\u202f\u205f\u3000"
)


def rs_trim(text):
    return text.strip(RUST_WHITESPACE)


def rs_trim_end(text):
    return text.rstrip(RUST_WHITESPACE)


def rs_lines(text):
    """Rust's str::lines(): split on \n, drop one \r before each \n, no empty trailing line."""
    if not text:
        return []
    parts = text.split("\n")
    lines = []
    for part in parts[:-1]:
        if part.endswith("\r"):
            part = part[:-1]
        lines.append(part)
    if parts[-1] != "":
        lines.append(parts[-1])
    return lines


def rust_os_name():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


class PatchParseError(Exception):
    """parser::ParseError: InvalidPatchError (line_number None) or InvalidHunkError."""

    def __init__(self, message, line_number=None):
        Exception.__init__(self, message)
        self.message = message
        self.line_number = line_number

    def __str__(self):
        if self.line_number is None:
            return "invalid patch: " + self.message
        return "invalid hunk at line %d, %s" % (self.line_number, self.message)

    def standalone_message(self):
        if self.line_number is None:
            return "Invalid patch: " + self.message
        return "Invalid patch hunk on line %d: %s" % (self.line_number, self.message)


class ApplyError(Exception):
    """An error whose Display is `message` (ApplyPatchError / anyhow context)."""

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __str__(self):
        return self.message


class CustomIoError(Exception):
    """An io::Error built from a message (io::Error::new(kind, msg)): Display is the message."""

    def __init__(self, message, kind="Other"):
        Exception.__init__(self, message)
        self.message = message
        self.kind = kind

    def __str__(self):
        return self.message


def io_error_text(error):
    """Display of a Rust io::Error."""
    if isinstance(error, OSError) and error.errno:
        return "%s (os error %d)" % (os.strerror(error.errno), error.errno)
    return str(error)


class _Value(object):
    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        fields = ", ".join("%s=%r" % (k, v) for k, v in sorted(self.__dict__.items()))
        return "%s(%s)" % (type(self).__name__, fields)


class AddFile(_Value):
    def __init__(self, path, contents=""):
        self.path = path
        self.contents = contents

    def affected_path(self):
        return self.path


class DeleteFile(_Value):
    def __init__(self, path):
        self.path = path

    def affected_path(self):
        return self.path


class UpdateFile(_Value):
    def __init__(self, path, move_path=None, chunks=None):
        self.path = path
        self.move_path = move_path
        self.chunks = [] if chunks is None else chunks

    def affected_path(self):
        return self.move_path if self.move_path is not None else self.path


class UpdateFileChunk(_Value):
    def __init__(self, change_context=None, old_lines=None, new_lines=None, context_line_indices=None,
                 is_end_of_file=False):
        self.change_context = change_context
        self.old_lines = [] if old_lines is None else old_lines
        self.new_lines = [] if new_lines is None else new_lines
        self.context_line_indices = [] if context_line_indices is None else context_line_indices
        self.is_end_of_file = is_end_of_file

    def push_context_line(self, line):
        self.context_line_indices.append((len(self.old_lines), len(self.new_lines)))
        self.old_lines.append(line)
        self.new_lines.append(line)

    def is_empty(self):
        return not self.old_lines and not self.new_lines


class ApplyPatchArgs(_Value):
    def __init__(self, hunks, patch, environment_id=None):
        self.hunks = hunks
        self.patch = patch
        self.environment_id = environment_id


# ---------------------------------------------------------------- parser (streaming_parser.rs)

NOT_STARTED, STARTED_PATCH, ADD_FILE, DELETE_FILE, UPDATE_FILE, ENDED_PATCH = range(6)

INVALID_HEADER = ("'%s' is not a valid hunk header. Valid hunk headers: '*** Add File: {path}', "
                  "'*** Delete File: {path}', '*** Update File: {path}'")
UNEXPECTED_LINE = ("Unexpected line found in update hunk: '%s'. Every line should start with ' ' "
                   "(context line), '+' (added line), or '-' (removed line)")
EXPECTED_CONTEXT = "Expected update hunk to start with a @@ context marker, got: '%s'"


class StreamingPatchParser(object):
    def __init__(self):
        self.line_buffer = ""
        self.mode = NOT_STARTED
        self.hunk_line_number = 0
        self.hunks = []
        self.environment_id = None
        self.line_number = 0

    def _ensure_update_hunk_is_not_empty(self, line):
        if self.hunks and isinstance(self.hunks[-1], UpdateFile):
            hunk = self.hunks[-1]
            if not hunk.chunks and self.mode == UPDATE_FILE:
                raise PatchParseError("Update file hunk for path '%s' is empty" % hunk.path,
                                      self.hunk_line_number)
            if hunk.chunks and hunk.chunks[-1].is_empty():
                if line == END_PATCH_MARKER:
                    raise PatchParseError("Update hunk does not contain any lines", self.line_number)
                raise PatchParseError(UNEXPECTED_LINE % line, self.line_number)

    def _handle_hunk_headers_and_end_patch(self, trimmed):
        if self.mode == STARTED_PATCH and trimmed.startswith(ENVIRONMENT_ID_MARKER):
            if self.environment_id is not None:
                raise PatchParseError("apply_patch environment_id cannot be specified more than once")
            environment_id = rs_trim(trimmed[len(ENVIRONMENT_ID_MARKER):])
            if not environment_id:
                raise PatchParseError("apply_patch environment_id cannot be empty")
            self.environment_id = environment_id
            return True
        if trimmed == END_PATCH_MARKER:
            self._ensure_update_hunk_is_not_empty(trimmed)
            self.mode = ENDED_PATCH
            return True
        if trimmed.startswith(ADD_FILE_MARKER):
            self._ensure_update_hunk_is_not_empty(trimmed)
            self.hunks.append(AddFile(trimmed[len(ADD_FILE_MARKER):], ""))
            self.mode = ADD_FILE
            return True
        if trimmed.startswith(DELETE_FILE_MARKER):
            self._ensure_update_hunk_is_not_empty(trimmed)
            self.hunks.append(DeleteFile(trimmed[len(DELETE_FILE_MARKER):]))
            self.mode = DELETE_FILE
            return True
        if trimmed.startswith(UPDATE_FILE_MARKER):
            self._ensure_update_hunk_is_not_empty(trimmed)
            self.hunks.append(UpdateFile(trimmed[len(UPDATE_FILE_MARKER):]))
            self.mode = UPDATE_FILE
            self.hunk_line_number = self.line_number
            return True
        return False

    def push_delta(self, delta):
        parts = delta.split("\n")
        for part in parts[:-1]:
            line = self.line_buffer + part
            self.line_buffer = ""
            if line.endswith("\r"):
                line = line[:-1]
            self.line_number += 1
            self._process_line(line)
        self.line_buffer += parts[-1]
        return list(self.hunks)

    def finish(self):
        if self.line_buffer:
            line = self.line_buffer
            self.line_buffer = ""
            self.line_number += 1
            if rs_trim(line) == END_PATCH_MARKER:
                self._ensure_update_hunk_is_not_empty(rs_trim(line))
                self.mode = ENDED_PATCH
            else:
                self._process_line(line)
        if self.mode != ENDED_PATCH:
            raise PatchParseError("The last line of the patch must be '*** End Patch'")
        return list(self.hunks)

    def _process_line(self, line):
        trimmed = rs_trim(line)
        mode = self.mode
        if mode == NOT_STARTED:
            if trimmed == BEGIN_PATCH_MARKER:
                self.mode = STARTED_PATCH
                return
            raise PatchParseError("The first line of the patch must be '*** Begin Patch'")
        if mode == STARTED_PATCH or mode == DELETE_FILE:
            if self._handle_hunk_headers_and_end_patch(trimmed):
                return
            raise PatchParseError(INVALID_HEADER % trimmed, self.line_number)
        if mode == ADD_FILE:
            if self._handle_hunk_headers_and_end_patch(trimmed):
                return
            if line.startswith("+") and self.hunks and isinstance(self.hunks[-1], AddFile):
                self.hunks[-1].contents += line[1:] + "\n"
                return
            raise PatchParseError(INVALID_HEADER % trimmed, self.line_number)
        if mode == UPDATE_FILE:
            self._process_update_line(line)
            return
        # ENDED_PATCH
        if trimmed == "":
            return
        raise PatchParseError("The last line of the patch must be '*** End Patch'")

    def _process_update_line(self, line):
        update_line = rs_trim_end(line)
        if self._handle_hunk_headers_and_end_patch(update_line):
            return
        hunk = self.hunks[-1] if self.hunks else None
        if isinstance(hunk, UpdateFile):
            chunks = hunk.chunks
            last = chunks[-1] if chunks else None
            is_context_marker = (update_line == EMPTY_CHANGE_CONTEXT_MARKER
                                 or update_line.startswith(CHANGE_CONTEXT_MARKER))
            if last is not None and last.is_end_of_file:
                if update_line == "":
                    return
                if not is_context_marker:
                    raise PatchParseError(EXPECTED_CONTEXT % line, self.line_number)
            if not chunks and hunk.move_path is None and update_line.startswith(MOVE_TO_MARKER):
                hunk.move_path = update_line[len(MOVE_TO_MARKER):]
                return
            if is_context_marker and last is not None and last.is_empty():
                raise PatchParseError(UNEXPECTED_LINE % line, self.line_number)
            if update_line == EMPTY_CHANGE_CONTEXT_MARKER:
                chunks.append(UpdateFileChunk())
                return
            if update_line.startswith(CHANGE_CONTEXT_MARKER):
                chunks.append(UpdateFileChunk(change_context=update_line[len(CHANGE_CONTEXT_MARKER):]))
                return
            if update_line == EOF_MARKER:
                if last is not None and last.is_empty():
                    raise PatchParseError("Update hunk does not contain any lines", self.line_number)
                if last is not None:
                    last.is_end_of_file = True
                return
            if line == "":
                if not chunks:
                    chunks.append(UpdateFileChunk())
                chunks[-1].push_context_line("")
                return
            if line.startswith(" "):
                if not chunks:
                    chunks.append(UpdateFileChunk())
                chunks[-1].push_context_line(line[1:])
                return
            if line.startswith("+"):
                if not chunks:
                    chunks.append(UpdateFileChunk())
                chunks[-1].new_lines.append(line[1:])
                return
            if line.startswith("-"):
                if not chunks:
                    chunks.append(UpdateFileChunk())
                chunks[-1].old_lines.append(line[1:])
                return
            if last is not None and not last.is_empty():
                raise PatchParseError(EXPECTED_CONTEXT % line, self.line_number)
        raise PatchParseError(UNEXPECTED_LINE % line, self.line_number)


# ---------------------------------------------------------------- parser (parser.rs)

def _check_patch_boundaries_strict(lines):
    first = rs_trim(lines[0]) if lines else None
    last = rs_trim(lines[-1]) if lines else None
    if first is not None and last is not None and first == BEGIN_PATCH_MARKER and last == END_PATCH_MARKER:
        return lines
    if first is not None and first != BEGIN_PATCH_MARKER:
        raise PatchParseError("The first line of the patch must be '*** Begin Patch'")
    raise PatchParseError("The last line of the patch must be '*** End Patch'")


def _check_patch_boundaries_lenient(lines):
    try:
        return _check_patch_boundaries_strict(lines)
    except PatchParseError as original_error:
        if (len(lines) >= 4 and lines[0] in ("<<EOF", "<<'EOF'", '<<"EOF"')
                and lines[-1].endswith("EOF")):
            return _check_patch_boundaries_strict(lines[1:-1])
        raise original_error


def parse_patch_text(patch, lenient=True):
    lines = rs_lines(rs_trim(patch))
    if lenient:
        patch_lines = _check_patch_boundaries_lenient(lines)
    else:
        patch_lines = _check_patch_boundaries_strict(lines)
    joined = "\n".join(patch_lines)
    parser = StreamingPatchParser()
    parser.push_delta(joined)
    hunks = parser.finish()
    return ApplyPatchArgs(hunks, joined, parser.environment_id)


def parse_patch(patch):
    """parser::parse_patch (always lenient in Codex): ApplyPatchArgs, or raises PatchParseError."""
    return parse_patch_text(patch, lenient=True)


# ---------------------------------------------------------------- seek_sequence.rs

_PUNCTUATION = {}
for _c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
    _PUNCTUATION[ord(_c)] = "-"
for _c in "\u2018\u2019\u201a\u201b":
    _PUNCTUATION[ord(_c)] = "'"
for _c in "\u201c\u201d\u201e\u201f":
    _PUNCTUATION[ord(_c)] = '"'
for _c in "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000":
    _PUNCTUATION[ord(_c)] = " "


def _normalise(text):
    return rs_trim(text).translate(_PUNCTUATION)


def seek_sequence(lines, pattern, start, eof):
    """Index of the first match of `pattern` in `lines` at or after `start` (or None): exact, then
    ignoring trailing whitespace, then ignoring surrounding whitespace, then also normalising Unicode
    punctuation. With `eof`, the search starts at the end of the file."""
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None
    if eof and len(lines) >= len(pattern):
        search_start = max(len(lines) - len(pattern), start)
    else:
        search_start = start
    stop = len(lines) - len(pattern) + 1
    count = len(pattern)
    for i in range(search_start, stop):
        if lines[i:i + count] == pattern:
            return i
    for transform in (rs_trim_end, rs_trim, _normalise):
        wanted = [transform(p) for p in pattern]
        for i in range(search_start, stop):
            ok = True
            for p_idx in range(count):
                if transform(lines[i + p_idx]) != wanted[p_idx]:
                    ok = False
                    break
            if ok:
                return i
    return None


# ---------------------------------------------------------------- text_file.rs

_LINE_ENDING = re.compile("\r\n|\r|\n")


class SourceFile(object):
    def __init__(self, contents):
        self.lines = []  # [text, ending or None]
        preferred = None
        line_start = 0
        for match in _LINE_ENDING.finditer(contents):
            ending = match.group(0)
            if preferred is None:
                preferred = ending
            self.lines.append([contents[line_start:match.start()], ending])
            line_start = match.end()
        if line_start < len(contents):
            self.lines.append([contents[line_start:], None])
        self.preferred_ending = preferred if preferred is not None else "\n"

    def line_texts(self):
        return [line[0] for line in self.lines]

    def apply_replacements(self, replacements):
        # A port of the Rust iterator walk: `position` is how far the source iterator has been consumed.
        source = self.lines
        new_lines = []
        source_index = 0
        position = 0
        for start_idx, old_len, new_segment in replacements:
            keep = start_idx - source_index
            if keep < 0:  # unreachable for sorted, non-overlapping replacements
                keep = len(source)
            kept = source[position:position + keep]
            new_lines.extend(kept)
            position = min(position + len(kept) + old_len, len(source))
            for text in new_segment:
                new_lines.append([text, self.preferred_ending])
            source_index = start_idx + old_len
        new_lines.extend(source[position:])
        for line in new_lines:
            if line[1] is None:
                line[1] = self.preferred_ending
        self.lines = new_lines

    def into_contents(self):
        return "".join(text + (ending or "") for text, ending in self.lines)


# ---------------------------------------------------------------- file_update.rs

def compute_replacements(original_lines, path, chunks):
    replacements = []
    line_index = 0
    for chunk in chunks:
        if chunk.change_context is not None:
            idx = seek_sequence(original_lines, [chunk.change_context], line_index, False)
            if idx is None:
                raise ApplyError("Failed to find context '%s' in %s" % (chunk.change_context, path))
            line_index = idx + 1

        if not chunk.old_lines:
            replacements.append((len(original_lines), 0, list(chunk.new_lines)))
            continue

        pattern = chunk.old_lines
        new_slice = chunk.new_lines
        found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)
        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)

        if found is None:
            raise ApplyError("Failed to find expected lines in %s:\n%s" % (path, "\n".join(chunk.old_lines)))

        start_idx = found
        old_start = 0
        new_start = 0
        for old_context, new_context in chunk.context_line_indices:
            if old_context >= len(pattern) or new_context >= len(new_slice):
                break
            if old_start != old_context or new_start != new_context:
                replacements.append((start_idx + old_start, old_context - old_start,
                                     list(new_slice[new_start:new_context])))
            old_start = old_context + 1
            new_start = new_context + 1
        if old_start != len(pattern) or new_start != len(new_slice):
            replacements.append((start_idx + old_start, len(pattern) - old_start,
                                 list(new_slice[new_start:])))
        line_index = start_idx + len(pattern)

    replacements.sort(key=lambda replacement: replacement[0])
    return replacements


def derive_new_contents(path, chunks):
    """(original, new) contents of the file at absolute `path` once `chunks` are applied."""
    try:
        original = read_file_text(path)
    except (OSError, CustomIoError) as error:
        raise ApplyError("Failed to read file to update %s: %s" % (path, io_error_text(error)))
    source_file = SourceFile(original)
    replacements = compute_replacements(source_file.line_texts(), path, chunks)
    source_file.apply_replacements(replacements)
    return original, source_file.into_contents()


# ---------------------------------------------------------------- filesystem (exec-server LOCAL_FS)

def resolve_path(cwd, path):
    """PathUri::join on a POSIX cwd, rendered back as a native path string."""
    if "\0" in path:
        raise ApplyError("'%s' is invalid on '%s'" % (path, rust_os_name()))
    if path == "":
        return cwd
    if path.startswith("/"):
        segments = []
        trailing = False
        for segment in path[1:].split("/"):
            if segment == "":
                trailing = True
            elif segment == ".":
                trailing = False
            elif segment == "..":
                trailing = False
                if segments:
                    segments.pop()
            else:
                segments.append(segment)
                trailing = False
        return "/" + "/".join(segments) + ("/" if trailing and segments else "")
    segments = [s for s in cwd.split("/") if s not in ("", ".")]
    for component in path.split("/"):
        if component in ("", "."):
            continue
        if component == "..":
            if segments:
                segments.pop()
            continue
        segments.append(component)
    return "/" + "/".join(segments)


def _parent(path):
    stripped = path.rstrip("/")
    if stripped == "":
        return None
    parent = stripped.rsplit("/", 1)[0]
    return parent if parent else "/"


def read_file_bytes(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise CustomIoError("path `%s` is not a file" % path, "InvalidInput")
        too_large = "file is too large to read: limit is %d bytes" % MAX_READ_FILE_BYTES
        if info.st_size > MAX_READ_FILE_BYTES:
            raise CustomIoError(too_large, "InvalidInput")
        pieces = []
        total = 0
        while True:
            piece = os.read(fd, 1 << 20)
            if not piece:
                break
            pieces.append(piece)
            total += len(piece)
            if total > MAX_READ_FILE_BYTES:
                raise CustomIoError(too_large, "InvalidInput")
        return b"".join(pieces)
    finally:
        os.close(fd)


def read_file_text(path):
    data = read_file_bytes(path)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data":
            raise CustomIoError("incomplete utf-8 byte sequence from index %d" % error.start, "InvalidData")
        raise CustomIoError("invalid utf-8 sequence of %d bytes from index %d"
                            % (error.end - error.start, error.start), "InvalidData")


def write_file(path, data):
    with open(path, "wb") as handle:
        handle.write(data)


def write_file_with_missing_parent_retry(path, data):
    try:
        write_file(path, data)
        return
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise ApplyError("Failed to write file %s" % path)
    parent = _parent(path)
    if parent is not None:
        try:
            if not os.path.isdir(parent):
                os.makedirs(parent)
        except OSError:
            if not os.path.isdir(parent):
                raise ApplyError("Failed to create parent directories for %s" % path)
    try:
        write_file(path, data)
    except OSError:
        raise ApplyError("Failed to write file %s" % path)


def ensure_not_directory(path):
    """Raises OSError / CustomIoError like exec-server's get_metadata + is_directory check."""
    info = os.stat(path)
    if stat.S_ISDIR(info.st_mode):
        raise CustomIoError("path is a directory", "InvalidInput")


def remove_path(path):
    info = os.lstat(path)
    if stat.S_ISDIR(info.st_mode):
        os.rmdir(path)
    else:
        os.unlink(path)


# ---------------------------------------------------------------- lib.rs / invocation.rs

def verify_hunks(hunks, cwd):
    """invocation::verify_apply_patch_args: everything is checked before anything is written."""
    seen = set()
    for hunk in hunks:
        path = resolve_path(cwd, hunk.path)
        if path in seen:
            raise PatchParseError("multiple operations target %s" % path)
        seen.add(path)
        if isinstance(hunk, DeleteFile):
            try:
                read_file_text(path)
            except (OSError, CustomIoError) as error:
                raise ApplyError("Failed to read %s: %s" % (path, io_error_text(error)))
        elif isinstance(hunk, UpdateFile):
            derive_new_contents(path, hunk.chunks)
            if hunk.move_path is not None:
                resolve_path(cwd, hunk.move_path)


def apply_hunks_to_files(hunks, cwd):
    """lib::apply_hunks_to_files: (added, modified, deleted) or ApplyError at the failing hunk."""
    if not hunks:
        raise ApplyError("No files were modified.")
    added, modified, deleted = [], [], []
    for hunk in hunks:
        affected = hunk.affected_path()
        path = resolve_path(cwd, hunk.path)
        if isinstance(hunk, AddFile):
            write_file_with_missing_parent_retry(path, hunk.contents.encode("utf-8"))
            added.append(affected)
        elif isinstance(hunk, DeleteFile):
            try:
                ensure_not_directory(path)
                remove_path(path)
            except (OSError, CustomIoError):
                raise ApplyError("Failed to delete file %s" % path)
            deleted.append(affected)
        else:
            _, new_contents = derive_new_contents(path, hunk.chunks)
            data = new_contents.encode("utf-8")
            if hunk.move_path is not None:
                destination = resolve_path(cwd, hunk.move_path)
                write_file_with_missing_parent_retry(destination, data)
                try:
                    ensure_not_directory(path)
                    remove_path(path)
                except (OSError, CustomIoError):
                    raise ApplyError("Failed to remove original %s" % path)
            else:
                try:
                    write_file(path, data)
                except OSError:
                    raise ApplyError("Failed to write file %s" % path)
            modified.append(affected)
    return added, modified, deleted


def summary(added, modified, deleted):
    out = ["Success. Updated the following files:\n"]
    out.extend("A %s\n" % p for p in added)
    out.extend("M %s\n" % p for p in modified)
    out.extend("D %s\n" % p for p in deleted)
    return "".join(out)


def apply_patch_standalone(patch, cwd):
    """The standalone `apply_patch` executable: (exit_code, stdout, stderr)."""
    try:
        args = parse_patch(patch)
    except PatchParseError as error:
        return 1, "", error.standalone_message() + "\n"
    return apply_hunks(args.hunks, cwd)


def apply_hunks(hunks, cwd):
    try:
        added, modified, deleted = apply_hunks_to_files(hunks, cwd)
    except ApplyError as error:
        return 1, "", error.message + "\n"
    return 0, summary(added, modified, deleted), ""


def format_wall_time(seconds):
    rounded = math.floor(seconds * 10.0 + 0.5) / 10.0
    if rounded == int(rounded):
        return "%d" % int(rounded)
    return repr(rounded)


def format_exec_output_for_model(exit_code, output, seconds):
    return "Exit code: %d\nWall time: %s seconds\nOutput:\n%s" % (exit_code, format_wall_time(seconds), output)


def apply_patch_tool(patch, cwd, model_output=False):
    """The Codex CLI `apply_patch` tool on one call: (exit_code, text the model reads)."""
    try:
        args = parse_patch(patch)
    except PatchParseError as error:
        return 1, "apply_patch verification failed: %s" % error
    if args.environment_id is not None:
        return 1, "apply_patch environment selection is unavailable for this turn"
    try:
        verify_hunks(args.hunks, cwd)
    except (PatchParseError, ApplyError) as error:
        return 1, "apply_patch verification failed: %s" % error
    if not args.hunks:
        return 1, "patch rejected: empty patch"
    started = time.time()
    code, out, err = apply_hunks(args.hunks, cwd)
    text = out + err
    if model_output:
        text = format_exec_output_for_model(code, text, time.time() - started)
    return code, text


def _write(stream, text):
    data = text.encode("utf-8")
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(data)
        buffer.flush()
    else:
        stream.write(text)
        stream.flush()


def main(argv):
    flags = [a for a in argv if a in ("--standalone", "--model-output")]
    positional = [a for a in argv if a not in ("--standalone", "--model-output")]
    if len(positional) != 1:
        _write(sys.stderr, "Usage: python3 -c PROGRAM <base64 PATCH> [--standalone] [--model-output]\n")
        return 2
    try:
        patch = base64.b64decode(positional[0].encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeError):
        _write(sys.stderr, "Error: apply_patch requires a UTF-8 PATCH argument.\n")
        return 1
    cwd = os.getcwd()
    if "--standalone" in flags:
        code, out, err = apply_patch_standalone(patch, cwd)
        _write(sys.stdout, out)
        _write(sys.stderr, err)
        return code
    code, text = apply_patch_tool(patch, cwd, model_output="--model-output" in flags)
    _write(sys.stdout, text)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

_namespace = {"__name__": "reliquary_harness.codex_patch_program"}
exec(compile(PROGRAM, "<codex_patch PROGRAM>", "exec"), _namespace)

PatchParseError = _namespace["PatchParseError"]
ApplyError = _namespace["ApplyError"]
AddFile = _namespace["AddFile"]
DeleteFile = _namespace["DeleteFile"]
UpdateFile = _namespace["UpdateFile"]
UpdateFileChunk = _namespace["UpdateFileChunk"]
ApplyPatchArgs = _namespace["ApplyPatchArgs"]
StreamingPatchParser = _namespace["StreamingPatchParser"]
parse_patch = _namespace["parse_patch"]
parse_patch_text = _namespace["parse_patch_text"]
seek_sequence = _namespace["seek_sequence"]
resolve_path = _namespace["resolve_path"]
compute_replacements = _namespace["compute_replacements"]
format_exec_output_for_model = _namespace["format_exec_output_for_model"]


def encode_patch(patch):
    """The program's argument for `patch`: its UTF-8 bytes in base64."""
    import base64

    return base64.b64encode(patch.encode("utf-8")).decode("ascii")


def apply_patch_to_dir(patch, cwd, model_output=False):
    """In-process twin of running PROGRAM in `cwd` (Codex tool mode): (exit_code, stdout text)."""
    import os

    return _namespace["apply_patch_tool"](patch, os.path.abspath(cwd), model_output)


def apply_patch_standalone_to_dir(patch, cwd):
    """In-process twin of `PROGRAM --standalone` in `cwd`: (exit_code, stdout, stderr)."""
    import os

    return _namespace["apply_patch_standalone"](patch, os.path.abspath(cwd))
