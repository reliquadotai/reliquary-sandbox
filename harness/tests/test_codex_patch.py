"""The Codex `apply_patch` port (`reliquary_harness.codex_patch`): its PROGRAM, run as the box runs it, against
the upstream Rust tests (codex-rs/apply-patch at c7234c1: parser.rs, streaming_parser.rs, seek_sequence.rs,
file_update_tests.rs, lib.rs, tests/suite/tool.rs, tests/fixtures/scenarios) and the Codex CLI tool's
answers to the model (core/src/tools/handlers/apply_patch.rs, core/tests/suite/apply_patch_cli.rs)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest
from reliquary_harness.codex_patch import (
    PROGRAM,
    AddFile,
    DeleteFile,
    PatchParseError,
    StreamingPatchParser,
    UpdateFile,
    UpdateFileChunk,
    apply_patch_standalone_to_dir,
    apply_patch_to_dir,
    encode_patch,
    parse_patch,
    parse_patch_text,
    resolve_path,
    seek_sequence,
)

HEADERS = ("Valid hunk headers: '*** Add File: {path}', '*** Delete File: {path}', "
           "'*** Update File: {path}'")
UNEXPECTED = ("Every line should start with ' ' (context line), '+' (added line), or '-' (removed line)")
VERIFY = "apply_patch verification failed: "


def wrap(body: str) -> str:
    return f"*** Begin Patch\n{body}\n*** End Patch"


def run(cwd, patch: str, *flags: str):
    """PROGRAM in `cwd` as the box runs it: (exit code, stdout, stderr)."""
    done = subprocess.run([sys.executable, "-c", PROGRAM, encode_patch(patch), *flags], cwd=str(cwd),
                          capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode("utf-8"), done.stderr.decode("utf-8")


def tool(cwd, patch: str):
    """The Codex tool's answer: (exit code, text); nothing goes to stderr."""
    code, out, err = run(cwd, patch)
    assert err == ""
    return code, out


def real(path) -> str:
    return os.path.realpath(str(path))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)


def tree(root) -> dict:
    files = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            files[os.path.relpath(full, root)] = open(full, "rb").read()
    return files


# ------------------------------------------------------------------ the program itself

def test_program_compiles_and_uses_no_python38_syntax():
    compile(PROGRAM, "x", "exec")
    assert ":=" not in PROGRAM
    if sys.version_info >= (3, 8):
        ast.parse(PROGRAM, feature_version=(3, 6))
    for later in ("removeprefix", "removesuffix", "dataclass", "from __future__"):
        assert later not in PROGRAM, later


def test_program_ignores_modules_planted_in_the_working_directory(tmp_path):
    (tmp_path / "base64.py").write_text("raise SystemExit('planted')\n")
    (tmp_path / "stat.py").write_text("raise SystemExit('planted')\n")
    code, out = tool(tmp_path, wrap("*** Add File: a.txt\n+x"))
    assert (code, out) == (0, "Success. Updated the following files:\nA a.txt\n")


def test_bad_arguments(tmp_path):
    done = subprocess.run([sys.executable, "-c", PROGRAM], cwd=str(tmp_path), capture_output=True)
    assert done.returncode == 2
    done = subprocess.run([sys.executable, "-c", PROGRAM, "/w=="],
                          cwd=str(tmp_path), capture_output=True)
    assert done.returncode == 1
    assert done.stderr == b"Error: apply_patch requires a UTF-8 PATCH argument.\n"


# ------------------------------------------------------------------ parser.rs

def test_parse_patch():
    with pytest.raises(PatchParseError) as error:
        parse_patch_text("bad", lenient=False)
    assert str(error.value) == "invalid patch: The first line of the patch must be '*** Begin Patch'"
    with pytest.raises(PatchParseError) as error:
        parse_patch_text("*** Begin Patch\nbad", lenient=False)
    assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"

    assert parse_patch_text("*** Begin Patch \n*** Add File: foo\n+hi\n *** End Patch", lenient=False).hunks \
        == [AddFile("foo", "hi\n")]
    with pytest.raises(PatchParseError) as error:
        parse_patch_text("*** Begin Patch\n*** Update File: test.py\n*** End Patch", lenient=False)
    assert (error.value.message, error.value.line_number) == ("Update file hunk for path 'test.py' is empty", 2)
    assert str(error.value) == "invalid hunk at line 2, Update file hunk for path 'test.py' is empty"
    assert parse_patch_text("*** Begin Patch\n*** End Patch", lenient=False).hunks == []

    assert parse_patch_text(
        "*** Begin Patch\n*** Add File: path/add.py\n+abc\n+def\n*** Delete File: path/delete.py\n"
        "*** Update File: path/update.py\n*** Move to: path/update2.py\n@@ def f():\n-    pass\n"
        "+    return 123\n*** End Patch", lenient=False).hunks == [
        AddFile("path/add.py", "abc\ndef\n"),
        DeleteFile("path/delete.py"),
        UpdateFile("path/update.py", "path/update2.py", [UpdateFileChunk(
            change_context="def f():", old_lines=["    pass"], new_lines=["    return 123"])]),
    ]
    assert parse_patch_text("*** Begin Patch\n*** Update File: file.py\n@@\n+line\n*** Add File: other.py\n"
                            "+content\n*** End Patch", lenient=False).hunks == [
        UpdateFile("file.py", None, [UpdateFileChunk(new_lines=["line"])]),
        AddFile("other.py", "content\n"),
    ]
    assert parse_patch_text("*** Begin Patch\n*** Update File: file2.py\n import foo\n+bar\n*** End Patch",
                            lenient=False).hunks == [
        UpdateFile("file2.py", None, [UpdateFileChunk(old_lines=["import foo"], new_lines=["import foo", "bar"],
                                                      context_line_indices=[(0, 0)])]),
    ]


def test_parse_patch_preserves_end_of_file_marker():
    patch = "*** Begin Patch\n*** Update File: file.txt\n@@\n+quux\n*** End of File\n\n*** End Patch"
    args = parse_patch(patch)
    assert args.hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(new_lines=["quux"], is_end_of_file=True)])]
    assert args.patch == patch
    assert args.environment_id is None


def test_parse_patch_accepts_relative_and_absolute_hunk_paths():
    patch = ("*** Begin Patch\n*** Add File: relative-add.py\n+content\n*** Delete File: /abs/absolute-delete.py\n"
             "*** Update File: /abs/absolute-update.py\n@@\n-old\n+new\n*** End Patch")
    assert parse_patch_text(patch, lenient=False).hunks == [
        AddFile("relative-add.py", "content\n"),
        DeleteFile("/abs/absolute-delete.py"),
        UpdateFile("/abs/absolute-update.py", None, [UpdateFileChunk(old_lines=["old"], new_lines=["new"])]),
    ]


def test_hunk_paths_resolve_like_path_uri_join():
    assert resolve_path("/work", "relative-add.py") == "/work/relative-add.py"
    assert resolve_path("/work", "/other/absolute.py") == "/other/absolute.py"
    assert resolve_path("/work", "./a//b/./c.txt") == "/work/a/b/c.txt"
    assert resolve_path("/work/sub", "../x.txt") == "/work/x.txt"
    assert resolve_path("/work", "../../../x.txt") == "/x.txt"
    assert resolve_path("/work", "dir/") == "/work/dir"
    assert resolve_path("/work", "/a/b/../c") == "/a/c"
    assert resolve_path("/work", "") == "/work"


def test_parse_patch_lenient():
    patch_text = "*** Begin Patch\n*** Update File: file2.py\n import foo\n+bar\n*** End Patch"
    expected = [UpdateFile("file2.py", None, [UpdateFileChunk(
        old_lines=["import foo"], new_lines=["import foo", "bar"], context_line_indices=[(0, 0)])])]
    first_line = "invalid patch: The first line of the patch must be '*** Begin Patch'"
    for opener in ("<<EOF", "<<'EOF'", '<<"EOF"'):
        heredoc = f"{opener}\n{patch_text}\nEOF\n"
        with pytest.raises(PatchParseError) as error:
            parse_patch_text(heredoc, lenient=False)
        assert str(error.value) == first_line
        args = parse_patch(heredoc)
        assert (args.hunks, args.patch) == (expected, patch_text)

    with pytest.raises(PatchParseError) as error:
        parse_patch(f"<<\"EOF'\n{patch_text}\nEOF\n")
    assert str(error.value) == first_line
    with pytest.raises(PatchParseError) as error:
        parse_patch("<<EOF\n*** Begin Patch\n*** Update File: file2.py\nEOF\n")
    assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"


def test_parse_patch_environment_id_preamble():
    args = parse_patch("*** Begin Patch\n*** Environment ID: remote\n*** Add File: hello.txt\n+hello\n*** End Patch")
    assert (args.hunks, args.environment_id) == ([AddFile("hello.txt", "hello\n")], "remote")
    with pytest.raises(PatchParseError) as error:
        parse_patch("*** Begin Patch\n*** Environment ID:   \n*** Add File: hello.txt\n+hello\n*** End Patch")
    assert str(error.value) == "invalid patch: apply_patch environment_id cannot be empty"
    with pytest.raises(PatchParseError) as error:
        parse_patch("*** Begin Patch\n*** Environment ID: first\n*** Environment ID: second\n*** End Patch")
    assert str(error.value) == "invalid patch: apply_patch environment_id cannot be specified more than once"


def test_empty_and_whitespace_only_patches():
    for patch in ("", "   \n\t"):
        with pytest.raises(PatchParseError) as error:
            parse_patch(patch)
        assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"
    with pytest.raises(PatchParseError) as error:
        parse_patch("*** Begin Patch")
    assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"


# ------------------------------------------------------------------ streaming_parser.rs

def _stream(text):
    parser = StreamingPatchParser()
    return parser, parser.push_delta(text)


def test_streaming_parser_streams_complete_lines_before_end_patch():
    parser, hunks = _stream("*** Begin Patch\n*** Add File: src/hello.txt\n+hello\n+wor")
    assert hunks == [AddFile("src/hello.txt", "hello\n")]
    assert parser.push_delta("ld\n") == [AddFile("src/hello.txt", "hello\nworld\n")]
    _, hunks = _stream("*** Begin Patch\n*** Update File: src/old.rs\n*** Move to: src/new.rs\n@@\n-old\n+new\n")
    assert hunks == [UpdateFile("src/old.rs", "src/new.rs", [UpdateFileChunk(old_lines=["old"], new_lines=["new"])])]
    parser, hunks = _stream("*** Begin Patch\n*** Delete File: gone.txt")
    assert hunks == []
    assert parser.push_delta("\n") == [DeleteFile("gone.txt")]


def test_streaming_parser_large_patch_split_by_character():
    patch = (
        "*** Begin Patch\n*** Add File: docs/release-notes.md\n+# Release notes\n+\n+## CLI\n"
        "*** Update File: src/config.rs\n@@ impl Config\n-    pub apply_patch_progress: bool,\n"
        "+    pub stream_apply_patch_progress: bool,\n     pub include_diagnostics: bool,\n"
        "@@ fn default_progress_interval()\n-    Duration::from_millis(500)\n+    Duration::from_millis(250)\n"
        "*** Delete File: src/legacy_patch_progress.rs\n*** Update File: crates/cli/src/main.rs\n"
        "*** Move to: crates/cli/src/bin/codex.rs\n@@ fn run()\n-    let args = Args::parse();\n"
        "+    let cli = Cli::parse();\n*** Add File: tests/fixtures/apply_patch_progress.json\n+{\n+}\n"
        "*** Update File: README.md\n@@ Development workflow\n Build the Rust workspace.\n+More.\n"
        "*** Delete File: docs/old-apply-patch-progress.md\n*** End Patch")
    parser = StreamingPatchParser()
    counts = []
    for ch in patch:
        hunks = parser.push_delta(ch)
        if hunks and (not counts or len(hunks) > counts[-1]):
            counts.append(len(hunks))
    hunks = parser.finish()
    assert counts == [1, 2, 3, 4, 5, 6, 7]
    kinds = [type(h).__name__ + ("+move" if getattr(h, "move_path", None) else "") for h in hunks]
    assert kinds == ["AddFile", "UpdateFile", "DeleteFile", "UpdateFile+move", "AddFile", "UpdateFile",
                     "DeleteFile"]
    assert hunks == parse_patch(patch).hunks


def test_streaming_parser_keeps_indented_update_markers_as_context_lines():
    _, hunks = _stream("*** Begin Patch\n*** Update File: a.txt\n@@\n-old a\n+new a\n *** Update File: b.txt\n"
                       "@@\n-old b\n+new b\n*** End Patch\n")
    assert hunks == [UpdateFile("a.txt", None, [
        UpdateFileChunk(old_lines=["old a", "*** Update File: b.txt"], new_lines=["new a", "*** Update File: b.txt"],
                        context_line_indices=[(1, 1)]),
        UpdateFileChunk(old_lines=["old b"], new_lines=["new b"]),
    ])]


def test_streaming_parser_preserves_bare_empty_update_lines():
    _, hunks = _stream("*** Begin Patch\n*** Update File: file.txt\n@@\n context before\n\n context after\n"
                       "*** End Patch\n")
    lines = ["context before", "", "context after"]
    assert hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(
        old_lines=lines, new_lines=list(lines), context_line_indices=[(0, 0), (1, 1), (2, 2)])])]


def test_streaming_parser_ignores_empty_lines_after_end_of_file():
    _, hunks = _stream("*** Begin Patch\n*** Update File: file.txt\n@@\n+quux\n*** End of File\n\n*** End Patch\n")
    assert hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(new_lines=["quux"], is_end_of_file=True)])]


def test_streaming_parser_matches_line_ending_behavior():
    _, hunks = _stream("*** Begin Patch\r\n*** Update File: file.txt\r\n@@\r\n-old\r\n+new\r\n*** End Patch\r\n")
    assert hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(old_lines=["old"], new_lines=["new"])])]
    _, hunks = _stream("*** Begin Patch\r\n*** Update File: file.txt\r\n@@\r\n-old\r\r\n+new\r\n*** End Patch\r\n")
    assert hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(old_lines=["old\r"], new_lines=["new"])])]
    # parse_patch strips one \r in str::lines() and the streaming parser another.
    assert parse_patch("*** Begin Patch\r\n*** Update File: file.txt\r\n@@\r\n-old\r\r\n+new\r\n*** End Patch\r\n"
                       ).hunks == [UpdateFile("file.txt", None, [UpdateFileChunk(old_lines=["old"], new_lines=["new"])])]


def test_streaming_parser_finish():
    parser, _ = _stream("*** Begin Patch\n*** Add File: file.txt\n+hello\n*** End Patch")
    assert parser.finish() == [AddFile("file.txt", "hello\n")]
    parser, _ = _stream("*** Begin Patch\n*** Update File: file.txt\n@@\n-old\n+new\n *** End Patch")
    assert parser.finish() == [UpdateFile("file.txt", None, [UpdateFileChunk(old_lines=["old"], new_lines=["new"])])]
    parser, _ = _stream("*** Begin Patch\n*** Add File: file.txt\n+hello\n")
    with pytest.raises(PatchParseError) as error:
        parser.finish()
    assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"


def test_streaming_parser_content_after_end_patch():
    with pytest.raises(PatchParseError) as error:
        _stream("*** Begin Patch\n*** Add File: file.txt\n+hello\n*** End Patch\nextra\n")
    assert str(error.value) == "invalid patch: The last line of the patch must be '*** End Patch'"
    _, hunks = _stream("*** Begin Patch\n*** Add File: file.txt\n+hello\n*** End Patch\n \t\n")
    assert hunks == [AddFile("file.txt", "hello\n")]


@pytest.mark.parametrize("text, message, line", [
    ("bad\n", "The first line of the patch must be '*** Begin Patch'", None),
    ("*** Begin Patch\nbad\n", f"'bad' is not a valid hunk header. {HEADERS}", 2),
    ("*** Begin Patch\n*** Add File: file.txt\nbad\n", f"'bad' is not a valid hunk header. {HEADERS}", 3),
    ("*** Begin Patch\n*** Delete File: file.txt\nbad\n", f"'bad' is not a valid hunk header. {HEADERS}", 3),
    ("*** Begin Patch\n*** Update File: file.txt\n*** End Patch\n",
     "Update file hunk for path 'file.txt' is empty", 2),
    ("*** Begin Patch\n*** Update File: old.txt\n*** Move to: new.txt\n*** Delete File: other.txt\n",
     "Update file hunk for path 'old.txt' is empty", 2),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n*** End Patch\n", "Update hunk does not contain any lines", 4),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n*** End of File\n", "Update hunk does not contain any lines", 4),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n@@\n", f"Unexpected line found in update hunk: '@@'. {UNEXPECTED}", 4),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n-old\nbad\n",
     "Expected update hunk to start with a @@ context marker, got: 'bad'", 5),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n*** Update File: other.txt\n",
     f"Unexpected line found in update hunk: '*** Update File: other.txt'. {UNEXPECTED}", 4),
    ("*** Begin Patch\n*** Update File: file.txt\n@@\n+x\n*** End of File\n+y\n",
     "Expected update hunk to start with a @@ context marker, got: '+y'", 6),
    ("*** Begin Patch\n*** Update File: file.txt\nbad\n", f"Unexpected line found in update hunk: 'bad'. {UNEXPECTED}", 3),
])
def test_streaming_parser_returns_errors(text, message, line):
    with pytest.raises(PatchParseError) as error:
        _stream(text)
    assert (error.value.message, error.value.line_number) == (message, line)


# ------------------------------------------------------------------ seek_sequence.rs

def test_seek_sequence():
    assert seek_sequence(["foo", "bar", "baz"], ["bar", "baz"], 0, False) == 1
    assert seek_sequence(["foo   ", "bar\t\t"], ["foo", "bar"], 0, False) == 0
    assert seek_sequence(["    foo   ", "   bar\t"], ["foo", "bar"], 0, False) == 0
    assert seek_sequence(["just one line"], ["too", "many", "lines"], 0, False) is None
    assert seek_sequence(["a"], [], 3, False) == 3
    assert seek_sequence(["x", "y", "x"], ["x"], 1, False) == 2
    assert seek_sequence(["x", "y", "x"], ["x"], 0, True) == 2
    assert seek_sequence(["x", "y", "z"], ["x"], 0, True) is None  # eof never falls back to the start
    assert seek_sequence(["a \u2013 b\u00a0c \u201cq\u201d \u2018s\u2019"], ['a - b c "q" \'s\''], 0, False) == 0
    # Exact matches win over earlier fuzzy ones.
    assert seek_sequence(["foo ", "foo"], ["foo"], 0, False) == 1
    # Python's str.strip() whitespace is not Rust's: U+001C is not trimmed.
    assert seek_sequence(["foo\x1c"], ["foo"], 0, False) is None


# ------------------------------------------------------------------ lib.rs (apply_patch, standalone summary text)

def test_add_file_hunk_creates_file_with_contents(tmp_path):
    path = os.path.join(real(tmp_path), "add.txt")
    assert tool(tmp_path, wrap(f"*** Add File: {path}\n+ab\n+cd")) == \
        (0, f"Success. Updated the following files:\nA {path}\n")
    assert open(path).read() == "ab\ncd\n"


def test_apply_patch_hunks_accept_relative_and_absolute_paths(tmp_path):
    root = real(tmp_path)
    for name, text in [("relative-delete.txt", "delete relative\n"), ("absolute-delete.txt", "delete absolute\n"),
                       ("relative-update.txt", "relative old\n"), ("absolute-update.txt", "absolute old\n")]:
        write(tmp_path / name, text)
    patch = wrap(f"*** Add File: relative-add.txt\n+relative add\n*** Add File: {root}/absolute-add.txt\n+absolute add\n"
                 f"*** Delete File: relative-delete.txt\n*** Delete File: {root}/absolute-delete.txt\n"
                 f"*** Update File: relative-update.txt\n@@\n-relative old\n+relative new\n"
                 f"*** Update File: {root}/absolute-update.txt\n@@\n-absolute old\n+absolute new")
    assert tool(tmp_path, patch) == (0, (
        f"Success. Updated the following files:\nA relative-add.txt\nA {root}/absolute-add.txt\n"
        f"M relative-update.txt\nM {root}/absolute-update.txt\nD relative-delete.txt\nD {root}/absolute-delete.txt\n"))
    assert tree(tmp_path) == {"relative-add.txt": b"relative add\n", "absolute-add.txt": b"absolute add\n",
                              "relative-update.txt": b"relative new\n", "absolute-update.txt": b"absolute new\n"}


def test_delete_file_hunk_removes_file(tmp_path):
    write(tmp_path / "del.txt", "x")
    assert tool(tmp_path, wrap("*** Delete File: del.txt")) == (0, "Success. Updated the following files:\nD del.txt\n")
    assert not (tmp_path / "del.txt").exists()


def test_update_file_hunk_modifies_content(tmp_path):
    write(tmp_path / "update.txt", "foo\nbar\n")
    assert tool(tmp_path, wrap("*** Update File: update.txt\n@@\n foo\n-bar\n+baz")) == \
        (0, "Success. Updated the following files:\nM update.txt\n")
    assert (tmp_path / "update.txt").read_text() == "foo\nbaz\n"


def test_update_file_hunk_can_move_file(tmp_path):
    write(tmp_path / "src.txt", "line\n")
    assert tool(tmp_path, wrap("*** Update File: src.txt\n*** Move to: dst.txt\n@@\n-line\n+line2")) == \
        (0, "Success. Updated the following files:\nM dst.txt\n")
    assert tree(tmp_path) == {"dst.txt": b"line2\n"}


def test_failed_move_keeps_destination_and_reports_source(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    write(tmp_path / "locked" / "src.txt", "line\n")
    (tmp_path / "out").mkdir()
    os.chmod(tmp_path / "locked", 0o555)
    try:
        code, out = tool(tmp_path, wrap("*** Update File: locked/src.txt\n*** Move to: out/dst.txt\n@@\n-line\n+line2"))
    finally:
        os.chmod(tmp_path / "locked", 0o755)
    assert (code, out) == (1, f"Failed to remove original {real(tmp_path)}/locked/src.txt\n")
    assert (tmp_path / "locked" / "src.txt").read_text() == "line\n"
    assert (tmp_path / "out" / "dst.txt").read_text() == "line2\n"


def test_apply_patch_fails_on_write_error(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    (tmp_path / "locked").mkdir()
    os.chmod(tmp_path / "locked", 0o555)
    try:
        code, out = tool(tmp_path, wrap("*** Add File: locked/new.txt\n+after"))
    finally:
        os.chmod(tmp_path / "locked", 0o755)
    assert (code, out) == (1, f"Failed to write file {real(tmp_path)}/locked/new.txt\n")


def test_multiple_update_chunks_apply_to_single_file(tmp_path):
    write(tmp_path / "multi.txt", "foo\nbar\nbaz\nqux\n")
    assert tool(tmp_path, wrap("*** Update File: multi.txt\n@@\n foo\n-bar\n+BAR\n@@\n baz\n-qux\n+QUX")) == \
        (0, "Success. Updated the following files:\nM multi.txt\n")
    assert (tmp_path / "multi.txt").read_text() == "foo\nBAR\nbaz\nQUX\n"


def test_update_file_hunk_interleaved_changes(tmp_path):
    write(tmp_path / "interleaved.txt", "a\nb\nc\nd\ne\nf\n")
    patch = wrap("*** Update File: interleaved.txt\n@@\n a\n-b\n+B\n@@\n c\n d\n-e\n+E\n@@\n f\n+g\n*** End of File")
    assert tool(tmp_path, patch) == (0, "Success. Updated the following files:\nM interleaved.txt\n")
    assert (tmp_path / "interleaved.txt").read_text() == "a\nB\nc\nd\nE\nf\ng\n"


def test_pure_addition_chunk_followed_by_removal(tmp_path):
    write(tmp_path / "panic.txt", "line1\nline2\nline3\n")
    patch = wrap("*** Update File: panic.txt\n@@\n+after-context\n+second-line\n@@\n line1\n-line2\n-line3\n"
                 "+line2-replacement")
    assert tool(tmp_path, patch)[0] == 0
    assert (tmp_path / "panic.txt").read_text() == "line1\nline2-replacement\nafter-context\nsecond-line\n"


def test_update_line_with_unicode_dash(tmp_path):
    write(tmp_path / "unicode.py", "import asyncio  # local import \u2013 avoids top\u2011level dep\n")
    patch = wrap("*** Update File: unicode.py\n@@\n-import asyncio  # local import - avoids top-level dep\n"
                 "+import asyncio  # HELLO")
    assert tool(tmp_path, patch) == (0, "Success. Updated the following files:\nM unicode.py\n")
    assert (tmp_path / "unicode.py").read_text() == "import asyncio  # HELLO\n"


def test_fuzzy_tiers_and_context_header(tmp_path):
    write(tmp_path / "f.py", "class A:\n    def f(self):   \n        return 1\nclass B:\n    def f(self):\n"
                             "        return 1\n")
    patch = wrap("*** Update File: f.py\n@@ class B:\n def f(self):\n-return 1\n+        return 2")
    assert tool(tmp_path, patch)[0] == 0
    assert (tmp_path / "f.py").read_text() == ("class A:\n    def f(self):   \n        return 1\nclass B:\n"
                                               "    def f(self):\n        return 2\n")
    # The context line keeps the file's own spelling (trailing spaces), the trimmed match is still found.
    patch = wrap("*** Update File: f.py\n@@\n     def f(self):\n-        return 1\n+        return 0")
    assert tool(tmp_path, patch)[0] == 0
    assert (tmp_path / "f.py").read_text().startswith("class A:\n    def f(self):   \n        return 0\n")


def test_context_not_found(tmp_path):
    write(tmp_path / "f.txt", "a\n")
    assert tool(tmp_path, wrap("*** Update File: f.txt\n@@ def missing():\n-a\n+b")) == \
        (1, f"{VERIFY}Failed to find context 'def missing():' in {real(tmp_path)}/f.txt")


# ------------------------------------------------------------------ file_update_tests.rs (new contents)

@pytest.mark.parametrize("original, body, expected", [
    ("foo\nbar\nbaz\nqux\n", "@@\n foo\n-bar\n+BAR\n@@\n baz\n-qux\n+QUX", "foo\nBAR\nbaz\nQUX\n"),
    ("foo\nbar\nbaz\n", "@@\n-foo\n+FOO\n bar\n", "FOO\nbar\nbaz\n"),
    ("foo\nbar\nbaz\n", "@@\n foo\n bar\n-baz\n+BAZ\n", "foo\nbar\nBAZ\n"),
    ("foo\nbar\nbaz\n", "@@\n+quux\n*** End of File\n", "foo\nbar\nbaz\nquux\n"),
    ("a\nb\nc\nd\ne\nf\n", "@@\n a\n-b\n+B\n@@\n d\n-e\n+E\n@@\n f\n+g\n*** End of File", "a\nB\nc\nd\nE\nf\ng\n"),
    # The trailing empty old/new line stands for the final newline and is retried without it.
    ("x\ny\n", "@@\n x\n-y\n-\n+z\n+", "x\nz\n"),
    ("", "@@\n+first", "first\n"),
])
def test_derived_new_contents(tmp_path, original, body, expected):
    write(tmp_path / "file.txt", original)
    assert tool(tmp_path, wrap(f"*** Update File: file.txt\n{body}"))[0] == 0
    assert (tmp_path / "file.txt").read_bytes() == expected.encode()


# ------------------------------------------------------------------ tests/suite/tool.rs (line endings, edge cases)

@pytest.mark.parametrize("original, body, expected", [
    (b"one\r\ntwo\r\nthree\r\n", "@@\n-one\n+uno\n@@\n two\n+\n+between\n three",
     b"uno\r\ntwo\r\n\r\nbetween\r\nthree\r\n"),
    (b"a\r\n\r\n", "@@\n+new", b"a\r\n\r\nnew\r\n"),
    (b"one\rtwo\rthree\r", "@@\n-one\n+uno\n@@\n two\n+\n+between\n three", b"uno\rtwo\r\rbetween\rthree\r"),
    (b"a\nb\n", "@@\n-a\n-b\n+b\n+b\n+a", b"b\nb\na\n"),
    (b"same\r\nsame\n", "@@\n-same\n same", b"same\n"),
    (b"one\r\ntwo\rthree\nfour\r\n", "@@\n one\n two\n-three\n+THREE\n four", b"one\r\ntwo\rTHREE\r\nfour\r\n"),
    (b"one\r\ntwo", "@@\n-one\n+ONE", b"ONE\r\ntwo\r\n"),
    (b"no newline at end", "@@\n-no newline at end\n+first line\n+second line", b"first line\nsecond line\n"),
])
def test_line_endings(tmp_path, original, body, expected):
    write(tmp_path / "t.txt", original)
    assert tool(tmp_path, wrap(f"*** Update File: t.txt\n{body}")) == \
        (0, "Success. Updated the following files:\nM t.txt\n")
    assert (tmp_path / "t.txt").read_bytes() == expected


def test_rejects_overlapping_end_of_file_chunks(tmp_path):
    write(tmp_path / "overlapping.txt", "one\n")
    patch = wrap("*** Update File: overlapping.txt\n@@\n-one\n+first\n@@\n-one\n+second\n*** End of File")
    assert tool(tmp_path, patch) == (1, f"{VERIFY}Failed to find expected lines in {real(tmp_path)}/overlapping.txt:\none")
    assert run(tmp_path, patch, "--standalone") == \
        (1, "", f"Failed to find expected lines in {real(tmp_path)}/overlapping.txt:\none\n")
    assert (tmp_path / "overlapping.txt").read_text() == "one\n"


def test_moves_file_to_new_directory_and_overwrites(tmp_path):
    write(tmp_path / "old" / "name.txt", "old content\n")
    write(tmp_path / "renamed" / "dir" / "name.txt", "existing\n")
    assert tool(tmp_path, wrap("*** Update File: old/name.txt\n*** Move to: renamed/dir/name.txt\n@@\n-old content\n"
                               "+new content")) == (0, "Success. Updated the following files:\nM renamed/dir/name.txt\n")
    assert tree(tmp_path) == {"renamed/dir/name.txt": b"new content\n"}


def test_add_overwrites_existing_file_and_creates_parents(tmp_path):
    write(tmp_path / "duplicate.txt", "old content\n")
    patch = wrap("*** Add File: duplicate.txt\n+new content\n*** Add File: a/b/c.txt\n*** Add File: e.txt\n+")
    assert tool(tmp_path, patch) == (0, "Success. Updated the following files:\nA duplicate.txt\nA a/b/c.txt\nA e.txt\n")
    assert tree(tmp_path) == {"duplicate.txt": b"new content\n", "a/b/c.txt": b"", "e.txt": b"\n"}


# ------------------------------------------------------------------ the Codex CLI tool's answers (core)

def test_tool_parse_errors(tmp_path):
    assert tool(tmp_path, "bad") == (1, f"{VERIFY}invalid patch: The first line of the patch must be '*** Begin Patch'")
    assert tool(tmp_path, wrap("*** Frobnicate File: foo")) == \
        (1, f"{VERIFY}invalid hunk at line 2, '*** Frobnicate File: foo' is not a valid hunk header. {HEADERS}")
    assert tool(tmp_path, wrap("*** Update File: foo.txt")) == \
        (1, f"{VERIFY}invalid hunk at line 2, Update file hunk for path 'foo.txt' is empty")
    assert tool(tmp_path, "") == (1, f"{VERIFY}invalid patch: The last line of the patch must be '*** End Patch'")


def test_tool_rejects_empty_patch_and_environment_ids(tmp_path):
    assert tool(tmp_path, "*** Begin Patch\n*** End Patch") == (1, "patch rejected: empty patch")
    assert tool(tmp_path, "*** Begin Patch\n*** Environment ID: remote\n*** Add File: a\n+x\n*** End Patch") == \
        (1, "apply_patch environment selection is unavailable for this turn")
    assert tree(tmp_path) == {}


def test_tool_missing_files(tmp_path):
    root = real(tmp_path)
    assert tool(tmp_path, wrap("*** Update File: missing.txt\n@@\n-old\n+new")) == \
        (1, f"{VERIFY}Failed to read file to update {root}/missing.txt: No such file or directory (os error 2)")
    assert tool(tmp_path, wrap("*** Delete File: missing.txt")) == \
        (1, f"{VERIFY}Failed to read {root}/missing.txt: No such file or directory (os error 2)")
    (tmp_path / "dir").mkdir()
    assert tool(tmp_path, wrap("*** Delete File: dir")) == \
        (1, f"{VERIFY}Failed to read {root}/dir: path `{root}/dir` is not a file")
    assert tool(tmp_path, wrap("*** Update File: dir\n@@\n+x")) == \
        (1, f"{VERIFY}Failed to read file to update {root}/dir: path `{root}/dir` is not a file")
    assert (tmp_path / "dir").is_dir()


def test_tool_non_utf8_files(tmp_path):
    root = real(tmp_path)
    write(tmp_path / "bin.dat", b"ok\xff\xfe")
    write(tmp_path / "cut.dat", b"ok\xe2\x82")
    assert tool(tmp_path, wrap("*** Delete File: bin.dat")) == \
        (1, f"{VERIFY}Failed to read {root}/bin.dat: invalid utf-8 sequence of 1 bytes from index 2")
    assert tool(tmp_path, wrap("*** Update File: cut.dat\n@@\n+x")) == \
        (1, f"{VERIFY}Failed to read file to update {root}/cut.dat: incomplete utf-8 byte sequence from index 2")


def test_tool_missing_context(tmp_path):
    write(tmp_path / "modify.txt", "line1\nline2\n")
    assert tool(tmp_path, wrap("*** Update File: modify.txt\n@@\n-missing\n+changed")) == \
        (1, f"{VERIFY}Failed to find expected lines in {real(tmp_path)}/modify.txt:\nmissing")
    write(tmp_path / "two_chunks.txt", "a\nb\nc\nd\n")
    assert tool(tmp_path, "*** Begin Patch\n*** Update File: two_chunks.txt\n@@\n-b\n+B\n\n-d\n+D\n*** End Patch") == \
        (1, f"{VERIFY}Failed to find expected lines in {real(tmp_path)}/two_chunks.txt:\nb\n\nd")
    assert (tmp_path / "two_chunks.txt").read_text() == "a\nb\nc\nd\n"


def test_tool_verification_failure_has_no_side_effects(tmp_path):
    patch = wrap("*** Add File: created.txt\n+hello\n*** Update File: missing.txt\n@@\n-old\n+new")
    assert tool(tmp_path, patch)[0] == 1
    assert tree(tmp_path) == {}
    # The standalone executable has no verification step: the Add stays applied.
    assert run(tmp_path, patch, "--standalone") == \
        (1, "", f"Failed to read file to update {real(tmp_path)}/missing.txt: No such file or directory (os error 2)\n")
    assert tree(tmp_path) == {"created.txt": b"hello\n"}


def test_tool_rejects_duplicate_resolved_paths(tmp_path):
    write(tmp_path / "duplicate.txt", "before\n")
    patch = wrap("*** Update File: duplicate.txt\n@@\n-before\n+one\n*** Update File: ./duplicate.txt\n@@\n-before\n+two")
    assert tool(tmp_path, patch) == (1, f"{VERIFY}invalid patch: multiple operations target {real(tmp_path)}/duplicate.txt")
    assert (tmp_path / "duplicate.txt").read_text() == "before\n"
    write(tmp_path / "first.txt", "first before\n")
    write(tmp_path / "second.txt", "second before\n")
    patch = wrap("*** Update File: first.txt\n@@\n-first before\n+first after\n*** Update File: second.txt\n@@\n"
                 "-second before\n+second after")
    assert tool(tmp_path, patch) == (0, "Success. Updated the following files:\nM first.txt\nM second.txt\n")


def test_tool_accepts_heredoc_wrapped_and_padded_patches(tmp_path):
    assert tool(tmp_path, "<<'EOF'\n*** Begin Patch\n*** Add File: lenient.txt\n+lenient\n*** End Patch\nEOF\n") == \
        (0, "Success. Updated the following files:\nA lenient.txt\n")
    assert (tmp_path / "lenient.txt").read_text() == "lenient\n"
    assert tool(tmp_path, "\n\n  *** Begin Patch  \n  *** Update File: lenient.txt\n@@\n-lenient\n+padded\n *** End Patch \n\n") \
        == (0, "Success. Updated the following files:\nM lenient.txt\n")
    assert (tmp_path / "lenient.txt").read_text() == "padded\n"


def test_tool_end_of_file_anchor(tmp_path):
    write(tmp_path / "tail.txt", "alpha\nlast\n")
    assert tool(tmp_path, wrap("*** Update File: tail.txt\n@@\n-last\n+end\n*** End of File"))[0] == 0
    assert (tmp_path / "tail.txt").read_text() == "alpha\nend\n"


def test_tool_apply_stage_error_after_partial_success(tmp_path):
    (tmp_path / "blocker").write_text("a file, not a directory\n")
    patch = wrap("*** Add File: ok.txt\n+fine\n*** Add File: blocker/new.txt\n+x")
    assert tool(tmp_path, patch) == (1, f"Failed to write file {real(tmp_path)}/blocker/new.txt\n")
    assert (tmp_path / "ok.txt").read_text() == "fine\n"


def test_model_output_wraps_exec_stage_results_only(tmp_path):
    code, out, _ = run(tmp_path, wrap("*** Add File: nested/new.txt\n+created"), "--model-output")
    assert code == 0
    assert re.fullmatch(r"Exit code: 0\nWall time: [0-9]+(?:\.[0-9]+)? seconds\nOutput:\n"
                        r"Success\. Updated the following files:\nA nested/new\.txt\n", out), out
    code, out, _ = run(tmp_path, wrap("*** Delete File: gone.txt"), "--model-output")
    assert (code, out) == (1, f"{VERIFY}Failed to read {real(tmp_path)}/gone.txt: No such file or directory (os error 2)")


def test_in_process_helpers_match_the_program(tmp_path):
    write(tmp_path / "a.txt", "x\n")
    patch = wrap("*** Update File: a.txt\n@@\n-x\n+y")
    assert apply_patch_to_dir(patch, str(tmp_path)) == (0, "Success. Updated the following files:\nM a.txt\n")
    assert apply_patch_standalone_to_dir(wrap("*** Frobnicate File: foo"), str(tmp_path)) == \
        (1, "", f"Invalid patch hunk on line 2: '*** Frobnicate File: foo' is not a valid hunk header. {HEADERS}\n")


# ------------------------------------------------------------------ standalone executable (tests/suite/tool.rs, cli.rs)

def test_standalone_messages(tmp_path):
    root = real(tmp_path)
    assert run(tmp_path, "*** Begin Patch\n*** End Patch", "--standalone") == (1, "", "No files were modified.\n")
    assert run(tmp_path, wrap("*** Update File: foo.txt"), "--standalone") == \
        (1, "", "Invalid patch hunk on line 2: Update file hunk for path 'foo.txt' is empty\n")
    assert run(tmp_path, "bad", "--standalone") == \
        (1, "", "Invalid patch: The first line of the patch must be '*** Begin Patch'\n")
    assert run(tmp_path, wrap("*** Delete File: missing.txt"), "--standalone") == \
        (1, "", f"Failed to delete file {root}/missing.txt\n")
    (tmp_path / "dir").mkdir()
    assert run(tmp_path, wrap("*** Delete File: dir"), "--standalone") == (1, "", f"Failed to delete file {root}/dir\n")
    assert run(tmp_path, wrap("*** Add File: a.txt\n+hello"), "--standalone") == \
        (0, "Success. Updated the following files:\nA a.txt\n", "")


# ------------------------------------------------------------------ tests/fixtures/scenarios (standalone executable)

SCENARIOS = {
    "001_add_file": ({}, "*** Begin Patch\n*** Add File: bar.md\n+This is a new file\n*** End Patch\n",
                     {"bar.md": "This is a new file\n"}),
    "002_multiple_operations": (
        {"delete.txt": "obsolete\n", "modify.txt": "line1\nline2\n"},
        "*** Begin Patch\n*** Add File: nested/new.txt\n+created\n*** Delete File: delete.txt\n"
        "*** Update File: modify.txt\n@@\n-line2\n+changed\n*** End Patch\n",
        {"modify.txt": "line1\nchanged\n", "nested/new.txt": "created\n"}),
    "003_multiple_chunks": (
        {"multi.txt": "line1\nline2\nline3\nline4\n"},
        "*** Begin Patch\n*** Update File: multi.txt\n@@\n-line2\n+changed2\n@@\n-line4\n+changed4\n*** End Patch\n",
        {"multi.txt": "line1\nchanged2\nline3\nchanged4\n"}),
    "004_move_to_new_directory": (
        {"old/name.txt": "old content\n", "old/other.txt": "unrelated file\n"},
        "*** Begin Patch\n*** Update File: old/name.txt\n*** Move to: renamed/dir/name.txt\n@@\n-old content\n"
        "+new content\n*** End Patch\n",
        {"old/other.txt": "unrelated file\n", "renamed/dir/name.txt": "new content\n"}),
    "005_rejects_empty_patch": ({"foo.txt": "stable\n"}, "*** Begin Patch\n*** End Patch\n", {"foo.txt": "stable\n"}),
    "006_rejects_missing_context": (
        {"modify.txt": "line1\nline2\n"},
        "*** Begin Patch\n*** Update File: modify.txt\n@@\n-missing\n+changed\n*** End Patch\n",
        {"modify.txt": "line1\nline2\n"}),
    "007_rejects_missing_file_delete": (
        {"foo.txt": "stable\n"}, "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch\n",
        {"foo.txt": "stable\n"}),
    "008_rejects_empty_update_hunk": (
        {"foo.txt": "stable\n"}, "*** Begin Patch\n*** Update File: foo.txt\n*** End Patch\n", {"foo.txt": "stable\n"}),
    "009_requires_existing_file_for_update": (
        {"foo.txt": "stable\n"}, "*** Begin Patch\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch\n",
        {"foo.txt": "stable\n"}),
    "010_move_overwrites_existing_destination": (
        {"old/name.txt": "from\n", "old/other.txt": "unrelated file\n", "renamed/dir/name.txt": "existing\n"},
        "*** Begin Patch\n*** Update File: old/name.txt\n*** Move to: renamed/dir/name.txt\n@@\n-from\n+new\n"
        "*** End Patch\n",
        {"old/other.txt": "unrelated file\n", "renamed/dir/name.txt": "new\n"}),
    "011_add_overwrites_existing_file": (
        {"duplicate.txt": "old content\n"}, "*** Begin Patch\n*** Add File: duplicate.txt\n+new content\n*** End Patch\n",
        {"duplicate.txt": "new content\n"}),
    "012_delete_directory_fails": (
        {"dir/foo.txt": "stable\n"}, "*** Begin Patch\n*** Delete File: dir\n*** End Patch\n", {"dir/foo.txt": "stable\n"}),
    "013_rejects_invalid_hunk_header": (
        {"foo.txt": "stable\n"}, "*** Begin Patch\n*** Frobnicate File: foo\n*** End Patch\n", {"foo.txt": "stable\n"}),
    "014_update_file_appends_trailing_newline": (
        {"no_newline.txt": "no newline at end"},
        "*** Begin Patch\n*** Update File: no_newline.txt\n@@\n-no newline at end\n+first line\n+second line\n"
        "*** End Patch\n",
        {"no_newline.txt": "first line\nsecond line\n"}),
    "015_failure_after_partial_success_leaves_changes": (
        {}, "*** Begin Patch\n*** Add File: created.txt\n+hello\n*** Update File: missing.txt\n@@\n-old\n+new\n"
            "*** End Patch\n",
        {"created.txt": "hello\n"}),
    "016_pure_addition_update_chunk": (
        {"input.txt": "line1\nline2\n"},
        "*** Begin Patch\n*** Update File: input.txt\n@@\n+added line 1\n+added line 2\n*** End Patch\n",
        {"input.txt": "line1\nline2\nadded line 1\nadded line 2\n"}),
    "017_whitespace_padded_hunk_header": (
        {"foo.txt": "old\n"}, "*** Begin Patch\n  *** Update File: foo.txt\n@@\n-old\n+new\n*** End Patch\n",
        {"foo.txt": "new\n"}),
    "018_whitespace_padded_patch_markers": (
        {"file.txt": "one\n"}, " *** Begin Patch\n*** Update File: file.txt\n@@\n-one\n+two\n*** End Patch \n",
        {"file.txt": "two\n"}),
    "019_unicode_simple": (
        {"foo.txt": "line1\nna\u00efve caf\u00e9\nline3\n"},
        "*** Begin Patch\n*** Update File: foo.txt\n@@\n line1\n-na\u00efve caf\u00e9\n+na\u00efve caf\u00e9 \u2705\n"
        "*** End Patch\n",
        {"foo.txt": "line1\nna\u00efve caf\u00e9 \u2705\nline3\n"}),
    "020_delete_file_success": (
        {"keep.txt": "keep\n", "obsolete.txt": "obsolete\n"}, "*** Begin Patch\n*** Delete File: obsolete.txt\n*** End Patch\n",
        {"keep.txt": "keep\n"}),
    "020_whitespace_padded_patch_marker_lines": (
        {"file.txt": "one\n"}, "*** Begin Patch \n*** Update File: file.txt\n@@\n-one\n+two\n *** End Patch\n",
        {"file.txt": "two\n"}),
    "021_update_file_deletion_only": (
        {"lines.txt": "line1\nline2\nline3\n"},
        "*** Begin Patch\n*** Update File: lines.txt\n@@\n line1\n-line2\n line3\n*** End Patch\n",
        {"lines.txt": "line1\nline3\n"}),
    "022_update_file_end_of_file_marker": (
        {"tail.txt": "first\nsecond\n"},
        "*** Begin Patch\n*** Update File: tail.txt\n@@\n first\n-second\n+second updated\n*** End of File\n*** End Patch\n",
        {"tail.txt": "first\nsecond updated\n"}),
    "023_preserves_crlf_line_endings": (
        {"lines.txt": "one\r\ntwo\r\nthree\r\n"},
        "*** Begin Patch\n*** Update File: lines.txt\n@@\n-one\n+ONE\n two\n+between\n three\n*** End Patch\n",
        {"lines.txt": "ONE\r\ntwo\r\nbetween\r\nthree\r\n"}),
    "024_preserves_mixed_line_endings": (
        {"lines.txt": "one\r\ntwo\rthree\nfour\r\n"},
        "*** Begin Patch\n*** Update File: lines.txt\n@@\n one\n two\n-three\n+THREE\n four\n*** End Patch\n",
        {"lines.txt": "one\r\ntwo\rTHREE\r\nfour\r\n"}),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios(tmp_path, name):
    inputs, patch, expected = SCENARIOS[name]
    for mode in ("--standalone", None):
        root = tmp_path / (mode or "tool").strip("-")
        root.mkdir()
        for rel, text in inputs.items():
            write(root / rel, text)
        run(root, patch, *([mode] if mode else []))
        want = dict(expected)
        if mode is None and name.startswith("015_"):
            want = {}  # the Codex tool verifies the whole patch first
        assert tree(root) == {rel: text.encode("utf-8") for rel, text in want.items()}
