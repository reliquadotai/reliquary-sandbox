"""The state archive format and its vetting, in pure Python.

An archive made by `LocalRuntime.archive(paths)` is a PAX tar file whose first
member is `.reliquary-archive.json` (`{"version": 1, "roots": [...]}`, the
absolute paths archived) and whose other members are named `<i>` (root i
itself) or `<i>/<path relative to root i>`. It holds only regular files,
directories and symlinks, with no owners and no setuid/setgid bits.

Nothing an agent produced ever reaches an extractor as-is. `normalise_created`
rebuilds busybox's output; `rebuild_for_restore` parses received bytes with
Python's tarfile (which stops at the first zero block, so members hidden after
it are dropped), vets every member, and builds fresh tars from the vetted
members only. Those are the only bytes busybox extracts.

Confinement is enforced at restore, not at archive time (a symlink is harmless
data inside an archive):

* names must be `<i>` or `<i>/<relative path>` with no `..`, for i < len(roots);
* a symlink target is resolved component by component against the archive's
  own symlink members (`_check_link`, at most 40 substitutions, `..` applied
  to the resolved prefix) and must end inside the roots; an absolute target
  ending outside them is kept as an inert link (venvs need them), which is
  safe because no member may pass through a symlink member, so nothing in the
  archive writes through it; later `LocalRuntime.write` calls refuse paths whose
  resolved parent leaves the restored roots; a walk that passes through such an
  outside link and continues, a relative walk leaving the roots, an empty
  target and a cycle are refused; with `contained_links` (the verifiers
  bridge's grading restore) no link may end outside the roots, and the
  restore then resolves every restored link inside the box itself
  (`LocalRuntime.restore_archive`);
* hardlinks stay hardlinks, only to an earlier regular file of the same root
  (never expanded: a 1 MiB file with thousands of links stays small); FIFOs,
  devices and anything else are refused; duplicate names are refused; no
  member may be below a file or symlink member;
* contents are streamed from the input into output buffers bounded by the
  input size (restore) or `max_bytes` (archive), never copied per member.

Work is bounded: at most MAX_MEMBERS members, MAX_COMPONENTS components per
name (below its root) and per symlink target, and MAX_RESOLUTION_STEPS steps
of symlink resolution per archive; names are checked over a tree built once
(`_Node`), so vetting is linear in the total length of the names.

Restore clears each root first (`TaskBoxRuntime.clear_root`, which refuses a
root that is a symlink in the image) and then extracts, so image symlinks
inside a root are never written through and deleted files stay deleted.
"""

from __future__ import annotations

import errno
import io
import json
import posixpath
import tarfile

MANIFEST = ".reliquary-archive.json"
FORMAT_VERSION = 1
MAX_MANIFEST_BYTES = 64 * 1024
_STRIP_MODE = 0o6000  # setuid, setgid
MAX_MEMBERS = 100_000
MAX_COMPONENTS = 256  # per member name (below its root) and per symlink target
MAX_RESOLUTION_STEPS = 16_000_000  # symlink resolution work per archive (~1.5 s of CPU)


class StateUnreadable(OSError):
    """Not a regular file, or an archive with members we refuse (EINVAL)."""

    def __init__(self, message: str):
        super().__init__(errno.EINVAL, message)


class FileTooLarge(OSError):
    """A read or an archive exceeded its byte bound (errno EFBIG)."""

    def __init__(self, what: str, limit: int):
        super().__init__(errno.EFBIG, f"{what} exceeds {limit} bytes")


def clean_roots(paths: list[str], workdir: str = "/") -> list[str]:
    """Absolute, normalised, distinct, non-nested roots; never `/` or `..`."""
    if not paths:
        raise ValueError("at least one path is required")
    roots = []
    for path in paths:
        if not isinstance(path, str) or not path or ".." in path.split("/"):
            raise ValueError(f"refusing path {path!r}")
        absolute = posixpath.normpath(posixpath.join(workdir, path))
        if absolute in ("/", "//"):
            raise ValueError("refusing to use / as a root")
        roots.append(absolute)
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            if _inside(a, b) or _inside(b, a):
                raise ValueError(f"roots {a!r} and {b!r} overlap")
    return roots


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _clean_relative(name: str) -> str:
    if not name or name.startswith("/") or "\x00" in name:
        raise StateUnreadable(f"archive: absolute or empty member {name!r}")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise StateUnreadable(f"archive: member {name!r} escapes its root")
    return "/".join(parts)


class _Bounded(io.BytesIO):
    """An output buffer that refuses to grow past `limit`: nothing built here
    holds more than the input plus this bound."""

    def __init__(self, limit: int, what: str):
        super().__init__()
        self.limit, self.what = limit, what

    def write(self, data) -> int:  # type: ignore[override]
        if self.tell() + len(data) > self.limit:
            raise FileTooLarge(self.what, self.limit)
        return super().write(data)


def _open(data: bytes) -> tuple[tarfile.TarFile, list[tarfile.TarInfo]]:
    """Member headers up to the first zero block (at most MAX_MEMBERS).
    Contents are not read: they are streamed from `data` when the clean tar
    is written."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        infos = []
        for info in tar:
            infos.append(info)
            if len(infos) > MAX_MEMBERS:
                raise StateUnreadable(f"archive: more than {MAX_MEMBERS} members")
        return tar, infos
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        if isinstance(exc, StateUnreadable):
            raise
        raise StateUnreadable(f"not a tar archive: {exc}") from exc


def _clean_info(name: str, info: tarfile.TarInfo, kind: bytes, size: int = 0,
                linkname: str = "") -> tarfile.TarInfo:
    clean = tarfile.TarInfo(name)
    clean.type = kind
    clean.mode = info.mode & 0o7777 & ~_STRIP_MODE
    clean.mtime = int(info.mtime) if isinstance(info.mtime, (int, float)) and 0 <= info.mtime < 2**33 else 0
    clean.uid = clean.gid = 0
    clean.uname = clean.gname = ""
    clean.size = size
    clean.linkname = linkname
    return clean


class _Node:
    """A path in the archive's absolute tree: built once, so every check below
    is linear in the total length of the names (no per-ancestor joins)."""

    __slots__ = ("children", "kind", "parent", "inside", "target")

    def __init__(self, parent: _Node | None, inside: bool):
        self.children: dict[str, _Node] = {}
        self.kind: bytes | None = None  # None: an implicit directory
        self.parent = parent
        self.inside = inside
        self.target: str | None = None  # symlink target


def _tree(roots: list[str]) -> tuple[_Node, list[_Node]]:
    top = _Node(None, False)
    nodes = []
    for root in roots:
        node = top
        for part in [p for p in root.split("/") if p]:
            node = node.children.setdefault(part, _Node(node, False))
        node.inside = True
        nodes.append(node)
    return top, nodes


def _locate(root_nodes: list[_Node], name: str, *, create: bool) -> _Node | None:
    head, _, rest = name.partition("/")
    node = root_nodes[int(head)]
    for part in rest.split("/") if rest else []:
        child = node.children.get(part)
        if child is None:
            if not create:
                return None
            if node.kind not in (None, tarfile.DIRTYPE):
                raise StateUnreadable(f"archive: {name!r} passes through a non-directory member")
            child = node.children[part] = _Node(node, True)
        node = child
    return node


def _vet(infos: list[tarfile.TarInfo], name_of, roots: list[str]
         ) -> tuple[list[tuple[str, tarfile.TarInfo]], _Node, dict[str, _Node]]:
    """(clean name, source header) per member, plus the tree. Only regular
    files, directories, symlinks and hardlinks to an earlier regular file of
    the same root; no duplicates; nothing below a file or symlink member; at
    most MAX_COMPONENTS components per name and per symlink target. Hardlinks
    stay hardlinks (never expanded into copies)."""
    top, root_nodes = _tree(roots)
    out = []
    symlinks: dict[str, _Node] = {}
    for info in infos:
        name = name_of(info.name)
        if name.count("/") > MAX_COMPONENTS:
            raise StateUnreadable(f"archive: a member has more than {MAX_COMPONENTS} components")
        node = _locate(root_nodes, name, create=True)
        if node.kind is not None:
            raise StateUnreadable(f"archive: duplicate member {name!r}")
        if info.islnk():
            target = _locate(root_nodes, name_of(info.linkname), create=False)
            if (target is None or target.kind not in (tarfile.REGTYPE, tarfile.LNKTYPE)
                    or name_of(info.linkname).split("/", 1)[0] != name.split("/", 1)[0]):
                raise StateUnreadable(f"archive: hardlink {name!r} to a non-file, later or "
                                      "other-root member")
            kind = tarfile.LNKTYPE
        elif info.isfile():
            kind = tarfile.REGTYPE
        elif info.isdir():
            kind = tarfile.DIRTYPE
        elif info.issym():
            if len([part for part in info.linkname.split("/") if part]) > MAX_COMPONENTS:
                raise StateUnreadable(f"archive: symlink {name!r} target has more than "
                                      f"{MAX_COMPONENTS} components")
            kind, node.target = tarfile.SYMTYPE, info.linkname
            symlinks[name] = node
        else:
            raise StateUnreadable(f"archive: {name!r} is not a file, directory or link")
        if kind != tarfile.DIRTYPE and node.children:
            raise StateUnreadable(f"archive: members pass through the non-directory {name!r}")
        node.kind = kind
        out.append((name, info))
    return out, top, symlinks


def _copy(source: tarfile.TarFile, target: tarfile.TarFile, info: tarfile.TarInfo,
          clean: tarfile.TarInfo) -> None:
    try:
        if clean.type == tarfile.REGTYPE:
            target.addfile(clean, source.extractfile(info))  # streamed in chunks
        else:
            target.addfile(clean)
    except FileTooLarge:
        raise
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise StateUnreadable(f"archive: {info.name!r} cannot be copied: {exc}") from exc


def _clean_member(name: str, info: tarfile.TarInfo, new_name: str, link_name) -> tarfile.TarInfo:
    if info.islnk():
        return _clean_info(new_name, info, tarfile.LNKTYPE, linkname=link_name(info.linkname))
    if info.isfile():
        return _clean_info(new_name, info, tarfile.REGTYPE, info.size)
    if info.isdir():
        return _clean_info(new_name, info, tarfile.DIRTYPE)
    return _clean_info(new_name, info, tarfile.SYMTYPE, linkname=info.linkname)


def normalise_created(raw: bytes, roots: list[str], max_bytes: int) -> bytes:
    """Rebuild busybox's `tar -c` output (names relative to `/`) into the
    state format, PAX, streaming contents into a buffer bounded by
    `max_bytes` (so at most the input plus `max_bytes` are held). Any
    symlink target is allowed here."""
    relative = [root.lstrip("/") for root in roots]

    def name_of(raw_name: str) -> str:
        name = _clean_relative(raw_name)
        for index, root in enumerate(relative):
            if name == root:
                return str(index)
            if name.startswith(root + "/"):
                return f"{index}/{name[len(root) + 1:]}"
        raise StateUnreadable(f"archive: {raw_name!r} is outside the archived paths")

    source, infos = _open(raw)
    with source:
        members, _, _ = _vet(infos, name_of, roots)
        manifest = json.dumps({"version": FORMAT_VERSION, "roots": roots}, sort_keys=True).encode()
        head = tarfile.TarInfo(MANIFEST)
        head.size, head.mode = len(manifest), 0o644
        buffer = _Bounded(max_bytes, "the archive")
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
            out.addfile(head, io.BytesIO(manifest))
            for name, info in members:
                _copy(source, out, info, _clean_member(name, info, name, name_of))
        return buffer.getvalue()


MAX_LINK_HOPS = 40


class _Budget:
    """Total resolution steps for one archive: bounds the CPU a crafted
    archive can cost (MAX_RESOLUTION_STEPS)."""

    def __init__(self) -> None:
        self.steps = 0

    def spend(self, steps: int) -> None:
        self.steps += steps
        if self.steps > MAX_RESOLUTION_STEPS:
            raise StateUnreadable("archive: symlinks too expensive to resolve")


def _check_link(name: str, node: _Node, top: _Node, budget: _Budget,
                contained: bool = False) -> None:
    """Resolve a symlink member's target as the kernel would, over the
    archive's tree (its own symlink members substituted, at most
    MAX_LINK_HOPS per link), with `..` applied to the resolved prefix.
    Accepted when the result is inside the roots, or (unless `contained`) when
    it ends outside them through an absolute target, here or at the end of a
    chain (an inert link, never written through). Refused: a relative walk
    leaving the roots, a walk that continues after passing through a link
    leaving the roots, a cycle, an empty target, and with `contained` any
    link ending outside the roots. Linear in the path lengths walked."""
    if not node.target or "\x00" in node.target:
        raise StateUnreadable(f"archive: symlink {name!r} has an empty target")
    hops = [0]

    def walk(start: list[tuple[_Node | None, bool]], text: str
             ) -> tuple[list[tuple[_Node | None, bool]], bool]:
        absolute = text.startswith("/")
        stack = [(top, False)] if absolute else list(start)
        budget.spend(len(stack))
        entered = stack[-1][1]
        pending = [p for p in text.split("/") if p not in ("", ".")]
        budget.spend(len(pending))
        position = 0
        while position < len(pending):
            component = pending[position]
            position += 1
            budget.spend(1)
            if component == "..":
                if len(stack) > 1:
                    stack.pop()
            else:
                parent, parent_inside = stack[-1]
                child = parent.children.get(component) if parent is not None else None
                stack.append((child, child.inside if child is not None else parent_inside))
            current, inside = stack[-1]
            if not inside:
                if entered or not absolute:
                    raise StateUnreadable(f"archive: symlink {name!r} resolves outside the roots")
                continue
            entered = True
            if current is not None and current.kind == tarfile.SYMTYPE:
                hops[0] += 1
                if hops[0] > MAX_LINK_HOPS:
                    raise StateUnreadable(f"archive: symlink {name!r} does not resolve (a cycle?)")
                stack, inside = walk(stack[:-1], current.target or "")
                if not inside:
                    if position < len(pending):
                        raise StateUnreadable(
                            f"archive: symlink {name!r} resolves through a link leaving the roots")
                    return stack, False
        return stack, stack[-1][1]

    start: list[tuple[_Node | None, bool]] = []
    parent = node.parent
    while parent is not None:
        start.append((parent, parent.inside))
        parent = parent.parent
    start.reverse()
    budget.spend(len(start))
    # Ending outside is only possible through an absolute target (here or at
    # the end of a chain, e.g. venv `python3 -> python -> /usr/bin/python3`):
    # an inert link. Relative walks that leave the roots raised above.
    _, inside = walk(start, node.target)
    if contained and not inside:
        raise StateUnreadable(f"archive: symlink {name!r} leads outside the roots")


def rebuild_for_restore(data: bytes, roots: list[str], *, contained_links: bool = False
                        ) -> list[tuple[str, str, bytes]]:
    """Vet a received state archive and return `(root, extract_dir, clean_tar)`
    per root. `contained_links`: every symlink must resolve inside the roots
    (no inert link leaving them; see `_check_link`). A root archived as a directory is rebuilt with names relative to
    it and extracted in the root; a root archived as a file or symlink is a
    single member named `basename(root)`, extracted in `dirname(root)`. Names
    only get shorter, hardlinks stay hardlinks, contents are streamed, and the
    output is bounded by the input size (plus a small slack), so at most about
    twice the input is held. Raises StateUnreadable for anything refused."""
    source, infos = _open(data)
    with source:
        if not infos or infos[0].name != MANIFEST or not infos[0].isfile() \
                or infos[0].size > MAX_MANIFEST_BYTES:
            raise StateUnreadable("archive: the manifest must come first")
        try:
            manifest = json.loads(source.extractfile(infos[0]).read())
        except ValueError as exc:
            raise StateUnreadable("archive: unreadable manifest") from exc
        if (not isinstance(manifest, dict) or manifest.get("version") != FORMAT_VERSION
                or not isinstance(manifest.get("roots"), list)):
            raise StateUnreadable("archive: unsupported manifest")
        if len(manifest["roots"]) != len(roots):
            raise StateUnreadable(f"archive: {len(manifest['roots'])} roots, {len(roots)} given")

        def name_of(raw_name: str) -> str:
            name = _clean_relative(raw_name)
            head = name.split("/", 1)[0]
            if not head.isdigit() or str(int(head)) != head or int(head) >= len(roots):
                raise StateUnreadable(f"archive: {raw_name!r} does not name a root")
            return name

        members, top, symlinks = _vet(infos[1:], name_of, roots)
        budget = _Budget()
        for name, node in symlinks.items():
            _check_link(name, node, top, budget, contained_links)

        limit, used, out = len(data) + 1024 * 1024, 0, []
        for index, root in enumerate(roots):
            prefix = str(index)
            own = [(name, info) for name, info in members if name.split("/", 1)[0] == prefix]
            top = next((info for name, info in own if name == prefix), None)
            flat = top is not None and not top.isdir()
            directory = posixpath.dirname(root) if flat else root

            def rename(name: str, flat: bool = flat, root: str = root) -> str:
                return posixpath.basename(root) if flat else name.split("/", 1)[1]

            buffer = _Bounded(limit - used, "the rebuilt archive")
            with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name, info in own:
                    if name == prefix and info.isdir():
                        continue  # the root directory itself is made by the extractor
                    clean = _clean_member(name, info, rename(name),
                                          lambda target, rename=rename: rename(name_of(target)))
                    _copy(source, tar, info, clean)
            used += buffer.tell()
            out.append((root, directory, buffer.getvalue()))
        return out


def restored_link_paths(rebuilt: list[tuple[str, str, bytes]]) -> list[str]:
    """The absolute paths of the symlinks a `rebuild_for_restore` result
    extracts (a root restored as a symlink included)."""
    paths = []
    for _, directory, clean in rebuilt:
        with tarfile.open(fileobj=io.BytesIO(clean), mode="r:") as tar:
            paths.extend(posixpath.normpath(posixpath.join(directory, info.name))
                         for info in tar if info.issym())
    return paths
