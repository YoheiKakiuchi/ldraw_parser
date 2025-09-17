"""ldraw_parser: Minimal LDraw-like part file parser & instancing system.

Specification Summary (from user-provided design):
-------------------------------------------------
Directory layout:
    topDir/
        parts/  (partsDir)
        p/      (subDir)
All referenced part files are expected beneath either ``partsDir`` or ``subDir``.

Parsing phase:
    * Each part file is parsed exactly once into immutable ``Pfile`` and ``Line`` objects.
    * Objects are cached in ``Library.registry`` keyed by the lowercase basename.

Runtime phase:
    * ``SubLines`` / ``SubLine`` are mutable instance copies carrying world transforms.
    * A FILE (LineType==FILE) spawns a child ``SubLines`` using its 4x4 transformation matrix.

Line types (numeric code at line start):
 0 COMMENT
 1 FILE      -> 1 <color> x y z a b c d e f g h i <filename>
 2 LINE      -> 2 <color> x1 y1 z1 x2 y2 z2
 3 TRIANGLE  -> 3 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3
 4 QUAD      -> 4 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
 5 OPTIONAL  -> 5 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 (treated separately from QUAD)

Special zero (0) lines:
    * Comment: ``0 // <comment>`` or ``0 <comment>``
    * Meta command: ``0 !<META ...>``  -> LineType.META_COMMAND
    * BFC statement: ``0 BFC ...`` -> LineType.BFC_STATEMENT with commands:
                NOCERTIFY | CERTIFY [CW|CCW]
                (CW|CCW) | CLIP [CW|CCW] | NOCLIP | INVERTNEXT

FILE transform layout (matrix columns a..i and translation x,y,z):
        / a b c x \
        | d e f y |  -> stored as 4x4 homogeneous matrix T
        | g h i z |
        \ 0 0 0 1 /

Data storage per geometry line:
    * LINE:      two 3D points
    * TRIANGLE:  three 3D points
    * QUAD:      four 3D points
    * OPTIONAL:  four 3D points (semantic differs from QUAD)

BFC Handling:
    * A running orientation state (CW / CCW) may be set by BFC commands (CERTIFY, CW, CCW, CLIP w/ direction).
    * INVERTNEXT flags the next non-BFC geometry/FILE line; stored in ``SubLine.INVERT``.
    * Each non-BFC geometry/FILE ``SubLine`` records effective winding in ``SubLine.CW_CCW``.

Recursive Accessors:
    ``SubLines`` provides getTriangles/getQuads/getLines/getOptionals/getLocations including descendants.

Helper gather_* functions remain for backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np

# ------------------------------------------------------------------ Enum ----


class LineType(Enum):
    COMMENT = 0
    FILE = 1
    LINE = 2
    TRIANGLE = 3
    QUAD = 4
    OPTIONAL = 5
    META_COMMAND = 6
    BFC_STATEMENT = 7
    INVALID = -1

    @classmethod
    def from_int(cls, v: int) -> "LineType":
        return {m.value: m for m in cls}.get(v, cls.INVALID)


# -------------------------------------------------- BFC definitions ----
class BFC_Command(Enum):
    CW = 1
    CCW = 2
    NOCERTIFY = 3
    CERTIFY = 4
    CLIP = 5
    NOCLIP = 6
    INVERTNEXT = 7
    INVALID = -1


class CW_CCW(Enum):
    CW = 1
    CCW = 2
    INVALID = -1


@dataclass(frozen=True)
class BFC_STATEMENT:
    command: BFC_Command
    direction: Optional[CW_CCW] = None  # CW or CCW when applicable


# ------------------------------------------------------ Immutable parsed ----
@dataclass(frozen=True)
class Line:
    type: LineType
    rawdata: str
    file: Optional["Pfile"]  # Only for FILE lines, else None
    color: Optional[str] = None
    T: Optional[np.ndarray] = None              # For FILE lines
    data: Tuple[np.ndarray, ...] = tuple()      # Geometry points (local)
    bfc: Optional[BFC_STATEMENT] = None         # For BFC_STATEMENT lines


@dataclass(frozen=True)
class Pfile:
    name: str                # part filename (basename)
    file: str                # absolute path
    lines: Tuple["Line", ...]  # immutable lines


# --------------------------------------------------------------- Library ----
@dataclass
class Library:
    topDir: Path
    partsDir: Path
    subDir: Path
    # key = basename.lower()
    registry: Dict[str, Pfile] = field(default_factory=dict)

    @classmethod
    def create(cls, topDir: str | Path) -> "Library":
        top = Path(topDir).resolve()
        return cls(topDir=top, partsDir=top / "parts", subDir=top / "p")

    def _normalize(self, partName: str) -> str:
        return Path(partName).name.lower()

    def _candidates(self, partName: str) -> List[Path]:
        pn = Path(partName)
        if pn.is_absolute():
            return [pn]
        return [
            self.partsDir / pn,
            self.subDir / pn,
            self.topDir / pn
        ]

    def loadParts(self, partsName: str) -> Optional[Pfile]:
        partsName = PureWindowsPath(partsName).as_posix()
        key = self._normalize(partsName)
        if key in self.registry:
            return self.registry[key]
        chosen: Optional[Path] = None
        for c in self._candidates(partsName):
            if c.exists():
                chosen = c.resolve()
                break
        if not chosen:
            print('Warning: {} is not found'.format(partsName))
            return None
        temp_pf = Pfile(name=chosen.name, file=str(chosen), lines=tuple())
        # register stub early for recursive references
        self.registry[key] = temp_pf
        parsed: List[Line] = []
        with chosen.open("r", encoding="utf-8") as f:
            for raw in f:
                ln = _parse_line(raw.rstrip("\n"), temp_pf, self)
                if ln:
                    parsed.append(ln)
        final = Pfile(
            name=chosen.name,
            file=str(chosen),
            lines=tuple(
                Line(
                    type=ln.type,
                    rawdata=ln.rawdata,
                    # child Pfile for FILE lines, None otherwise
                    file=ln.file,
                    color=ln.color,
                    T=ln.T.copy() if ln.T is not None else None,
                    data=tuple(np.array(p) for p in ln.data),
                    bfc=ln.bfc,
                )
                for ln in parsed
            ),
        )
        self.registry[key] = final
        return final


# ------------------------------------------------------- Runtime copies ----
@dataclass
class SubLine:
    type: LineType
    rawdata: str
    file: Pfile
    color: Optional[str] = None
    T: Optional[np.ndarray] = None
    data: Tuple[np.ndarray, ...] = tuple()
    parent: "SubLines" = field(repr=False, default=None)
    # Orientation / BFC related
    CW_CCW: Optional[CW_CCW] = None  # Effective winding state at this line
    INVERT: bool = False  # True if preceded by INVERTNEXT
    bfc: Optional[BFC_STATEMENT] = None  # If this subline came from a BFC statement (LineType.BFC_STATEMENT)
    child: Optional["SubLines"] = None
    locations: List[Tuple[np.ndarray, str]] = field(default_factory=list)

    def updateLocations(self):
        self.locations.clear()
        # Only FILE lines have a location: (world transform, child Pfile.name)
        if self.type == LineType.FILE and self.child:
            pose = self.child.world.copy()
            self.locations.append((pose, self.child.name))


@dataclass
class SubLines:
    name: str
    id: str
    lines: List[SubLine]
    world: np.ndarray  # 4x4
    # Recursive accessors (include this node and all descendants)
    def getTriangles(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        # Collect own
        for ln in self.lines:
            if ln.type == LineType.TRIANGLE and len(ln.data) == 3:
                pts = ln.data
                # Reverse rule: (not INVERT and CW) or (INVERT and CCW)
                if ln.CW_CCW is not None:
                    reverse_needed = ((not ln.INVERT and ln.CW_CCW == CW_CCW.CW) or
                                      (ln.INVERT and ln.CW_CCW == CW_CCW.CCW))
                    ## mirror (left-handed coords)
                    if np.linalg.det(self.world[:3,:3]) < 0:
                        reverse_needed = not reverse_needed
                    if reverse_needed:
                        pts = _invert_points(pts, LineType.TRIANGLE)
                acc.append(_world_pts(self.world, pts).reshape(1, 3, 3))
        # Recurse
        for ln in self.lines:
            if ln.child:
                child_tris = ln.child.getTriangles()
                if child_tris.size:
                    acc.append(child_tris)
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 3, 3))

    def getQuads(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        for ln in self.lines:
            if ln.type == LineType.QUAD and len(ln.data) == 4:
                pts = ln.data
                if ln.CW_CCW is not None:
                    reverse_needed = ((not ln.INVERT and ln.CW_CCW == CW_CCW.CW) or
                                      (ln.INVERT and ln.CW_CCW == CW_CCW.CCW))
                    ## mirror (left-handed coords)
                    if np.linalg.det(self.world[:3,:3]) < 0:
                        reverse_needed = not reverse_needed
                    if reverse_needed:
                        pts = _invert_points(pts, LineType.QUAD)
                acc.append(_world_pts(self.world, pts).reshape(1, 4, 3))
        for ln in self.lines:
            if ln.child:
                child_quads = ln.child.getQuads()
                if child_quads.size:
                    acc.append(child_quads)
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 4, 3))

    def getLines(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        for ln in self.lines:
            if ln.type == LineType.LINE and len(ln.data) == 2:
                acc.append(_world_pts(self.world, ln.data).reshape(1, 2, 3))
        for ln in self.lines:
            if ln.child:
                child_lines = ln.child.getLines()
                if child_lines.size:
                    acc.append(child_lines)
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 2, 3))

    def getOptionals(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        for ln in self.lines:
            if ln.type == LineType.OPTIONAL and len(ln.data) == 4:
                acc.append(_world_pts(self.world, ln.data).reshape(1, 4, 3))
        for ln in self.lines:
            if ln.child:
                child_opts = ln.child.getOptionals()
                if child_opts.size:
                    acc.append(child_opts)
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 4, 3))

    def getLocations(self) -> List[Tuple[np.ndarray, str]]:
        out: List[Tuple[np.ndarray, str]] = []
        for ln in self.lines:
            out.extend(ln.locations)
        for ln in self.lines:
            if ln.child:
                out.extend(ln.child.getLocations())
        return out


# ---------------------------------------------------- Parsing internals ----
def _parse_line(text: str, pf: Pfile, lib: "Library") -> Optional[Line]:
    if not text.strip():
        return None
    toks = text.strip().split()
    try:
        code = int(toks[0])
    except ValueError:
        return None
    rest = toks[1:]
    if code == 0:
        if not rest:
            return Line(type=LineType.COMMENT, rawdata="", file=None)
        first = rest[0]
        if first.startswith("!"):
            return Line(
                type=LineType.META_COMMAND,
                rawdata=" ".join(rest),
                file=None,
            )
        if first.upper() == "BFC":
            bfc_stmt = _parse_bfc(rest[1:])
            return Line(
                type=LineType.BFC_STATEMENT,
                rawdata=" ".join(rest),
                file=None,
                bfc=bfc_stmt,
            )
        return Line(type=LineType.COMMENT, rawdata=" ".join(rest), file=None)

    ltype = LineType.from_int(code)
    if ltype == LineType.FILE:
        if len(rest) < 14:
            return None
        color = rest[0]
        mat_vals = rest[1:13]
        fname = rest[13]
        try:
            nums = [float(v) for v in mat_vals]
        except ValueError:
            return None
        x, y, z = nums[0:3]
        a, b, c, d, e, f_, g, h, i = nums[3:12]
        T = np.eye(4)
        T[0, 0:3] = [a, b, c]
        T[1, 0:3] = [d, e, f_]
        T[2, 0:3] = [g, h, i]
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        child_pf = lib.loadParts(fname) if fname else None
        return Line(
            type=ltype,
            rawdata=" ".join(rest),
            file=child_pf,
            color=color,
            T=T,
            data=tuple(),
        )

    if ltype in (
        LineType.LINE,
        LineType.TRIANGLE,
        LineType.QUAD,
        LineType.OPTIONAL,
    ):
        if not rest:
            return None
        color = rest[0]
        coord_tokens = rest[1:]
        try:
            nums = [float(v) for v in coord_tokens]
        except ValueError:
            return None
        needed = {
            LineType.LINE: 2,
            LineType.TRIANGLE: 3,
            LineType.QUAD: 4,
            LineType.OPTIONAL: 4
        }[ltype]
        if len(nums) < needed * 3:
            return None
        pts = [
            np.array(nums[i:i + 3], dtype=float)
            for i in range(0, needed * 3, 3)
        ]
        return Line(
            type=ltype,
            rawdata=" ".join(rest),
            file=None,
            color=color,
            data=tuple(pts),
        )
    return None


# ------------------------------------------------------ Instancing API ----
def instantiate(
    pf: Pfile,
    lib: Library,
    world: Optional[np.ndarray] = None,
    invert: bool = False,
) -> SubLines:
    """Instantiate a parsed part file into a mutable hierarchy.

    Parameters
    ----------
    pf : Pfile
        Parsed immutable part.
    lib : Library
        Library for recursive loads.
    world : np.ndarray, optional
        4x4 world matrix for this root (defaults to identity).
    invert : bool, default False
        Incoming inversion state from parent. Effective inversion for a
        geometry / FILE line is computed as (parent invert) XOR (pending
        INVERTNEXT). When effective inversion is True, TRIANGLE/QUAD/OPTIONAL
        vertex order is reversed at instancing time and the SubLine.INVERT
        flag is set.
    """
    if world is None:
        world = np.eye(4)
    inst = SubLines(name=pf.name, id=uuid.uuid4().hex, lines=[], world=world)
    # Track current winding orientation (None until certified)
    current_winding: Optional[CW_CCW] = None
    invert_next = False  # Tracks pending INVERTNEXT for next geometry/FILE
    for ln in pf.lines:
        # Create SubLine
        sub = SubLine(
            type=ln.type,
            rawdata=ln.rawdata,
            file=ln.file if ln.type == LineType.FILE else None,
            color=ln.color,
            T=ln.T.copy() if ln.T is not None else None,
            data=tuple(np.array(p) for p in ln.data),
            parent=inst,
        )
        # Attach BFC statement meta if present
        if ln.type == LineType.BFC_STATEMENT and ln.bfc:
            sub.bfc = ln.bfc
            cmd = ln.bfc.command
            # Adjust state machine
            if cmd == BFC_Command.INVERTNEXT:
                invert_next = True
            elif cmd in (BFC_Command.CW, BFC_Command.CCW):
                current_winding = CW_CCW.CW if cmd == BFC_Command.CW else CW_CCW.CCW
            elif cmd in (BFC_Command.CERTIFY, BFC_Command.NOCERTIFY):
                # If CERTIFY/NOCERTIFY may include direction
                if ln.bfc.direction:
                    current_winding = ln.bfc.direction
            elif cmd == BFC_Command.CLIP:
                # Optional direction update
                if ln.bfc.direction:
                    current_winding = ln.bfc.direction
            # NOCLIP does not change winding
        # Apply orientation flags to geometry carrying lines (except pure BFC statements)
        if ln.type not in (LineType.BFC_STATEMENT, LineType.META_COMMAND, LineType.COMMENT):
            sub.CW_CCW = current_winding
            # Effective inversion for this line = parent invert XOR invert_next
            sub.INVERT = invert ^ invert_next
            # Consume invert_next only once for the next geometry/FILE line
            if invert_next:
                invert_next = False
        inst.lines.append(sub)
    for sub in inst.lines:
        if sub.type == LineType.FILE and sub.T is not None:
            tokens = sub.rawdata.split()
            if tokens:
                fname = tokens[-1]
                child_pf = lib.loadParts(fname)
                if child_pf:
                    child_world = inst.world @ sub.T
                    # Propagate inversion state into child instantiation
                    sub.child = instantiate(child_pf, lib, child_world, invert=sub.INVERT)
    for sub in inst.lines:
        sub.updateLocations()
    return inst


def gather_triangles(root: SubLines) -> np.ndarray:
    return root.getTriangles()


def gather_quads(root: SubLines) -> np.ndarray:
    return root.getQuads()


def gather_locations(root: SubLines) -> List[Tuple[np.ndarray, str]]:
    return root.getLocations()


def gather_lines(root: SubLines) -> np.ndarray:
    return root.getLines()


def gather_optionals(root: SubLines) -> np.ndarray:
    return root.getOptionals()


# ------------------------------------------------------------- Helpers ----
def _world_pts(world: np.ndarray, pts: Tuple[np.ndarray, ...]) -> np.ndarray:
    arr = np.vstack(pts)
    homog = np.hstack([arr, np.ones((arr.shape[0], 1))])
    return (world @ homog.T).T[:, :3]


def _invert_points(pts: Tuple[np.ndarray, ...], ltype: LineType) -> Tuple[np.ndarray, ...]:
    """Return a new tuple of points with winding inverted.

    For TRIANGLE: swap v1 & v3 (reverse order) -> (p0, p2, p1)
    For QUAD / OPTIONAL: swap second and fourth to keep adjacent edges: (p0, p3, p2, p1)
    This mirrors typical LDraw winding inversion used when applying INVERTNEXT.
    """
    if not pts:
        return pts
    if ltype == LineType.TRIANGLE and len(pts) == 3:
        p0, p1, p2 = pts
        return (p0, p2, p1)
    if ltype in (LineType.QUAD, LineType.OPTIONAL) and len(pts) == 4:
        p0, p1, p2, p3 = pts
        return (p0, p3, p2, p1)
    # Fallback generic reversal
    return tuple(reversed(pts))  # type: ignore




def _parse_bfc(tokens: List[str]) -> Optional[BFC_STATEMENT]:
    """Parse tokens after the 'BFC' keyword.
    tokens examples:
      ['NOCERTIFY']
      ['CERTIFY']
      ['CERTIFY','CW']
      ['CW']
      ['CLIP']
      ['CLIP','CCW']
      ['NOCLIP']
      ['INVERTNEXT']
    """
    if not tokens:
        return None
    primary_map = {
        'NOCERTIFY': BFC_Command.NOCERTIFY,
        'CERTIFY': BFC_Command.CERTIFY,
        'CW': BFC_Command.CW,
        'CCW': BFC_Command.CCW,
        'CLIP': BFC_Command.CLIP,
        'NOCLIP': BFC_Command.NOCLIP,
        'INVERTNEXT': BFC_Command.INVERTNEXT,
    }
    direction_map = {'CW': CW_CCW.CW, 'CCW': CW_CCW.CCW}
    first = tokens[0].upper()
    cmd = primary_map.get(first, BFC_Command.INVALID)
    direction: Optional[CW_CCW] = None
    if cmd in (BFC_Command.CERTIFY, BFC_Command.NOCERTIFY, BFC_Command.CLIP):
        if len(tokens) > 1:
            direction = direction_map.get(tokens[1].upper())
    # A lone CW/CCW acts as command with no extra direction
    return BFC_STATEMENT(command=cmd, direction=direction)


# --------------------------------------------------------------- CLI ----
def _demo():  # pragma: no cover
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Parse and instance parts.")
    parser.add_argument("topDir")
    parser.add_argument("parts", nargs="+")
    args = parser.parse_args()
    lib = Library.create(args.topDir)
    roots = []
    for name in args.parts:
        pf = lib.loadParts(name)
        if pf:
            roots.append(instantiate(pf, lib))
    report = []
    for r in roots:
        report.append({
            "part": r.name,
            "triangles": int(r.getTriangles().shape[0]),
            "quads": int(r.getQuads().shape[0]),
            "lines": int(r.getLines().shape[0]),
            "optionals": int(r.getOptionals().shape[0]),
            "locations": len(r.getLocations()),
            "top_level_lines": len(r.lines)
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _demo()
