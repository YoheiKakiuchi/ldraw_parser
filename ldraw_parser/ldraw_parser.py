"""
Parts parsing & instancing system per specification.

Directory layout:
 topDir/
   parts/  (partsDir)
   p/      (subDir)

Parsing phase:
 - Each part file parsed ONCE into immutable Pfile / Line and cached
     in Library.registry.
 - Registry key is lower-case basename.

Runtime phase:
 - SubLines / SubLine are mutable instance copies with world transforms.
 - FILE lines create child SubLines (recursive) using their 4x4 matrix.

Line formats:
 0 COMMENT
 0 !<META ...>                               -> META_COMMAND
 0 BFC (NOCERTIFY|CERTIFY ... / CW / CCW...) -> BFC_STATEMENT
 1 FILE : 1 <color> x y z a b c d e f g h i <filename>
 2 LINE : 2 <color> x1 y1 z1 x2 y2 z2
 3 TRI  : 3 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3
 4 QUAD : 4 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
 5 OPT  : 5 <color> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from pathlib import Path
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


@dataclass(frozen=True)
class BFC_STATEMENT:
    command: BFC_Command
    direction: Optional[BFC_Command] = None  # CW or CCW when applicable


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
        key = self._normalize(partsName)
        if key in self.registry:
            return self.registry[key]
        chosen: Optional[Path] = None
        for c in self._candidates(partsName):
            if c.exists():
                chosen = c.resolve()
                break
        if not chosen:
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
    CW_CCW: Optional[bool] = None
    INVERT: bool = False
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

    def getTriangles(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        for ln in self.lines:
            if ln.type == LineType.TRIANGLE and len(ln.data) == 3:
                acc.append(_world_pts(self.world, ln.data).reshape(1, 3, 3))
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 3, 3))

    def getQuads(self) -> np.ndarray:
        acc: List[np.ndarray] = []
        for ln in self.lines:
            if (
                ln.type in (LineType.QUAD, LineType.OPTIONAL)
                and len(ln.data) == 4
            ):
                acc.append(_world_pts(self.world, ln.data).reshape(1, 4, 3))
        return np.concatenate(acc, axis=0) if acc else np.zeros((0, 4, 3))

    def getLocations(self) -> List[Tuple[np.ndarray, str]]:
        out: List[Tuple[np.ndarray, str]] = []
        for ln in self.lines:
            out.extend(ln.locations)
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
    pf: Pfile, lib: Library, world: Optional[np.ndarray] = None
) -> SubLines:
    if world is None:
        world = np.eye(4)
    inst = SubLines(name=pf.name, id=uuid.uuid4().hex, lines=[], world=world)
    for ln in pf.lines:
        sub = SubLine(
            type=ln.type,
            rawdata=ln.rawdata,
            file=ln.file if ln.type == LineType.FILE else None,
            color=ln.color,
            T=ln.T.copy() if ln.T is not None else None,
            data=tuple(np.array(p) for p in ln.data),
            parent=inst
        )
        inst.lines.append(sub)
    for sub in inst.lines:
        if sub.type == LineType.FILE and sub.T is not None:
            tokens = sub.rawdata.split()
            if tokens:
                fname = tokens[-1]
                child_pf = lib.loadParts(fname)
                if child_pf:
                    child_world = inst.world @ sub.T
                    sub.child = instantiate(child_pf, lib, child_world)
    for sub in inst.lines:
        sub.updateLocations()
    return inst


def gather_triangles(root: SubLines) -> np.ndarray:
    acc: List[np.ndarray] = []

    def dfs(node: SubLines):
        acc.append(node.getTriangles())
        for ln in node.lines:
            if ln.child:
                dfs(ln.child)
    dfs(root)
    if any(a.size for a in acc):
        return np.concatenate([a for a in acc if a.size], axis=0)
    return np.zeros((0, 3, 3))


def gather_quads(root: SubLines) -> np.ndarray:
    acc: List[np.ndarray] = []

    def dfs(node: SubLines):
        acc.append(node.getQuads())
        for ln in node.lines:
            if ln.child:
                dfs(ln.child)
    dfs(root)
    if any(a.size for a in acc):
        return np.concatenate([a for a in acc if a.size], axis=0)
    return np.zeros((0, 4, 3))


def gather_locations(root: SubLines) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []

    def dfs(node: SubLines):
        out.extend(node.getLocations())
        for ln in node.lines:
            if ln.child:
                dfs(ln.child)
    dfs(root)
    return out


# ------------------------------------------------------------- Helpers ----
def _world_pts(world: np.ndarray, pts: Tuple[np.ndarray, ...]) -> np.ndarray:
    arr = np.vstack(pts)
    homog = np.hstack([arr, np.ones((arr.shape[0], 1))])
    return (world @ homog.T).T[:, :3]


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
    direction_map = {'CW': BFC_Command.CW, 'CCW': BFC_Command.CCW}
    first = tokens[0].upper()
    cmd = primary_map.get(first, BFC_Command.INVALID)
    direction: Optional[BFC_Command] = None
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
            "triangles": int(gather_triangles(r).shape[0]),
            "quads": int(gather_quads(r).shape[0]),
            "locations": len(gather_locations(r)),
            "lines": len(r.lines)
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _demo()
