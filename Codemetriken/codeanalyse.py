#!/usr/bin/env python3
from __future__ import annotations
"""
Minimalanalyse für Kapitel 6.3 – funktional-paradigmenneutrale Metriken

Ziel:
- Hotspots auf Funktionsebene (Zweigzahl/Zyklomatik, externe vs. interne Aufrufe)
- Modulkopplung (interne/externe Imports), Dichte (externe Imports je 100 SLOC)
- Optional: DOT-Graph der internen Importe (für Paket-/Komponentendiagramm)

Nicht enthalten: ausführliche Klassenmetriken (LCOM/TCC/CBO). Lediglich
funktionale Steuerung und Modulkopplung werden erhoben.

Nur Standardbibliothek erforderlich.
"""

import argparse
import ast
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------- Hilfsfunktionen -----------------------------

TEST_DIR_CANDIDATES = {"tests", "test"}


def sloc_count(src: str) -> int:
    count = 0
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        count += 1
    return count


def module_name_from_path(file: Path, root: Path) -> str:
    rel = file.resolve().relative_to(root.resolve())
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_py_files(root: Path, include_tests: bool = False) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if not include_tests and (p.name in TEST_DIR_CANDIDATES):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(p / fn)
    return sorted(files)


def rel_to_abs(module: str, current: str) -> str:
    if not module or module[0] != '.':
        return module
    i = 0
    while i < len(module) and module[i] == '.':
        i += 1
    up = i
    rest = module[i:]
    parts = current.split('.')
    if len(parts) < up:
        return module
    base = parts[: len(parts) - up]
    out = '.'.join([*base, rest] if rest else base)
    return out or current

# ----------------------------- Projektindex -------------------------------

@dataclass
class ImportMap:
    imported_modules: Set[str] = field(default_factory=set)
    alias_to_module: Dict[str, str] = field(default_factory=dict)
    alias_to_symbol: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    path: Path
    module: str
    top_level_funcs: Set[str] = field(default_factory=set)
    import_map: ImportMap = field(default_factory=ImportMap)
    sloc: int = 0


class ProjectIndex:
    def __init__(self, root: Path):
        self.root = root
        self.modules: Dict[str, ModuleInfo] = {}

    def build(self, files: List[Path]) -> None:
        for file in files:
            module = module_name_from_path(file, self.root)
            try:
                src = file.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(file))
            except Exception:
                continue
            mi = ModuleInfo(path=file, module=module, sloc=sloc_count(src))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    mi.top_level_funcs.add(node.name)
            imap = ImportMap()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        imap.imported_modules.add(mod)
                        if alias.asname:
                            imap.alias_to_module[alias.asname] = mod
                        else:
                            top = mod.split('.')[0]
                            imap.alias_to_module[top] = top
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    dots = node.level or 0
                    rel = ("." * dots + base) if dots else base
                    mod = rel_to_abs(rel, module)
                    if mod:
                        imap.imported_modules.add(mod)
                    for alias in node.names:
                        name = alias.name
                        tgt = f"{mod}.{name}" if mod else name
                        imap.alias_to_symbol[alias.asname or name] = tgt
            mi.import_map = imap
            self.modules[module] = mi

    @property
    def project_modules(self) -> Set[str]:
        return set(self.modules.keys())

    def resolve_symbol_origin(self, module: str, name: str) -> Optional[str]:
        mi = self.modules.get(module)
        if not mi:
            return None
        if name in mi.top_level_funcs:
            return module
        imap = mi.import_map
        if name in imap.alias_to_module:
            return imap.alias_to_module[name]
        if name in imap.alias_to_symbol:
            sym = imap.alias_to_symbol[name]
            return sym.rsplit('.', 1)[0] if '.' in sym else sym
        return None

    def is_internal_origin(self, origin_module: Optional[str]) -> bool:
        if not origin_module:
            return False
        if origin_module in self.project_modules:
            return True
        top = origin_module.split('.')[0]
        return top in {m.split('.')[0] for m in self.project_modules}

# ----------------------------- Funktionsanalyse ---------------------------

@dataclass
class FunctionMetrics:
    module: str
    qualname: str
    params: int
    cyclomatic: int
    calls_internal_funcs: int
    calls_external_funcs: int


class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self, project: ProjectIndex, module: str):
        self.project = project
        self.module = module
        self.calls_internal = 0
        self.calls_external = 0
        self.cyclomatic = 1

    def generic_visit(self, node):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.IfExp)):
            self.cyclomatic += 1
        elif isinstance(node, ast.BoolOp):
            self.cyclomatic += max(0, len(getattr(node, 'values', [])) - 1)
        elif isinstance(node, (ast.comprehension,)):
            self.cyclomatic += 1
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call):
        target = node.func
        origin = None
        if isinstance(target, ast.Name):
            origin = self.project.resolve_symbol_origin(self.module, target.id)
        elif isinstance(target, ast.Attribute):
            cur = target
            while isinstance(cur, ast.Attribute):
                base = cur.value
                cur = base
            if isinstance(cur, ast.Name):
                origin = self.project.resolve_symbol_origin(self.module, cur.id)
        if self.project.is_internal_origin(origin):
            self.calls_internal += 1
        else:
            self.calls_external += 1
        self.generic_visit(node)


def analyze_functions(mi: ModuleInfo, project: ProjectIndex) -> List[FunctionMetrics]:
    try:
        src = mi.path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(mi.path))
    except Exception:
        return []
    results: List[FunctionMetrics] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fa = FunctionAnalyzer(project, mi.module)
            fa.visit(node)
            results.append(
                FunctionMetrics(
                    module=mi.module,
                    qualname=f"{mi.module}.{node.name}",
                    params=len(node.args.args),
                    cyclomatic=fa.cyclomatic,
                    calls_internal_funcs=fa.calls_internal,
                    calls_external_funcs=fa.calls_external,
                )
            )
    return results

# ----------------------------- Importgraph --------------------------------

def build_import_graph(project: ProjectIndex, imports_by_module: Dict[str, ImportMap]):
    modules = set(project.project_modules)
    internal_graph: Dict[str, Set[str]] = {m: set() for m in modules}
    external_imports: Dict[str, Set[str]] = defaultdict(set)
    for mod, imap in imports_by_module.items():
        for base in imap.imported_modules:
            base_abs = rel_to_abs(base, mod)
            top = base_abs.split('.')[0] if base_abs else base_abs
            if base_abs in modules:
                internal_graph[mod].add(base_abs)
            elif top:
                external_imports[mod].add(top)
        for base_abs in imap.alias_to_module.values():
            if base_abs in modules:
                internal_graph[mod].add(base_abs)
            else:
                external_imports[mod].add(base_abs.split('.')[0])
        for sym in imap.alias_to_symbol.values():
            base_abs = sym.rsplit('.', 1)[0]
            if base_abs in modules:
                internal_graph[mod].add(base_abs)
            else:
                external_imports[mod].add(base_abs.split('.')[0])
    return internal_graph, external_imports

# ----------------------------- CLI / Orchestrierung ------------------------

def main():
    ap = argparse.ArgumentParser(description=(
        "Minimalanalyse: Funktions-Hotspots und Modulkopplung für Kapitel 6.3"
    ))
    ap.add_argument('path', help='Projektordner')
    ap.add_argument('--csv-dir', required=True, help='Zielordner für CSV-Tabellen')
    ap.add_argument('--dot', dest='dot_out', help='Optional: Graphviz-DOT-Datei für interne Importe')
    ap.add_argument('--json', dest='json_out', help='Optional: JSON-Sammelausgabe')
    ap.add_argument('--top', type=int, default=5, help='Top-N Hotspot-Funktionen ausgeben (Standard: 5)')
    ap.add_argument('--include-tests', action='store_true', help='tests/-Verzeichnisse einbeziehen')
    args = ap.parse_args()

    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Fehler: {root} ist kein Verzeichnis")
        raise SystemExit(2)

    files = discover_py_files(root, include_tests=args.include_tests)
    if not files:
        print("Keine Python-Dateien gefunden.")
        raise SystemExit(0)

    project = ProjectIndex(root)
    project.build(files)

    all_functions: List[FunctionMetrics] = []
    imports_by_module: Dict[str, ImportMap] = {}

    for file in files:
        module = module_name_from_path(file, root)
        mi = project.modules.get(module)
        if not mi:
            continue
        all_functions.extend(analyze_functions(mi, project))
        imports_by_module[module] = mi.import_map

    internal_graph, external_imports = build_import_graph(project, imports_by_module)

    # Modulkennzahlen
    indeg: Dict[str, int] = defaultdict(int)
    outdeg: Dict[str, int] = {m: len(neigh) for m, neigh in internal_graph.items()}
    for src, neigh in internal_graph.items():
        for dst in neigh:
            indeg[dst] += 1

    modules_rows = []
    for m, mi in project.modules.items():
        sloc = mi.sloc
        ext = len(external_imports.get(m, set()))
        inte = len(internal_graph.get(m, set()))
        eff = outdeg.get(m, 0)
        aff = indeg.get(m, 0)
        ext_per_100 = (ext / sloc * 100) if sloc > 0 else 0.0
        instability = (eff / (eff + aff)) if (eff + aff) > 0 else float('nan')
        modules_rows.append({
            "module": m,
            "sloc": sloc,
            "external_imports": ext,
            "internal_imports": inte,
            "efferent": eff,
            "afferent": aff,
            "ext_per_100_sloc": round(ext_per_100, 2),
            "instability": round(instability, 2) if instability == instability else "",
        })

    # Hotspot-Funktionen
    hotspots = sorted(all_functions, key=lambda x: (x.cyclomatic, x.calls_external_funcs), reverse=True)[: args.top]

    outdir = Path(args.csv_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV: functions_hotspots.csv
    with (outdir / "functions_hotspots.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "qualname", "params", "cyclomatic", "calls_internal_funcs", "calls_external_funcs"])
        for f in hotspots:
            w.writerow([f.module, f.qualname, f.params, f.cyclomatic, f.calls_internal_funcs, f.calls_external_funcs])

    # CSV: modules_overview.csv
    with (outdir / "modules_overview.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "sloc", "external_imports", "internal_imports", "efferent", "afferent", "ext_per_100_sloc", "instability"])
        for r in modules_rows:
            w.writerow([r["module"], r["sloc"], r["external_imports"], r["internal_imports"], r["efferent"], r["afferent"], r["ext_per_100_sloc"], r["instability"]])

    # DOT (optional)
    if args.dot_out:
        with Path(args.dot_out).open("w", encoding="utf-8") as fh:
            fh.write("digraph imports {\n  rankdir=LR;\n  node [shape=box];\n")
            for src, dsts in internal_graph.items():
                for dst in dsts:
                    fh.write(f'  "{src}" -> "{dst}";\n')
            fh.write("}\n")
        print(f"DOT-Graph gespeichert: {args.dot_out}")

    # JSON (optional)
    if args.json_out:
        payload = {
            "functions": [asdict(f) for f in all_functions],
            "hotspots": [asdict(f) for f in hotspots],
            "modules": modules_rows,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON gespeichert: {args.json_out}")

    # Konsole – Kurzbericht
    print("\n=== Hotspots (Top Funktionen) ===")
    for f in hotspots:
        print(f"- {f.qualname}: cyclomatic={f.cyclomatic}, extern={f.calls_external_funcs}, intern={f.calls_internal_funcs}")

    print("\n=== Module (Kopplung/Dichte) ===")
    for r in sorted(modules_rows, key=lambda x: (x["external_imports"], x["efferent"]), reverse=True)[:10]:
        print(f"- {r['module']}: extern={r['external_imports']}, intern={r['internal_imports']}, ext/100SLOC={r['ext_per_100_sloc']}, eff={r['efferent']}, aff={r['afferent']}")


if __name__ == '__main__':
    main()
