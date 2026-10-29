"""
Build the manifest of OpenQuake verification tables for djura's GMMs.

OpenQuake verifies each GMM with a test case calling ``self.check(...)`` on
CSV tables. The test modules are parsed, never executed: a small evaluator
follows literals, string templates, module helpers returning an expression,
``for`` loops and class attributes, and records each ``check`` call with its
GMM, keyword arguments, tables and tolerances. Calls it cannot follow are
listed, as are calls for GMMs that djura does not implement.

    python tests/rs/gsim_verification/generate_manifest.py [--tag v3.23.0]
"""

import argparse
import ast
import itertools
import json
import operator
import time
import urllib.error
import urllib.request
from pathlib import Path

from djura.record_selection.gsim import models

#: Version djura's gsim subpackage was ported from, see gsim/NOTICE.md
DEFAULT_TAG = "v3.23.0"
REPO = "gem/oq-engine"
TESTS_PATH = "openquake/hazardlib/tests/gsim"
MANIFEST = Path(__file__).resolve().parent / "oq_gsim_tables.json"

_BINARY = {ast.Add: operator.add, ast.Mod: operator.mod,
           ast.Mult: operator.mul, ast.Sub: operator.sub,
           ast.Div: operator.truediv}
_ITERATORS = {"zip": zip, "range": range, "enumerate": enumerate,
              "itertools.product": itertools.product}

#: Regions OpenQuake selects with an argument and djura with a subclass
REGIONAL_CLASSES = {
    "ChiouYoungs2014": {"JPN": "ChiouYoungs2014Japan",
                        "ITA": "ChiouYoungs2014Italy",
                        "WEN": "ChiouYoungs2014Wenchuan"},
}


def _campbell_bozorgnia():
    """Checks of campbell_bozorgnia_2014_test.py::test_all, written out

    OpenQuake runs them from a module-level parametrised function, passing
    its Q coefficient tables and Japan flag as arguments, where djura has the
    HighQ, LowQ and JapanSite subclasses
    """
    checks = []
    for year in ("2014", "2019"):
        for tag, q_class in (("", ""), ("_HIGHQ", "HighQ"),
                             ("_LOWQ", "LowQ")):
            for japan in (False, True):
                name = tag + ("_JAPAN" if japan else "")
                checks.append({
                    "gsim": f"CampbellBozorgnia{year}{q_class}"
                            + ("JapanSite" if japan else ""),
                    "kwargs": {},
                    "tables": [f"CB{year[2:]}/CB{year}{name}_{kind}.csv"
                               for kind in ("MEAN", "STD_INTRA",
                                            "STD_INTER", "STD_TOTAL")],
                    "mean_tolerance": 0.1,
                    "std_tolerance": 0.1,
                    "oq_test": "campbell_bozorgnia_2014_test.py::test_all",
                })
    return checks


class Unresolved(Exception):
    """A construct outside what the evaluator follows"""


class GsimRef(str):
    """Name of a class imported from OpenQuake"""


class ModuleRef(str):
    """An OpenQuake module imported under an alias"""


class _Module:
    """Evaluates the test cases of one OpenQuake test module"""

    def __init__(self, source, name):
        self.name = name
        self.tree = ast.parse(source)
        self.globals = {}
        self.functions = {}
        self.classes = {}
        self.tainted = set()
        for node in self.tree.body:
            self._statement(node)

    def _statement(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.globals[alias.asname or alias.name] = \
                    ModuleRef(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                self.globals[alias.asname or alias.name] = (
                    GsimRef(alias.name)
                    if (node.module or "").startswith("openquake")
                    else ModuleRef(alias.name))
        elif isinstance(node, ast.Assign):
            try:
                value = self.eval(node.value, self.globals)
            except Unresolved:
                value = None
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if value is None:
                        self.globals.pop(target.id, None)
                        self.tainted.add(target.id)
                    else:
                        self.globals[target.id] = value
                        self.tainted.discard(target.id)
                else:
                    # e.g. d[key] = value, mutating a module-level name
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name):
                            self.globals.pop(child.id, None)
                            self.tainted.add(child.id)
        elif isinstance(node, ast.FunctionDef):
            self.functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            self.classes[node.name] = node
        else:
            # A loop or call may change what a name holds, so any name it
            # touches can no longer be trusted
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    self.globals.pop(child.id, None)
                    self.tainted.add(child.id)

    def eval(self, node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.tainted:
                raise Unresolved(node.id)
            if node.id in self.globals:
                return self.globals[node.id]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise Unresolved(node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.eval(item, env) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {self.eval(key, env): self.eval(value, env)
                    for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.eval(node.operand, env)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self.eval(node.left, env),
                                          self.eval(node.right, env))
        if isinstance(node, ast.JoinedStr):
            return "".join(
                str(self.eval(part.value, env))
                if isinstance(part, ast.FormattedValue)
                else part.value for part in node.values)
        if isinstance(node, ast.Subscript):
            return self.eval(node.value, env)[self.eval(node.slice, env)]
        if isinstance(node, ast.Attribute):
            owner = self.eval(node.value, env)
            if isinstance(owner, ModuleRef):
                return GsimRef(node.attr)
            if isinstance(owner, dict) and node.attr in owner:
                return owner[node.attr]
            raise Unresolved(ast.unparse(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in self.functions:
            return self._call(self.functions[node.func.id], node, env)
        if isinstance(node, ast.Call) and not node.keywords and \
                ast.unparse(node.func) in _ITERATORS:
            return tuple(_ITERATORS[ast.unparse(node.func)](
                *(self.eval(arg, env) for arg in node.args)))
        raise Unresolved(ast.unparse(node))

    def _call(self, function, node, env):
        body = [stmt for stmt in function.body
                if not isinstance(stmt, ast.Expr)]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            raise Unresolved(function.name)
        names = [arg.arg for arg in function.args.args]
        local = dict(zip(names, (self.eval(arg, env) for arg in node.args)))
        local.update({kw.arg: self.eval(kw.value, env)
                      for kw in node.keywords})
        return self.eval(body[0].value, local)

    def _bases(self, case):
        """The case and its ancestors defined in the module, base first"""
        chain = []
        for base in case.bases:
            if isinstance(base, ast.Name) and base.id in self.classes:
                chain += self._bases(self.classes[base.id])
        return chain + [case]

    def _is_test_case(self, case):
        names = {ast.unparse(base) for case in self._bases(case)
                 for base in case.bases}
        return any(name.endswith("BaseGSIMTestCase") for name in names)

    def checks(self):
        """Every check call of every test case, and what was not followed"""
        calls, skipped = [], []
        for case in self.classes.values():
            if not self._is_test_case(case):
                continue
            attributes, methods = {}, {}
            for ancestor in self._bases(case):
                for node in ancestor.body:
                    if isinstance(node, ast.Assign):
                        try:
                            value = self.eval(node.value, attributes)
                        except Unresolved:
                            continue
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                attributes[target.id] = value
                    elif isinstance(node, ast.FunctionDef):
                        methods[node.name] = node
            for name, method in methods.items():
                if not name.startswith("test"):
                    continue
                env = {"self": dict(attributes)}
                try:
                    self._run(method.body, env, calls, case.name)
                except (Unresolved, KeyError, TypeError) as error:
                    skipped.append(f"{self.name}::{case.name}.{name} "
                                   f"({type(error).__name__}: {error})")
        return calls, skipped

    def _run(self, body, env, calls, case):
        for node in body:
            if isinstance(node, ast.For):
                for value in self.eval(node.iter, env):
                    local = dict(env)
                    self._bind(node.target, value, local)
                    self._run(node.body, local, calls, case)
            elif isinstance(node, ast.Assign):
                value = self.eval(node.value, env)
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and \
                            ast.unparse(target.value) == "self":
                        env["self"][target.attr] = value
                    else:
                        self._bind(target, value, env)
            elif isinstance(node, ast.Expr) and \
                    isinstance(node.value, ast.Call) and \
                    ast.unparse(node.value.func) == "self.check":
                calls.append(self._check(node.value, env, case))
            elif isinstance(node, (ast.Expr, ast.Pass)):
                continue
            else:
                raise Unresolved(type(node).__name__)

    def _bind(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item, part in zip(target.elts, value):
                self._bind(item, part, env)
        else:
            raise Unresolved(ast.unparse(target))

    def _check(self, node, env, case):
        tables = [self.eval(arg, env) for arg in node.args]
        kwargs = {kw.arg: self.eval(kw.value, env) for kw in node.keywords}
        mean = kwargs.pop("max_discrep_percentage")
        std = kwargs.pop("std_discrep_percentage", mean)
        kwargs.pop("truncation_level", None)
        gsim = env["self"].get("GSIM_CLASS")
        if not isinstance(gsim, GsimRef):
            raise Unresolved("GSIM_CLASS")
        return {"gsim": str(gsim), "kwargs": kwargs, "tables": tables,
                "mean_tolerance": mean, "std_tolerance": std,
                "oq_test": f"{self.name}::{case}"}


def _fetch(url, attempts=5):
    # GitHub's raw host intermittently fails the TLS handshake
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _test_modules(tag):
    url = (f"https://api.github.com/repos/{REPO}/contents/{TESTS_PATH}"
           f"?ref={tag}")
    return sorted(item["name"] for item in json.loads(_fetch(url))
                  if item["name"].endswith("_test.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tag", default=DEFAULT_TAG)
    tag = parser.parse_args().tag
    raw = f"https://raw.githubusercontent.com/{REPO}/{tag}/{TESTS_PATH}"

    entries, unmatched, skipped = _campbell_bozorgnia(), set(), []
    for name in _test_modules(tag):
        source = _fetch(f"{raw}/{name}").decode("utf-8")
        calls, missed = _Module(source, name).checks()
        skipped += missed
        for call in calls:
            regional = REGIONAL_CLASSES.get(call["gsim"], {})
            if call["kwargs"].get("region") in regional:
                call["gsim"] = regional[call["kwargs"].pop("region")]
            if not hasattr(models, call["gsim"]):
                unmatched.add(call["gsim"])
            elif call not in entries:
                entries.append(call)

    MANIFEST.write_text(json.dumps(
        {"tag": tag, "entries": entries}, separators=(",", ":")) + "\n",
        encoding="utf-8")
    print(f"{len(entries)} checks over "
          f"{len({e['gsim'] for e in entries})} djura GMMs -> {MANIFEST}")
    print(f"{len(unmatched)} OpenQuake GMMs not implemented in djura")
    for line in skipped:
        print("not followed:", line)


if __name__ == "__main__":
    main()
