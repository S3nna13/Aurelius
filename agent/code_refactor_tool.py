"""AST-based Python refactoring tool for coding agents.

Pure stdlib (``ast`` + ``tokenize``). Never executes user code. Every
operation parses the input, mutates the AST safely, and re-emits source
via :func:`ast.unparse`. Callers may wrap calls in ``try/except
SyntaxError`` to recover from malformed input; this module itself
propagates parse errors verbatim.

Supported operations (all exposed as methods on :class:`CodeRefactorTool`):

* ``rename_symbol`` - rename a variable, function, or class. Scope may
  be ``"module"`` (default) or ``"function"`` (renames only within the
  first function that defines or uses the name).
* ``inline_variable`` - replace a single-assignment variable with its
  right-hand-side expression.
* ``remove_unused_imports`` - drop ``import`` / ``from ... import``
  targets whose bound names are never referenced.
* ``extract_function`` - lift a contiguous block of statements into a
  new top-level function; free variables become parameters.
* ``add_type_hint`` - annotate a single parameter of a named function.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field

__all__ = ["CodeRefactorTool", "RefactorResult"]


@dataclass
class RefactorResult:
    """Result of a refactor operation.

    Attributes
    ----------
    new_code:
        The transformed source. Equal to the input when no change was
        made (and :attr:`changes` is ``0``).
    operation:
        Short identifier of the operation that produced the result.
    changes:
        Count of distinct edits applied (e.g. renamed occurrences,
        imports removed).
    warnings:
        Human-readable warnings collected during the operation; empty
        when no caveats apply.
    """

    new_code: str
    operation: str
    changes: int
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unparse(tree: ast.AST) -> str:
    """Emit source for ``tree``; require Python with :func:`ast.unparse`.

    Python 3.9+ ships ``ast.unparse``; raise a clear error otherwise so
    older interpreters fail loudly rather than silently miscompiling.
    """

    if not hasattr(ast, "unparse"):  # pragma: no cover - 3.14 has unparse
        raise RuntimeError("ast.unparse requires Python 3.9+")
    return ast.unparse(tree)


def _collect_assignments(tree: ast.AST, name: str) -> list[ast.AST]:
    """Return every node that (re)binds ``name`` at module scope."""

    targets: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    targets.append(node)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                targets.append(node)
    return targets


# ---------------------------------------------------------------------------
# AST transformers
# ---------------------------------------------------------------------------


class _RenameTransformer(ast.NodeTransformer):
    """Rename every binding/reference of ``old`` to ``new``.

    Only identifier-bearing nodes are touched (Name, arg, FunctionDef,
    ClassDef, attribute-free keywords, ``global``/``nonlocal``). String
    literals and comments are left alone by construction: the AST does
    not represent their contents as identifiers.
    """

    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new
        self.changes = 0

    def _swap(self, current: str) -> str:
        if current == self.old:
            self.changes += 1
            return self.new
        return current

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._swap(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._swap(node.arg)
        return self.generic_visit(node) or node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._swap(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self._swap(node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._swap(node.name)
        self.generic_visit(node)
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self._swap(n) for n in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = [self._swap(n) for n in node.names]
        return node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CodeRefactorTool:
    """Collection of AST-driven refactor operations.

    Instances are stateless; methods may be called in any order and are
    safe to share across threads.
    """

    # ------------------------------------------------------------------
    # rename_symbol
    # ------------------------------------------------------------------
    def rename_symbol(
        self,
        code: str,
        old: str,
        new: str,
        scope: str = "module",
    ) -> RefactorResult:
        """Rename every occurrence of ``old`` to ``new``.

        Parameters
        ----------
        code:
            Source text to refactor.
        old, new:
            Identifiers; ``new`` must be a valid Python identifier.
        scope:
            ``"module"`` (default) renames across the whole file.
            ``"function"`` restricts the rewrite to the first function
            whose body references ``old``.

        Notes
        -----
        String literals and comments are never touched because the AST
        does not expose their contents as identifiers. If nothing
        matches, the input is returned unchanged with ``changes == 0``.
        """

        warnings: list[str] = []
        if not new.isidentifier():
            raise ValueError(f"{new!r} is not a valid Python identifier")

        tree = ast.parse(code)

        if scope == "module":
            tx = _RenameTransformer(old, new)
            tree = tx.visit(tree)
            ast.fix_missing_locations(tree)
            changes = tx.changes
        elif scope == "function":
            # Walk top-down and rename inside the first function that
            # mentions ``old``; leaves other functions untouched.
            target: ast.AST | None = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    probe = _RenameTransformer(old, new)
                    # Probe without mutating: count references by walking.
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Name) and sub.id == old:
                            target = node
                            break
                        if isinstance(sub, ast.arg) and sub.arg == old:
                            target = node
                            break
                    if target is not None:
                        break
                    del probe
            if target is None:
                changes = 0
            else:
                tx = _RenameTransformer(old, new)
                tx.visit(target)
                ast.fix_missing_locations(target)
                changes = tx.changes
        else:
            raise ValueError(f"unknown scope {scope!r}")

        if changes == 0:
            warnings.append(f"no occurrences of {old!r} found")
            return RefactorResult(code, "rename_symbol", 0, warnings)

        return RefactorResult(_unparse(tree), "rename_symbol", changes, warnings)

    # ------------------------------------------------------------------
    # inline_variable
    # ------------------------------------------------------------------
    def inline_variable(self, code: str, var_name: str) -> RefactorResult:
        """Replace references to ``var_name`` with its assigned value.

        Only fires when the variable is assigned exactly once at module
        scope with a simple ``x = expr`` form. Augmented and annotated
        assignments disqualify the variable (and produce a warning).
        """

        warnings: list[str] = []
        tree = ast.parse(code)
        assigns = _collect_assignments(tree, var_name)

        if len(assigns) == 0:
            warnings.append(f"{var_name!r} is not assigned at module scope")
            return RefactorResult(code, "inline_variable", 0, warnings)
        if len(assigns) > 1:
            warnings.append(f"{var_name!r} has {len(assigns)} assignments; refusing to inline")
            return RefactorResult(code, "inline_variable", 0, warnings)

        assign = assigns[0]
        if not isinstance(assign, ast.Assign) or len(assign.targets) != 1:
            warnings.append(f"{var_name!r} is not a simple single-target assignment")
            return RefactorResult(code, "inline_variable", 0, warnings)

        value_expr = assign.value

        # Substitute and drop the defining assignment in one pass.
        changes = 0

        class _Inliner(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                nonlocal changes
                if isinstance(node.ctx, ast.Load) and node.id == var_name:
                    changes += 1
                    # Copy to preserve original value_expr identity.
                    return ast.copy_location(
                        ast.parse(_unparse(value_expr), mode="eval").body, node
                    )
                return node

        new_body: list[ast.stmt] = []
        for stmt in tree.body:
            if stmt is assign:
                continue  # drop the original binding
            new_body.append(_Inliner().visit(stmt))
        tree.body = new_body
        ast.fix_missing_locations(tree)

        return RefactorResult(_unparse(tree), "inline_variable", changes + 1, warnings)

    # ------------------------------------------------------------------
    # remove_unused_imports
    # ------------------------------------------------------------------
    def remove_unused_imports(self, code: str) -> RefactorResult:
        """Drop imports whose bound names are never referenced.

        Uses :mod:`tokenize` only to verify input is lexable (so we can
        attach a precise warning for non-ASCII oddities); the actual
        rewrite is AST-level. Imports inside functions / classes are
        considered in their enclosing scope.
        """

        warnings: list[str] = []
        tree = ast.parse(code)

        # Collect every referenced Name.id anywhere in the tree.
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                # The root of a dotted access is a Name; capture it too.
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    used.add(root.id)

        changes = 0

        def _filter_aliases(aliases: list[ast.alias]) -> list[ast.alias]:
            nonlocal changes
            kept: list[ast.alias] = []
            for alias in aliases:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in used:
                    kept.append(alias)
                else:
                    changes += 1
            return kept

        class _Pruner(ast.NodeTransformer):
            def visit_Import(self, node: ast.Import) -> ast.AST | None:  # noqa: N802
                node.names = _filter_aliases(node.names)
                return node if node.names else None

            def visit_ImportFrom(self, node: ast.ImportFrom):  # noqa: N802
                # Never drop star imports: we cannot reason about them.
                if any(a.name == "*" for a in node.names):
                    return node
                node.names = _filter_aliases(node.names)
                return node if node.names else None

        tree = _Pruner().visit(tree)
        ast.fix_missing_locations(tree)

        if changes == 0:
            return RefactorResult(code, "remove_unused_imports", 0, warnings)

        # Lex the original (not the rewrite) just to keep the tokenize
        # import load-bearing and to surface encoding-level warnings.
        try:
            list(tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline))
        except tokenize.TokenizeError as exc:  # pragma: no cover - sanity
            warnings.append(f"tokenize warning: {exc}")

        return RefactorResult(_unparse(tree), "remove_unused_imports", changes, warnings)

    # ------------------------------------------------------------------
    # extract_function
    # ------------------------------------------------------------------
    def extract_function(
        self,
        code: str,
        start_line: int,
        end_line: int,
        new_fn_name: str,
    ) -> RefactorResult:
        """Lift statements in ``[start_line, end_line]`` into a new function.

        Free variables referenced inside the block that are defined
        before it become parameters of ``new_fn_name``; names assigned
        inside the block are returned as a tuple (single assignment:
        returned bare). The original block is replaced by a call to the
        new function. ``start_line``/``end_line`` are 1-indexed and
        inclusive, matching :mod:`ast` line numbers.
        """

        warnings: list[str] = []
        if not new_fn_name.isidentifier():
            raise ValueError(f"{new_fn_name!r} is not a valid Python identifier")
        tree = ast.parse(code)

        # Locate the owning statement list (module-level only).
        body = tree.body
        block_idxs = [
            i
            for i, stmt in enumerate(body)
            if getattr(stmt, "lineno", -1) >= start_line
            and getattr(stmt, "end_lineno", -1) <= end_line
        ]
        if not block_idxs:
            warnings.append(f"no module-level statements in lines {start_line}-{end_line}")
            return RefactorResult(code, "extract_function", 0, warnings)

        block = [body[i] for i in block_idxs]

        # Names defined *before* the block (candidate parameters).
        defined_before: set[str] = set()
        for stmt in body[: block_idxs[0]]:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    defined_before.add(sub.id)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_before.add(sub.name)

        # Names loaded vs stored inside the block.
        loaded_in_block: list[str] = []
        stored_in_block: list[str] = []
        seen_load: set[str] = set()
        seen_store: set[str] = set()
        for stmt in block:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name):
                    if isinstance(sub.ctx, ast.Load) and sub.id not in seen_load:
                        loaded_in_block.append(sub.id)
                        seen_load.add(sub.id)
                    elif isinstance(sub.ctx, ast.Store) and sub.id not in seen_store:
                        stored_in_block.append(sub.id)
                        seen_store.add(sub.id)

        params = [n for n in loaded_in_block if n in defined_before]
        returns = [n for n in stored_in_block]

        # Build the new function.
        new_body: list[ast.stmt] = list(block)
        if returns:
            if len(returns) == 1:
                ret = ast.Return(value=ast.Name(id=returns[0], ctx=ast.Load()))
            else:
                ret = ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(id=n, ctx=ast.Load()) for n in returns],
                        ctx=ast.Load(),
                    )
                )
            new_body.append(ret)

        fn_def = ast.FunctionDef(
            name=new_fn_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=p, annotation=None) for p in params],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=new_body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_params=[],
        )

        # Replace the original block with a call (optionally unpacked).
        call = ast.Call(
            func=ast.Name(id=new_fn_name, ctx=ast.Load()),
            args=[ast.Name(id=p, ctx=ast.Load()) for p in params],
            keywords=[],
        )
        if returns:
            if len(returns) == 1:
                replacement: ast.stmt = ast.Assign(
                    targets=[ast.Name(id=returns[0], ctx=ast.Store())],
                    value=call,
                )
            else:
                replacement = ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=[ast.Name(id=n, ctx=ast.Store()) for n in returns],
                            ctx=ast.Store(),
                        )
                    ],
                    value=call,
                )
        else:
            replacement = ast.Expr(value=call)

        new_module_body: list[ast.stmt] = []
        inserted = False
        for i, stmt in enumerate(body):
            if i in block_idxs:
                if not inserted:
                    new_module_body.append(replacement)
                    inserted = True
                continue
            new_module_body.append(stmt)
        # Place the new function at the top of the module.
        new_module_body.insert(0, fn_def)
        tree.body = new_module_body
        ast.fix_missing_locations(tree)

        return RefactorResult(
            _unparse(tree),
            "extract_function",
            len(block),
            warnings,
        )

    # ------------------------------------------------------------------
    # add_type_hint
    # ------------------------------------------------------------------
    def add_type_hint(
        self,
        code: str,
        function_name: str,
        param_name: str,
        type_annotation: str,
    ) -> RefactorResult:
        """Attach ``type_annotation`` to ``param_name`` of ``function_name``.

        The annotation is parsed as an expression so invalid syntax
        surfaces immediately. Existing annotations are overwritten (a
        warning is attached).
        """

        warnings: list[str] = []
        annotation_node = ast.parse(type_annotation, mode="eval").body

        tree = ast.parse(code)
        changes = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ):
                all_args = (
                    list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
                )
                for a in all_args:
                    if a.arg == param_name:
                        if a.annotation is not None:
                            warnings.append(f"overwriting existing annotation on {param_name!r}")
                        a.annotation = annotation_node
                        changes += 1
                        break

        if changes == 0:
            warnings.append(f"no parameter {param_name!r} on function {function_name!r}")
            return RefactorResult(code, "add_type_hint", 0, warnings)

        ast.fix_missing_locations(tree)
        return RefactorResult(_unparse(tree), "add_type_hint", changes, warnings)
