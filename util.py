import types
from abc import abstractmethod
from inspect import isclass
from types import GenericAlias
from typing import Union, get_args, get_origin, override
from weakref import WeakKeyDictionary


def next_unique_name(prefix: str, used_names: set[str]) -> str:
    i = 0
    while True:
        name = prefix + str(i)
        if name not in used_names:
            used_names.add(name)
            return name
        i += 1

class ObjectCounter[T]:
    _id: WeakKeyDictionary[T, int]
    _next_id: int

    def __init__(self, next_id: int = 0) -> None:
        self._id = WeakKeyDictionary()
        self._next_id = next_id

    def get_id(self, obj: T) -> int:
        if obj in self._id:
            return self._id[obj]
        ret = self._next_id
        self._next_id += 1
        self._id[obj] = ret
        return ret

class StrBiMap[V]:
    _k2v: dict[str, V]
    _v2k: dict[V, str]

    def __init__(self) -> None:
        self._k2v = {}
        self._v2k = {}

    def get_key(self, value: V) -> str:
        return self._v2k[value]

    def get_value(self, key: str) -> V:
        return self._k2v[key]

    def has_key(self, key: str):
        return key in self._k2v

    def has_value(self, value: V):
        return value in self._v2k

    def next_unique_name(self, prefix: str='') -> str:
        if prefix not in self._k2v:
            return prefix
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self._k2v:
                return name
            i += 1

    def add(self, key: str, value: V):
        assert key not in self._k2v
        assert value not in self._v2k
        self._k2v[key] = value
        self._v2k[value] = key

    def items(self):
        return self._k2v.items()

    def keys(self):
        return self._k2v.keys()

    def values(self):
        return self._k2v.values()

class PtrObject:
    @override
    def __eq__(self, value: object, /) -> bool:
        return self is value

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)

class SourceBuilder:
    lines: list[str]
    indents: int

    def __init__(self, indents: int = 0) -> None:
        self.lines = []
        self.indents = indents

    def emit(self, line: str):
        self.lines.append(('    ' * self.indents) + line)

class ClassVisitorCodeGen:
    @abstractmethod
    def gen_leaf(self, type: type) -> str:
        raise NotImplementedError

class SubExprFnBuilder:
    _base_cls: type
    _var_counter: int

    def __init__(self, base_cls: type) -> None:
        self._base_cls = base_cls
        self._var_counter = 0

    def _var(self) -> str:
        ret = f"var{self._var_counter}"
        self._var_counter += 1
        return ret

    def _check_type(self, type: type):
        return isclass(type) and issubclass(type, self._base_cls)

    def _contains_expr(self, type) -> bool:
        """Whether the type (or a nested collection of it) mentions an ``Expr``
        subclass, i.e. whether ``map`` must descend into it."""
        if self._check_type(type):
            return True
        if get_origin(type) is None:
            return False
        return any(
            (isclass(a) or isinstance(a, GenericAlias)) and self._contains_expr(a)
            for a in get_args(type)
        )

    def _process_get_child(self, var_name: str, type: type) -> list[str]:
        if self._check_type(type):
            return [f'ret.append({var_name})']
        head = get_origin(type)
        if head is list:
            args = get_args(type)
            v = self._var()
            body = self._process_get_child(v, args[0])
            if len(body) == 0:
                return []
            return [f"for {v} in {var_name}:", *('    ' + line for line in body)]
        if head is tuple:
            args = get_args(type)
            if len(args) == 2 and args[1] == Ellipsis and self._check_type(args[0]):
                v = self._var()
                body = self._process_get_child(v, args[0])
                if len(body) == 0:
                    return []
                return [f"for {v} in {var_name}:", *('    ' + line for line in body)]
            else:
                lines: list[str] = []
                for i, arg in enumerate(args):
                    lines.extend(self._process_get_child(f"{var_name}[{i}]", arg))
                return lines
        return []

    def _process_map(self, op: str, var_name: str, type: type) -> str:
        if self._check_type(type):
            # recurse into the child's own map so that ``op`` reaches every node
            # of the tree (deep substitution)
            return f"{var_name}.map({op})"
        if not self._contains_expr(type):
            # collections of plain values (e.g. ``tuple[tuple[int, int], ...]``)
            # are not expressions and stay untouched
            return var_name
        head = get_origin(type)
        args = get_args(type)
        if head is list or head is tuple and args[1] is Ellipsis:
            v = self._var()
            head_fn = 'tuple' if head is tuple else 'list'
            return f"{head_fn}({self._process_map(op, v, args[0])} for {v} in {var_name})"
        if head is tuple:
            return f"tuple({', '.join(self._process_map(op, f"{var_name}[{i}]", arg) for i, arg in enumerate(args))})"
        return var_name

    def _process_compare(self, self_arg: str, other_arg: str, type: type, compare_fn: str) -> list[str]:
        if self._check_type(type):
            v = self._var()
            return [f"{v} = {self_arg}.{compare_fn}({other_arg})", f"if {v} != 0:", f"    return {v}"]
        head = get_origin(type)
        args = get_args(type)
        if head is list or head is tuple and args[1] is Ellipsis:
            v1 = self._var()
            v2 = self._var()
            body = self._process_compare(v1, v2, args[0], compare_fn)
            if len(body) == 0:
                return []
            return [f"for {v1}, {v2} in zip({self_arg}, {other_arg}):", *('    ' + a for a in body)]
        if head is tuple:
            ret: list[str] = []
            for i, arg in enumerate(args):
                ret.extend(self._process_compare(f"{self_arg}[{i}]", f"{other_arg}[{i}]", arg, compare_fn))
            return ret
        if head is types.UnionType or head is Union:
            # a union with a single non-``None`` member, e.g. ``tuple | None``:
            # ``None`` compares as the smallest value and the other member is
            # compared by its own rules
            non_none = [a for a in args if a is not types.NoneType]
            if len(non_none) == 1:
                body = [
                    f'if {self_arg} is None and {other_arg} is not None:',
                    '    return -1',
                    f'if {self_arg} is not None and {other_arg} is None:',
                    '    return 1',
                ]
                body.extend(self._process_compare(self_arg, other_arg, non_none[0], compare_fn))
                return body

        return [
            f'if {self_arg} > {other_arg}:',
            '    return 1',
            f'if {self_arg} < {other_arg}:',
            '    return -1',
        ]

    def generate_get_children(self, cls: type, fn_name: str, excludes: set[str] | None = None):
        body: list[str] = [f'def {fn_name}(self):', "    ret = []"]
        for b in cls.mro()[-1::-1]:
            if b is object or b is self._base_cls:
                continue
            for name, type in b.__annotations__.items():
                if excludes is not None and name in excludes:
                    continue
                body.extend(('    ' + i) for i in self._process_get_child(f"self.{name}", type))
        body.append('    return ret')
        return body

    def generate_map(self, cls: type, fn_name: str, exclude: set[str] | None = None) -> list[str]:
        op = 'op'
        body: list[str] = [f'def {fn_name}(self, {op}):', "    ret = {}"]
        for b in cls.mro()[-1::-1]:
            if b is object or b is self._base_cls:
                continue
            for name, type in b.__annotations__.items():
                if exclude is not None and name in exclude:
                    continue
                body.append('    ' + f'ret["{name}"] = {self._process_map(op, f'self.{name}', type)}')
        # rebuild with the runtime type: subclasses of a decorated class (e.g.
        # ``Sin`` of ``UnaryNumericFunction``) are mapped back to themselves
        body.append('    return op(type(self)(**ret))')
        return body

    def generate_compare_body(self, cls: type, self_arg: str, other_arg: str, compare_fn: str, exclude: set[str] | None = None):
        body: list[str] = []
        for b in cls.mro()[-1::-1]:
            if b is object or b is self._base_cls:
                continue
            for name, type in b.__annotations__.items():
                if exclude is not None and name in exclude:
                    continue
                body.extend(self._process_compare(f"{self_arg}.{name}", f"{other_arg}.{name}", type, compare_fn))
        body.append('return 0')
        return body

def resize[T](arr: list[T], value: T, length: int):
    while len(arr) < length:
        arr.append(value)

def add_line_numbers(lines: list[str]):
    max_num_len = len(str(len(lines) + 1))
    ret: list[str] = []
    for i, line in enumerate(lines):
        n = str(i + 1)
        ret.append(n + (' ' * (max_num_len - len(n))) + '|' + line)
    return ret

def gen_get_children(cls: type | None = None, base: type | None = None, excludes: set[str] | None = None, method_name: str = 'get_children'):
    def wrapper[T](cls: type[T]) -> type[T]:
        base0 = base
        if base0 is None:
            base0 = cls.mro()[-2]

        children_builder = SubExprFnBuilder(base0)
        get_children = 'get_children'
        globals = {}
        source = '\n'.join(children_builder.generate_get_children(cls, get_children, excludes))
        exec(source, globals=globals)  # noqa: S102
        setattr(cls, method_name, globals[get_children])
        return cls

    if cls is not None:
        return wrapper(cls)

    return wrapper

class IdentityObj:
    def __hash__(self) -> int:
        return object.__hash__(self)

    def __eq__(self, value: object, /) -> bool:
        return self is value

class IndexedMap[K, V]:
    def __init__(self) -> None:
        self._by_id: list[V] = []
        self._by_key: dict[K, int] = {}

    def add(self, key: K, value: V) -> int:
        assert key not in self._by_key
        id = len(self._by_id)
        self._by_key[key] = id
        self._by_id.append(value)
        return id

    def get_by_id(self, id: int) -> V:
        return self._by_id[id]

    def get_by_key(self, key: K) -> V:
        return self._by_id[self._by_key[key]]
