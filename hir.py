"""The untyped HIR.

Like ``symlat.jit.llvm``, the HIR is a *linear* stream of instructions:
every instruction object is also its own result register (instructions
have identity; operands of later instructions reference earlier
operand objects).  ``astgen`` flattens expressions into temporary
instructions, so no instruction is ever nested inside another one.
There are no types anywhere in the HIR: typing happens only when the
interpreter *runs* the instructions with the concrete argument types.

Calls follow *result location semantics* (RLS): a call writes its
result into the slot of its ``ret`` operand (:class:`CallInplace`) and
produces no register of its own.  A caller that needs the value
allocates a slot and loads it back; the interpreter keeps scalar and
compile-time results in the slot without giving it real memory.  The
return of a function is governed by the same semantics: its body ends
with a write into the function's result location (:class:`ResultLoc`)
followed by a value-less :class:`Ret` terminator; the interpreter turns
the write into the return value of a direct-return function, or into a
store through the result pointer of a result-pointer function.

``astgen`` performs all name resolution: a read of a variable - a
parameter or a block-local declaration - becomes a :class:`Load` of
the variable's :class:`Alloca`.  A global is an *immutable value*: in a
value context the name becomes a :class:`Const` holding the resolved
Python object (spy types, the ``spy`` module functions, functions to
call/inline, ...), in a reference context it becomes a
:class:`ConstRef` - a reference to that value (how callable callees
are passed to :class:`CallInplace`).  Attribute access on such
compile-time objects is evaluated there as well.  The HIR never
carries a variable *name*.

Because every parameter is addressable, the instruction list of a
function body begins with one ``Alloca``/``Store`` pair per parameter::

    %a = Alloca();  Store(%a, Arg(0))
    %b = Alloca();  Store(%b, Arg(1))

A local variable is declared the same way at its first assignment - the
``Alloca``/``Store`` pair sits at the declaration point instead of in
the prologue - and a later assignment to it is a plain ``Store`` of
its slot.

(:class:`Arg` refers to the i-th by-value argument of the function being
executed.)  The interpreter *types* an ``Alloca`` when its first store
executes, so the untyped HIR needs no type information.

Operands of instructions are therefore either

* :class:`Const` leaves - Python literals and the values of immutable
  globals,
* :class:`ConstRef` leaves - references (const pointers) to immutable
  globals,
* :class:`Arg` leaves - the by-value arguments of the function,
* instruction objects produced by earlier instructions.

Statements: every function body is one *flat* list of instructions
containing all of its control flow.  A conditional is an :class:`If`
instruction followed - in the same list - by the instructions of its
then branch, the :class:`Else` marker and the else branch (when one
exists), and the :class:`End` marker that closes the block, like WASM's
``if ... else ... end``.  The interpreter walks the flat list: a
compile-time ``if`` skips the branch it does not choose (its
instructions are never run, so the branch is dead), a runtime ``if``
types both branches (both survive at runtime).  Future block
instructions (loops, ...) will use the same marker representation.
"""

from dataclasses import dataclass
from typing import Any


class Value:
    pass


@dataclass(frozen=True)
class Const(Value):
    """A leaf holding a Python object: a literal, or the *value* of an
    immutable global (see :class:`ConstRef` for a reference to one)."""

    value: Any


@dataclass(frozen=True)
class ConstRef(Value):
    """A *reference* to an immutable global object: ``value`` is the
    resolved global (a function entry, a spy type, a captured constant,
    ...) as embedded by ``astgen`` in a reference context (``is_ref``).
    It denotes a const pointer to the global: the interpreter types a
    ``ConstRef(expr)`` as ``type.PointerType(typeof(expr), True)``.  A
    function value - whose type is a runtime DST that cannot be used by
    value - is only ever referenced through such a reference (a
    function pointer)."""

    value: Any


@dataclass(frozen=True)
class Arg(Value):
    """The index-th by-value argument of the function being executed."""

    index: int


@dataclass(eq=False)
class ResultLoc(Value):
    """The result location of the function whose body is being executed:
    a per-function leaf that only ever appears as the *target* of a
    return statement (see :class:`Ret`).  ``astgen`` evaluates the
    expression of a ``return`` into this location (result-location
    semantics, like the ``ret`` of a :class:`CallInplace`); the
    interpreter types it and decides from the function's return type how
    the value is delivered: written into the result pointer of a
    result-pointer function, or handed back as the return value of a
    direct-return function."""


class Inst(Value):
    """An instruction; the object itself acts as its result register."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(eq=False)
class Alloca(Inst):
    """Reserve an addressable slot for one value.  The slot is untyped
    until it is used: the first ``Store`` that targets it types and
    allocates it, while a ``CallInplace`` result (RLS) is only recorded
    in it - scalar and compile-time results are never given real memory.
    Function bodies start with one Alloca/Store pair per parameter."""


@dataclass(eq=False)
class Load(Inst):
    """Read the value of a slot (or pointer) into a register."""

    ptr: Value


@dataclass(eq=False)
class Store(Inst):
    """Write a value to a slot (or pointer)."""

    ptr: Value
    value: Value


@dataclass(eq=False)
class FieldAddr(Inst):
    """The address of the field ``name`` of the struct ``base`` points
    at.  ``base`` denotes the *storage* of a struct value: the slot of a
    variable (an ``Alloca``), or the address of a nested field (another
    ``FieldAddr``); the interpreter resolves it - and the field's type -
    from the static type it has typed ``base`` with (see ``interp``),
    auto-dereferencing a base that points at a pointer (a ``self``
    passed by pointer, a pointer-valued field, ...) first."""

    base: Value
    name: str


@dataclass(eq=False)
class CallMethodInplace(Inst):
    """A call of the method ``name`` of the struct ``base`` points at
    (result-location semantics like :class:`CallInplace`).  A method is
    an ordinary function whose first parameter is the struct type of
    ``base`` (by value) or a pointer to it (``ptr_self``): the
    interpreter resolves the method from the static type of the struct
    and runs the call like any other, with the base prepended as that
    first argument (see ``interp``)."""

    base: Value
    name: str
    args: tuple[Value, ...]
    ret: Value


@dataclass(eq=False)
class CallInplace(Inst):
    """A call whose result is written into a *result location* (RLS),
    like Zig: ``ret`` is the pointer the callee's result goes to and the
    instruction itself produces no register.  The ``callee`` is a
    reference to the function value (see ``astgen``'s ``is_ref``
    context), the ``args`` are by-value leaves/registers.  A consumer
    that needs the value loads it back from ``ret``."""

    callee: Value
    args: tuple[Value, ...]
    ret: Value


@dataclass(eq=False)
class Binary(Inst):
    """Arithmetic: '+', '-', '*', '/', '//', '%', '**'."""

    op: str
    lhs: Value
    rhs: Value


@dataclass(eq=False)
class Compare(Inst):
    """Comparison: '==', '!=', '<', '<=', '>', '>='."""

    op: str
    lhs: Value
    rhs: Value


@dataclass(eq=False)
class BoolOp(Inst):
    """Short-circuit 'and'/'or'.  Only compile-time operands are
    supported for now; the operands are evaluated eagerly when the HIR
    runs, so both sides of a compile-time ``and`` are always computed."""

    op: str
    lhs: Value
    rhs: Value


@dataclass(eq=False)
class Unary(Inst):
    """Unary operator: 'not', 'neg' (unary minus)."""

    op: str
    operand: Value


@dataclass(eq=False)
class Ret(Inst):
    """End one path of the function; a path is terminated by a ``return``
    statement, whose expression was already evaluated into the function's
    result location (:class:`ResultLoc`).  The value itself is carried by
    the result location - the interpreter turns it into the function's
    return value (a direct-return function) or leaves it in the result
    pointer (a result-pointer function)."""


@dataclass(eq=False)
class If(Inst):
    """Conditional statement (WASM-style): the instructions of the two
    branches follow this instruction in the same list, delimited by the
    matching :class:`Else` (when an else branch exists) and
    :class:`End` markers.  A compile-time condition is evaluated while
    the HIR runs and only the chosen branch survives; a runtime
    condition becomes a runtime branch in the MIR (both branch bodies
    are typed and compiled then)."""

    cond: Value


@dataclass(eq=False)
class Else(Inst):
    """The marker that starts the else branch of an :class:`If` block
    (absent when the ``if`` has no else branch): everything between the
    ``If`` and this marker is the then branch, everything between this
    marker and the matching :class:`End` is the else branch.  A marker
    produces no register; it only delimits the flat instruction
    stream."""


@dataclass(eq=False)
class End(Inst):
    """The marker that closes a block opened by an :class:`If` (or a
    future block instruction): everything between the ``If`` (or its
    :class:`Else`) and this marker is one branch body, and the code
    after this marker is the continuation of the enclosing block.  A
    marker produces no register; it only delimits the flat instruction
    stream."""
