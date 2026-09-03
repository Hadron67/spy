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
compile-time results in the slot without giving it real memory.

``astgen`` performs all name resolution: a variable (parameter) read
becomes a :class:`Load` of the variable's :class:`Alloca`, a global name
becomes a :class:`Const` holding the resolved Python object (spy types,
the ``spy`` module functions, functions to call/inline, ...), and
attribute access on such compile-time objects is evaluated there as
well.  The HIR never carries a variable *name*.

Because every parameter is addressable, the instruction list of a
function body begins with one ``Alloca``/``Store`` pair per parameter::

    %a = Alloca();  Store(%a, Arg(0))
    %b = Alloca();  Store(%b, Arg(1))

(:class:`Arg` refers to the i-th by-value argument of the function being
executed.)  The interpreter *types* an ``Alloca`` when its first store
executes, so the untyped HIR needs no type information.

Operands of instructions are therefore either

* :class:`Const` leaves - Python literals and resolved globals,
* :class:`Arg` leaves - the by-value arguments of the function,
* instruction objects produced by earlier instructions.

Statements and sub-lists: a function body is one list of instructions;
a compile-time ``if`` is an :class:`If` instruction carrying the two
branch bodies as plain instruction lists (the branches are not executed
together; runtime control flow with blocks and phis will build on the
same list representation later).
"""

from dataclasses import dataclass
from typing import Any


class Value:
    pass


@dataclass(frozen=True)
class Const(Value):
    """A leaf holding a Python object: a literal or a resolved global."""

    value: Any


@dataclass(frozen=True)
class Arg(Value):
    """The index-th by-value argument of the function being executed."""

    index: int


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
    value: Value


@dataclass(eq=False)
class If(Inst):
    """Conditional statement.  A compile-time condition is evaluated
    while the HIR runs and only the chosen branch survives; a runtime
    condition becomes a runtime branch in the MIR (both branch bodies
    are typed and compiled then)."""

    cond: Value
    then_body: tuple[Inst, ...]
    else_body: tuple[Inst, ...]
