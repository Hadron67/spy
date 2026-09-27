"""The untyped HIR.

Like ``llvm``, the HIR is a *linear* stream of instructions:
every instruction object is also its own result register (instructions
have identity; operands of later instructions reference earlier
operand objects).  ``astgen`` flattens expressions into temporary
instructions, so no instruction is ever nested inside another one.
There are no types anywhere in the HIR: typing happens only when the
interpreter *runs* the instructions with the concrete argument types.

Calls follow *result location semantics* (RLS): a call writes its
result into the slot of its ``ret`` operand (:class:`CallInplace`) and
produces no register of its own.  A caller that needs the value
allocates a slot and loads it back.  The slot only becomes real memory
when it is committed (``CommitSlot``): a slot all of whose stores are
compile-time - an inline temporary, an ``Alloca`` whose
:class:`InlineMode` allows it - stays a compile-time value, a
zero-sized slot only records its unit value, and
anything else is materialized as memory.  The store/load round trip a
runtime call result leaves behind is folded back into registers
afterwards by ``opt`` (see ``interp``).  The return of a function is
governed by the same semantics: each ``return`` writes into the
function's result location (:class:`ResultLoc`) and is followed by a
value-less :class:`Ret` terminator, and ``astgen`` appends a trailing
:class:`StoreVoidRetloc` to every body - the store of the void unit
value for a path that falls off the end; the interpreter turns the
write into the return value of a direct-return function, or into a
store through the result pointer of a result-pointer function.

``astgen`` performs all name resolution: a read of a variable - a
parameter or a block-local declaration - becomes a :class:`Load` of
the variable's storage.  A global is an *immutable value*: in a
value context the name becomes a :class:`Const` holding the resolved
Python object (spy types, the ``spy`` module functions, functions to
call/inline, ...), in a reference context it becomes a
:class:`ConstRef` - a reference to that value (how callable callees
are passed to :class:`CallInplace`).  Attribute access on such
compile-time objects is evaluated there as well.  The HIR never
carries a variable *name*.

At HIR level a parameter is passed by reference (its address): the
translated body binds the name of the i-th parameter directly to its
:class:`Arg` leaf, and the interpreter materializes the parameter's
storage (a MIR alloca holding ``mir.Param``) when the first store runs.
A local variable is declared with a fresh :class:`Alloca` at its first
assignment, and a later assignment to it is a plain ``Store`` of its
slot.

(:class:`Arg` is the *address* of the i-th argument of the function
being executed.)  The interpreter *types* an ``Alloca`` when its first
store executes, so the untyped HIR needs no type information.

Operands of instructions are therefore either

* :class:`Const` leaves - Python literals and the values of immutable
  globals,
* :class:`ConstRef` leaves - references (const pointers) to immutable
  globals,
* :class:`Arg` leaves - the addresses of the arguments of the
  function (parameters are passed by reference),
* instruction objects produced by earlier instructions.

Statements: every function body is one *flat* list of instructions
containing all of its control flow.  A conditional is an :class:`If`
instruction followed - in the same list - by the instructions of its
then branch, the :class:`Else` marker and the else branch (when one
exists), and the :class:`End` marker that closes the block, like WASM's
``if ... else ... end``.  The interpreter walks the flat list: a
compile-time ``if`` skips the branch it does not choose (its
instructions are never run, so the branch is dead), a runtime ``if``
types both branches (both survive at runtime).  A loop is a
:class:`Loop` instruction followed by its body and the ``End`` that
closes it, like WASM's ``loop ... end``: the interpreter types the body
once and jumping back to it is the next iteration; a :class:`Break`
leaves the loop (jumping to the code after its ``End``) and a
:class:`Continue` starts the next iteration (jumping back to the
``Loop`` itself).
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any

from .binop import BinaryOp, CompareOp, UnaryOp
from .binop import BoolOp as BoolOpType
from .fn import ArgEntry, RawArgList, frozendict


class Value:
    pass


class InlineMode(IntEnum):
    """How much of a slot's value may be kept inline (see :class:`Alloca`).

    ``NONE`` is a plain runtime location: the slot holds a value with a
    runtime representation.  ``NON_AGGREGATE`` may keep any value inline
    except an *aggregate* (a struct or an array), which has no inline storage
    of its own here (see ``ComptimeAggregatePtr``).  ``FULL`` may keep
    anything inline - it is what a ``Comptime`` variable declares.  A
    zero-sized value has no runtime representation at all, so its slot only
    records its unit value whatever the mode is."""

    NONE = auto()
    NON_AGGREGATE = auto()
    FULL = auto()


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
    ``ConstRef(expr)`` as ``sval.PointerType(sval.type_of(expr), True)``.  A
    function value - whose type is a runtime DST that cannot be used by
    value - is only ever referenced through such a reference (a
    function pointer)."""

    value: Any


@dataclass(frozen=True)
class Arg(Value):
    """The address of the index-th argument of the function being
    executed (parameters are passed by reference at HIR level)."""

    index: int


@dataclass(frozen=True)
class ResultLoc(Value):
    """The result location of the function whose body is being executed"""


@dataclass(frozen=True)
class ErrorLoc(Value):
    """The error location of the function whose body is being executed: the
    place a ``raise`` delivers its exception into (the error code and the
    payload).  Like :class:`ResultLoc` it is a leaf the interpreter resolves
    to the state of the function proper, not a register of its own."""


class Inst(Value):
    """An instruction; the object itself acts as its result register."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(eq=False)
class Alloca(Inst):
    """Reserve an addressable slot for one value.  The slot is untyped
    until it is used: the stores that target it (a plain ``Store``, or a
    ``CallInplace`` result under RLS) type it, and the ``CommitSlot``
    that follows then materializes it - a slot whose stores may all be
    kept inline stays a compile-time value instead of memory (see
    :class:`InlineMode` and ``interp``).

    An annotated local variable (``x: T``/``x: Comptime[T]``) declares its
    type here instead: ``type`` is the compile-time type value of the
    annotation (``None`` for the bare ``Comptime``, whose type is left to
    the stores), and the interpreter materializes the slot right away -
    memory for a runtime type, a compile-time value for a ``Comptime`` or a
    zero-sized one.  ``inline`` is how much of the value may be kept inline:
    the default is a plain slot, ``NON_AGGREGATE`` an expression temporary
    and ``FULL`` a ``Comptime`` variable."""
    inline: InlineMode = InlineMode.NONE
    type: Value | None = None

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
class StoreVoidRetloc(Inst):
    """Equivalent to ``Store(ResultLoc(), Const(sval.Void()))``."""


@dataclass(eq=False)
class FieldAddr(Inst):
    """The address of the field ``name`` of the struct ``base`` points
    at.  ``base`` denotes the *storage* of a struct value: the slot of a
    variable (an ``Alloca``, or the ``Arg`` of a parameter), or the
    address of a nested field (another ``FieldAddr``); the interpreter
    resolves it - and the field's type -
    from the static type it has typed ``base`` with (see ``interp``),
    auto-dereferencing a base that points at a pointer (a ``self``
    passed by pointer, a pointer-valued field, ...) first.

    ``is_aggregate_init`` marks the field addresses a construction
    writes through: the base is then the storage the aggregate is built
    in, which may still be an uncommitted slot - the field then gets a
    pending place of its own (see ``FieldIndexAddr``)."""

    base: Value
    name: str
    is_aggregate_init: bool = False

@dataclass(eq=False)
class FieldIndexAddr(Inst):
    """The address of the ``index``-th field of the aggregate ``base``
    points at - the address a positional argument of a struct
    construction, or a positional element of an array construction, is
    written into (the ``index`` names a declaration-ordered field of a
    struct, an element of an array).  Unlike :class:`FieldAddr` the base
    is never auto-dereferenced (the base of a construction is the
    storage it is built in, already a pointer to the aggregate).

    ``is_aggregate_init`` marks the addresses a construction writes
    through: when the base is a slot that is not committed yet - the
    type of an aggregate being built is not known until it is closed -
    the field/element gets a pending place of its own, which the
    ``FinishStruct``/``FinishArray`` closing the construction turns into
    the address of its field/element (see ``interp``)."""

    base: Value
    index: int
    is_aggregate_init: bool = False

@dataclass(eq=False)
class FinishStruct(Inst):
    """Close the construction of the struct ``struct`` names in the
    storage ``dest``.  ``struct`` is a reference to the struct built - a
    ``ConstRef`` of a ``@struct()`` class, or the ``Subscript`` that
    specializes a struct template - and ``dest`` is the storage the
    fields were generated into.  ``indices`` are the addresses the
    positional arguments were written into, in the order they were given,
    and ``names`` the ones the keyword arguments were written into, by
    field name; every field has to be written - a field may only be left
    out when it has a default, which is not implemented yet - so a missing
    field is an error, a zero-sized one included.  The parser only has to
    know the syntax, not the field layout: the interpreter resolves the
    struct type (inferring the generic arguments a template was not given)
    from ``struct`` and the field addresses."""

    struct: Value
    dest: Value
    indices: tuple[Value, ...]
    names: frozendict[str, Value]

@dataclass(eq=False)
class FinishArray(Inst):
    """Close the construction of an array in the storage ``array`` points
    at.  The ``elements`` are the addresses the elements were generated into
    (``FieldIndexAddr`` instructions), in the order they were given: their
    count is the length of the array, and they resolve its element type.  An
    array has no element omitted (there is no default for one), so an element
    that is left out is an error."""

    array: Value
    elements: tuple[Value, ...]

@dataclass(eq=False)
class CallMethodInplace(Inst):
    """A call of the method ``name`` of the struct ``base`` points at
    (result-location semantics like :class:`CallInplace`).  A method is
    an ordinary function whose first parameter is the struct type of
    ``base``: the base's address is prepended as that first argument
    (passed by reference), unless the method declares ``self`` as a
    pointer type, in which case the base's address already is the value
    the parameter expects.  The interpreter resolves the method from the
    static type of the struct and runs the call like any other (see
    ``interp``)."""

    base: Value
    name: str
    args: RawArgList[ArgEntry[Value]]
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
    args: RawArgList[ArgEntry[Value]]
    ret: Value


@dataclass(eq=False)
class Subscript(Inst):
    base: Value
    index: ArgEntry[Value]

@dataclass(eq=False)
class InitTuple(Inst):
    tuple_ptr: Value
    length: int


@dataclass(eq=False)
class TuplePtrElement(Inst):
    tuple_ptr: Value
    index: int


@dataclass(eq=False)
class Tuple(Inst):
    values: tuple[ArgEntry[Value], ...]

@dataclass(eq=False)
class Dict(Inst):
    values: frozendict[str, ArgEntry[Value]]

@dataclass(eq=False)
class Binary(Inst):
    """Arithmetic: '+', '-', '*', '/', '//', '%', '**'."""

    op: BinaryOp
    lhs: ArgEntry[Value]
    rhs: ArgEntry[Value]
    ret: Value

@dataclass(eq=False, slots=True)
class BinaryAssign(Inst):
    op: BinaryOp
    lhs: Value
    rhs: ArgEntry[Value]


@dataclass(eq=False)
class Compare(Inst):
    """Comparison: '==', '!=', '<', '<=', '>', '>='."""

    op: CompareOp
    lhs: ArgEntry[Value]
    rhs: ArgEntry[Value]


@dataclass(eq=False)
class BoolOp(Inst):
    """Short-circuit 'and'/'or'.  Only compile-time operands are
    supported for now; the operands are evaluated eagerly when the HIR
    runs, so both sides of a compile-time ``and`` are always computed."""

    op: BoolOpType
    lhs: ArgEntry[Value]
    rhs: ArgEntry[Value]


@dataclass(eq=False)
class Unary(Inst):
    """Unary operator: '-', 'not'."""

    op: UnaryOp
    operand: ArgEntry[Value]
    ret: Value

@dataclass(eq=False)
class Ret(Inst):
    """End one path of the function; a path is terminated by a ``return``
    statement, whose expression was already evaluated into the function's
    result location (:class:`ResultLoc`).  The value itself is carried by
    the result location - the interpreter turns it into the function's
    return value (a direct-return function) or leaves it in the result
    pointer (a result-pointer function)."""


@dataclass(eq=False)
class Raise(Inst):
    """End one path of the function with an error; a path is terminated by a
    ``raise`` statement, whose exception was already delivered into the
    function's error location (:class:`ErrorLoc`) by the result-location
    evaluation that precedes it - like a ``Ret``, whose value the result
    location already holds.  The instruction therefore carries nothing."""


@dataclass(eq=False)
class AsBool(Inst):
    """Converts a value to a boolean."""
    value: ArgEntry[Value]

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
class Loop(Inst):
    """The start of a ``loop`` block (WASM-style, like :class:`If`): the
    instructions of the body follow it in the same list, closed by the
    matching :class:`End`.  The block is a *dead loop*: falling off its
    end jumps back to the ``Loop`` itself (the next iteration, whose head
    re-evaluates whatever the body computes), and it is left only by a
    :class:`Break` (or a ``return``/``raise``).  The ``Loop`` instruction
    carries nothing; it only delimits the flat instruction stream.

    ``is_inline`` marks a *compile-time* loop (a ``while`` whose condition
    the source wraps in ``syntax.inline_loop``): the interpreter does not
    emit a back edge but unrolls the body once per compile-time iteration,
    so a :class:`Break` leaves the whole unrolled sequence and a
    :class:`Continue` jumps to the next unrolled body (see ``interp``)."""

    is_inline: bool = False


@dataclass(eq=False)
class Break(Inst):
    """Leave the innermost open :class:`Loop` unconditionally: the path
    ends at the code after the loop's matching :class:`End`.  Like a
    ``ret`` the instruction carries nothing - the interpreter ends the
    current path with a jump to the loop's exit block (see ``interp``)."""


@dataclass(eq=False)
class Continue(Inst):
    """Start the next iteration of the innermost open :class:`Loop`
    unconditionally: the path ends back at the loop's head, so the rest
    of the body (and the ``else`` clause of the ``while``) is skipped and
    the loop's condition is evaluated again.  Like :class:`Break` the
    instruction carries nothing."""


@dataclass(eq=False)
class Try(Inst):
    """Open a ``try`` block: the instructions of the try body follow, then one
    :class:`Except` marker per ``except`` clause (each followed by its clause
    body), and the matching :class:`End`.  ``binds`` holds the ``as`` name's
    slot of every clause (None when the clause names none), created *before*
    the ``Try`` so that the clause body can read it - the interpreter fills it
    with the caught exception when the clause runs (see ``interp``)."""

    binds: tuple[Value | None, ...]


@dataclass(eq=False)
class Except(Inst):
    """The marker that starts one ``except`` clause of the innermost open
    :class:`Try` block: ``type`` names the exception struct the clause catches
    (None for a bare ``except:``), and ``index`` is the clause's position among
    the try's clauses (its handler and ``as`` slot are held by the enclosing
    block's ``TryExceptBlockData``)."""

    type: Value | None
    index: int


@dataclass(eq=False)
class End(Inst):
    """The marker that closes a block opened by an :class:`If`, a
    :class:`Loop` or a :class:`Try`: everything between the ``If`` (or its
    :class:`Else`) and this marker is one branch body, everything between
    the ``Loop`` and this marker is the loop body, and the code after
    this marker is the continuation of the enclosing block.  A marker
    produces no register; it only delimits the flat instruction stream."""

@dataclass(eq=False)
class CommitSlot(Inst):
    slot: Value


def scan_block(insts: tuple[Inst, ...], entry: int) -> tuple[int | None, int]:
    """The positions of the ``Else`` (or None when the block has no
    else branch) and ``End`` markers that close the block opened at
    ``entry`` (an ``hir.If`` or an ``hir.Loop``) of the executing frame's
    flat instruction list, found by a balanced scan forward from the
    entry (nested blocks close their own markers first).  A ``Loop``
    block has no ``Else`` marker of its own, so its ``Else`` is always
    None."""
    depth = 0
    p_else: int | None = None
    for i in range(entry + 1, len(insts)):
        inst = insts[i]
        if isinstance(inst, (If, Loop, Try)):
            depth += 1
        elif isinstance(inst, End):
            if depth == 0:
                return p_else, i
            depth -= 1
        elif isinstance(inst, Else) and depth == 0:
            p_else = i
    assert False, 'unclosed block in the HIR'


def scan_try(insts: tuple[Inst, ...], entry: int) -> tuple[list[int], int]:
    """The positions of the ``Except`` markers and of the closing ``End`` of the
    ``hir.Try`` block opened at ``entry``, of the executing frame's flat
    instruction list (found by a balanced scan forward, like
    :func:`scan_block`)."""
    depth = 0
    excepts: list[int] = []
    for i in range(entry + 1, len(insts)):
        inst = insts[i]
        if isinstance(inst, (If, Loop, Try)):
            depth += 1
        elif isinstance(inst, End):
            if depth == 0:
                return excepts, i
            depth -= 1
        elif isinstance(inst, Except) and depth == 0:
            excepts.append(i)
    assert False, 'unclosed block in the HIR'
