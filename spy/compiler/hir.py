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
when it is committed (``CommitSlot``): a slot whose stores may all be kept
inline - an ``Alloca`` whose :class:`InlineMode` allows it and no store
that crossed a runtime block - stays a compile-time value, a
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
once and jumping back to it is the next iteration; a :class:`BreakLoop`
leaves the loop (jumping to the code after its ``End``) and a
:class:`Continue` starts the next iteration (jumping back to the
``Loop`` itself).
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag, auto
from typing import Any

from .binop import BinaryOp, CompareOp, UnaryOp
from .fn import ArgEntry, ClosureFunction, RawArgList, frozendict


class Value:
    pass


class InlineMode(IntEnum):
    """How much of a slot's value may be kept inline (see :class:`Alloca`).

    ``NONE`` is a plain runtime location: the slot holds a value with a
    runtime representation.  ``NON_AGGREGATE`` may keep any value inline
    except an *aggregate* (a struct or an array), which has no inline storage
    of its own here (see ``ComptimeAggregatePtr``).  ``FULL`` may keep
    anything inline - it is what a ``Comptime`` variable declares, and an
    aggregate in one is always held by its fields' places, whatever the
    values of those fields are.  A delivery that needs the slot's own single
    address (a result pointer a callee writes through), or a store that
    crossed a runtime block boundary since the slot was created, forces it
    into memory (see ``interp``).  A zero-sized value has no runtime
    representation at all, so its slot only records its unit value whatever
    the mode is."""

    NONE = auto()
    NON_AGGREGATE = auto()
    FULL = auto()


class DeferPath(IntFlag):
    """Which exits of the enclosing region trigger a ``with syntax.defer():``
    block (see :class:`Defer`), as the bitflags ``syntax.defer`` is written
    with: ``OK`` a normal exit (a ``return``/``break``/``continue``/falling off
    the region's end), ``RAISE`` a spy error leaving it and ``UNWIND`` a panic
    unwinding through it.  The values mirror the ``syntax.OK``/``syntax.RAISE``/
    ... constants, so an ``syntax.defer(flags)`` argument is these flags
    directly."""

    OK = 1
    RAISE = 2
    UNWIND = 4
    ERR = RAISE | UNWIND
    ALL = OK | ERR


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
class Closure(Value):
    """The address (the place) of the index-th variable a closure captured
    from the enclosing function (see :class:`MakeClosure` and ``interp``).
    Unlike :class:`Arg` it does not name a declared parameter: a closure's
    captures are carried by the frame's ``closure_values``, which the
    interpreter fills from the capture pointers a call passes (a compiled
    closure) or from the closure value itself (an inlined one)."""

    index: int


@dataclass(frozen=True)
class ResultLoc(Value):
    """The result location of the function whose body is being executed"""


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
    field name; a field that is left out takes its declared default (a
    zero-sized one included - it has no storage to write), so only a field
    that has no default is a missing field, which is an error.  The parser
    only has to know the syntax, not the field layout: the interpreter
    resolves the struct type (inferring the generic arguments a template
    was not given) from ``struct`` and the field addresses."""

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
class PointerType(Inst):
    """Build the spy pointer type an expression such as ``syntax.Ptr[T]`` or
    ``syntax.ConstMultiPtr[T]`` names: ``elem`` is the element type (a value
    operand) and ``is_const``/``is_multi`` say which of the four spellings it
    is.  The result is a compile-time type value (see ``interp``)."""

    elem: Value
    is_const: bool
    is_multi: bool

@dataclass(eq=False)
class ArrayType(Inst):
    """Build the spy array type ``syntax.Array[T, N]``: ``length`` values of
    the element type ``elem`` (both value operands; the length is a *value*,
    which may be a type parameter the call solves).  A ``length`` of ``None``
    builds an *unsized* array (``syntax.Array[T, None]``), a dynamically-sized
    type (see ``sval.ArrayType``)."""

    elem: Value
    length: Value | None

@dataclass(eq=False)
class OptionType(Inst):
    """Build the spy option type ``syntax.Option[T]``: the child type ``child``
    (a value operand)."""

    child: Value

@dataclass(eq=False)
class PtrCast(Inst):
    """``syntax.ptr_cast(ptr, T)``: reinterpret the pointer ``value`` as the
    pointer type the value operand ``type`` names (see ``interp``)."""

    value: Value
    type: Value

@dataclass(eq=False)
class AsFuncPtr(Inst):
    """``syntax.as_func_ptr(T, f)``: the runtime pointer to the spy function
    ``f`` of the function type ``T`` (both value operands).  The result is a
    ``ConstPtr[T]`` value (see ``interp``)."""

    type: Value
    obj: Value


@dataclass(eq=False)
class TypeOfBegin(Inst):
    """``syntax.typeof(expr)``: open a *type probe*.  The instructions that
    evaluate ``expr`` follow this marker in the same list, closed by the matching
    :class:`TypeOfEnd`.  The probe emits no MIR: the interpreter builds the
    probe's instructions into a detached block that no path reaches, so they
    only *type* ``expr`` - compiling whatever functions it names - and are then
    discarded.  A transfer out of the probe (a ``return``/``raise``/
    ``break``/``continue``, or a call whose error is not caught inside) is
    rejected; the probe only ever types (see ``interp``)."""


@dataclass(eq=False)
class TypeOfEnd(Inst):
    """Close the type probe opened by :class:`TypeOfBegin`: ``value`` is the
    probe expression's place (a reference when ``is_ref``), and this
    instruction's register is its spy type, held as a compile-time value.  The
    probe's instructions are never lowered, so the type is the only thing it
    produces (see ``interp``)."""

    value: Value
    is_ref: bool


@dataclass(eq=False)
class MakeClosure(Inst):
    """Create a closure value: the nested ``def``/``lambda`` ``fn`` bound to
    the captured variables ``captures`` (the enclosing function's slots, or an
    enclosing closure's :class:`Closure` - every capture is a place).  The
    parameter annotations/default values/return annotation are value operands
    evaluated here (in the enclosing frame); the interpreter assembles the
    closure's concrete signature from them (see ``interp``).  ``fn`` is a
    :class:`~spy.compiler.fn.ClosureFunction`, the parsed body of the nested
    function - its declared parameters stay separate from the captures, so
    ``*args``/``**kwargs`` can be added later without disturbing them."""

    fn: ClosureFunction
    annotations: tuple[Value | None, ...]
    defaults: tuple[Value | None, ...]
    ret_annotation: Value | None
    captures: tuple[Value, ...]

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
    """A tuple *value*: the operands (a value or a reference to one) the shape
    of a ``tuple`` expression is made of (see ``interp.ComptimeTuple``)."""

    values: tuple[ArgEntry[Value], ...]


@dataclass(eq=False)
class TuplePtr(Inst):
    """A tuple *place*: one address per element, the destructuring target
    ``a, b = ...`` is built as (a nested target nests; see
    ``interp.ComptimeTuplePtr`` and ``astgen._gen_target_tuple``)."""

    values: tuple[Value, ...]

@dataclass(eq=False)
class Dict(Inst):
    values: frozendict[str, ArgEntry[Value]]


@dataclass(eq=False)
class Binary(Inst):
    """Binary operator: arithmetic ('+', '-', '*', '/', '//', '%', '**'),
    bitwise/shift ('|', '&', '^', '<<', '>>') and '|' as a tagged-union type
    value.  The interpreter decides which of the two meanings '|' has (see
    ``interp``), because the same syntax spells both."""

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
class Not(Inst):
    """Boolean negation, value -> value: the negation of the boolean
    ``value`` (``astgen`` feeds it the :class:`AsBool` of the source
    operand, so it already holds a ``bool``).  Unlike :class:`Unary` it has
    no result location; the instruction itself is the register holding the
    ``bool`` result."""

    value: Value


@dataclass(eq=False)
class Unary(Inst):
    """Unary operator: '-' (negation), '~' (bitwise not)."""

    op: UnaryOp
    operand: ArgEntry[Value]
    ret: Value

@dataclass(eq=False)
class Ord(Inst):
    """``ord(x)``: the encoding of the byte the compile-time byte string
    ``operand`` holds, written into the result location ``ret``.  The operand
    has to be a ``bytes`` value of exactly one byte (see ``interp``); the
    result is an *untyped* integer, so the destination gives it its type."""

    operand: Value
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
    """End one path with an error: the exception value has already been
    written into the slot ``value`` denotes (the result-location evaluation
    that precedes this instruction built it there), and the interpreter tags
    the current function's error location / dispatches it to the catching
    ``except`` clause (see ``interp``)."""

    value: Value


@dataclass(eq=False)
class ExceptBind(Inst):
    """The address an ``except E as e`` clause caught its exception through:
    the value of the clause's error-payload ``Phi`` - the one place the caught
    exception lives, written by whoever raised it.  The clause's ``as`` name is
    bound to this value directly, so reading it loads the exception and
    ``ref(e)`` is the pointer itself (see ``interp``)."""


@dataclass(eq=False)
class AsBool(Inst):
    """Converts a value to a boolean."""
    value: ArgEntry[Value]

@dataclass(eq=False)
class Len(Inst):
    """``len(x)``: the number of elements of ``x``, as a register holding an
    *untyped* integer (the destination gives it its type, like an integer
    literal).  The operand is logically a *value*: a tuple (``ComptimeTuple``,
    or its ``ComptimeTuplePtr`` place form when passed by reference) yields its
    length directly, and a struct is asked through its ``__len__`` method (see
    ``interp``)."""

    value: ArgEntry[Value]

@dataclass(eq=False)
class IsNull(Inst):
    """Whether the option value ``opt`` denotes is *absent*: the boolean the
    source ``expr is None`` tests (``expr is not None`` negates it with
    :class:`Not`).  The operand has to be an ``Option[T]`` (a bare ``null``
    counts as the absent one); any other type is rejected.  The result is a
    ``bool`` register."""

    opt: ArgEntry[Value]

@dataclass(eq=False)
class OptionPayloadPtr(Inst):
    """The address of the payload of the option ``ptr`` points at - the place
    the value of a *present* ``Option[T]`` lives in (see ``interp``).
    ``ptr`` has to point at an ``Option[T]``; the result points at its child
    ``T``.  A construction taking a payload address marks the option present;
    the ``:=`` unwrap does not (the option may still be absent when the address
    is taken)."""

    ptr: Value

@dataclass(eq=False)
class IsInstance(Inst):
    """Whether the tagged union ``value`` holds is the variant the second
    operand names (the source ``isinstance(value, T)``): the result is a
    ``bool`` register.  ``value`` has to be a tagged union and ``type`` one of
    its variants; any other type is rejected."""

    value: ArgEntry[Value]
    type: Value


@dataclass(eq=False)
class TaggedUnionPayloadPtr(Inst):
    """The address of the payload of the variant ``type`` of the tagged union
    the pointer ``ptr`` points at - what the ``isinstance(e := value, T)``
    unwrap binds ``e`` to.  ``ptr`` has to point at a tagged union and ``type``
    has to be one of its variants; the result points at the variant.  Taking the
    address does not change the tag (the tag was written when the value was
    stored)."""

    ptr: Value
    type: Value


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
    :class:`BreakLoop` (or a ``return``/``raise``).  The ``Loop`` instruction
    carries nothing; it only delimits the flat instruction stream.

    ``is_inline`` marks a *compile-time* loop (a loop preceded by the source
    statement ``syntax.unroll()``): the interpreter does not
    emit a back edge but unrolls the body once per compile-time iteration,
    so a :class:`BreakLoop` leaves the whole unrolled sequence and a
    :class:`Continue` jumps to the next unrolled body (see ``interp``)."""

    is_inline: bool = False


@dataclass(eq=False)
class BreakLoop(Inst):
    """Leave the innermost open :class:`Loop` unconditionally: the path
    ends at the code after the loop's matching :class:`End`.  Like a
    ``ret`` the instruction carries nothing - the interpreter ends the
    current path with a jump to the loop's exit block (see ``interp``)."""


@dataclass(eq=False)
class Continue(Inst):
    """Start the next iteration of the innermost open :class:`Loop`
    unconditionally: the path ends back at the loop's head, so the rest
    of the body (and the ``else`` clause of the ``while``) is skipped and
    the loop's condition is evaluated again.  Like :class:`BreakLoop` the
    instruction carries nothing."""


@dataclass(eq=False)
class Block(Inst):
    """The start of a ``block`` (WASM-style, like :class:`If`/:
    class:`Loop`): the instructions of the body follow it in the same list,
    closed by the matching :class:`End`.  Unlike a loop it has no back edge:
    falling off its end continues right after the ``End``, and a
    :class:`BreakIf` leaves it early - the interpreter types the code after
    the ``End`` in one block that both the falling end and the breaks reach.
    The ``Block`` instruction carries nothing; it only delimits the flat
    instruction stream.

    ``and``/``or`` chains are lowered into one: every operand but the last
    is stored as the chain's result and tested with a :class:`BreakIf` (the
    chain short-circuits by leaving the block), and an ``if``/``while`` whose
    condition is an ``and`` chain puts its branch body inside the block, so
    the operands' ``:=`` bindings dominate it (see ``astgen``)."""


@dataclass(eq=False)
class BreakIf(Inst):
    """Leave ``levels`` enclosing :class:`Block` blocks when the boolean
    ``cond`` is true (``None`` leaves them unconditionally), the incoming
    control continuing right after the ``End`` of the outermost of them - the
    ``Block`` counterpart of :class:`BreakLoop`, which leaves a
    :class:`Loop`.  ``levels`` counts ``Block`` blocks only (1 is the
    innermost), so a ``break`` of a ``Loop`` inside a ``Block`` is unaffected.
    A conditional break splits the block being typed: the taken edge leaves
    the blocks, the other continues in a block of its own (see ``interp``)."""

    cond: Value | None
    levels: int = 1


@dataclass(eq=False)
class Defer(Inst):
    """The start of a ``with syntax.defer():`` region (WASM-style, like
    :class:`If`/`Loop`/`Block`/`Try`): the instructions of the defer body follow
    it in the same list, closed by the matching :class:`End`.  The body is not
    executed where it is written - it is *deferred* to the exit of the enclosing
    region - so the interpreter emits it into a detached block tree and the
    transfers that leave the region carry its entry (see ``interp``/``mir``).

    ``flags`` is the compile-time integer value the exits are read off: it is
    evaluated where the region is opened (a ``hir.Value``, so an expression such
    as ``syntax.UNWIND | syntax.OK`` is allowed), and its bits say which exits
    trigger the body (see :class:`DeferPath`): ``ALL`` (the ``syntax.defer()``
    default) runs on any exit of the enclosing region, ``OK`` only on a normal
    one (a ``return``, a ``break``/``continue``, or falling off the region's
    end), ``ERR`` only when an error or a panic leaves it."""

    flags: Value


@dataclass(eq=False)
class Try(Inst):
    """Open a ``try`` block: the instructions of the try body follow, then one
    :class:`Except` marker per ``except`` clause (each followed by its clause
    body), and the matching :class:`End`.  ``binds`` holds the ``as`` name's
    slot of every clause (None when the clause names none), created *before*
    the ``Try`` so that the clause body can read it - the interpreter fills it
    with the caught exception when the clause runs (see ``interp``).
    ``except_types`` holds every clause's type expression, *evaluated as a
    value* before the ``Try`` (None for a bare ``except:``): it is the
    exception struct the clause catches, which the interpreter resolves once
    the ``Try`` runs - every dispatch asks it which clause catches an error
    (see ``interp``)."""

    binds: tuple[Value | None, ...]
    except_types: tuple[Value | None, ...]


@dataclass(eq=False)
class Slice(Inst):
    """Build the ``std.slice`` object of a slice subscript (``p[a:b:c]``): its
    result is the slice *object* - a compile-time aggregate of the bounds, of no
    runtime shape of its own - which the subscript of a multi pointer then turns
    into a ``SlicePtr`` (see ``interp``).  The bounds are value operands; a bound
    the source left out is a null constant, held as the absent option value of
    the ``std.slice`` field (see ``interp``)."""

    lower: Value
    upper: Value
    step: Value


@dataclass(eq=False)
class Except(Inst):
    """The marker that starts one ``except`` clause of the innermost open
    :class:`Try` block: ``index`` is the clause's position among the try's
    clauses (its exception type, handler and ``as`` slot are held by the
    enclosing block's ``TryExceptBlockData``; the type is carried by
    ``hir.Try.except_types``)."""

    index: int


@dataclass(eq=False)
class End(Inst):
    """The marker that closes a block opened by an :class:`If`, a
    :class:`Loop`, a :class:`Block` or a :class:`Try`: everything between the
    ``If`` (or its :class:`Else`) and this marker is one branch body,
    everything between the ``Loop``/``Block`` and this marker is its body, and
    the code after this marker is the continuation of the enclosing block.  A
    marker produces no register; it only delimits the flat instruction
    stream."""

@dataclass(eq=False)
class CommitSlot(Inst):
    slot: Value


def scan_block(insts: tuple[Inst, ...], entry: int) -> tuple[int | None, int]:
    """The positions of the ``Else`` (or None when the block has no
    else branch) and ``End`` markers that close the block opened at
    entry (an ``hir.If``, an ``hir.Loop`` or an ``hir.Block``) of the executing
    frame's flat instruction list, found by a balanced scan forward from the
    entry (nested blocks close their own markers first).  A ``Loop``/``Block``
    has no ``Else`` marker of its own, so its ``Else`` is always None."""
    depth = 0
    p_else: int | None = None
    for i in range(entry + 1, len(insts)):
        inst = insts[i]
        if isinstance(inst, (If, Loop, Block, Try, Defer, TypeOfBegin)):
            depth += 1
        elif isinstance(inst, (End, TypeOfEnd)):
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
        if isinstance(inst, (If, Loop, Block, Try, Defer, TypeOfBegin)):
            depth += 1
        elif isinstance(inst, (End, TypeOfEnd)):
            if depth == 0:
                return excepts, i
            depth -= 1
        elif isinstance(inst, Except) and depth == 0:
            excepts.append(i)
    assert False, 'unclosed block in the HIR'
