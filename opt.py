"""Cleanup passes over the typed MIR (``mir``), run once the
interpreter has finished typing a function body (see ``dsl``).

The interpreter delivers every value an inlined function returns
through the shared memory of the call's result location (see
``interp``) - the mechanism a body whose runtime paths return on
several branches needs, since only the memory can join them.  A body
that returns on a single path (or whose runtime branches all fall
through to one trailing return) stores only once, and its
store/load round trip is pure overhead: this module folds such slots
back into registers - the single stored value replaces every load -
and drops slots that are never read.

The fold is deliberately conservative.  A slot (its ``mir.Alloca``) is
folded only when

* its address never escapes (its uses are exactly one ``Store`` of it
  and ``Load``s of it),
* the store textually precedes every load with no control construct in
  between that could bypass the store (no ``If``/``Block``/``Else``/
  ``Break``/``Ret`` between them, and no ``If`` whose region contains
  the store but not the load - the other arm of such an ``If`` reaches
  the load without the store), and
* no ``Break`` jumps over the store into a load (a ``Break`` from
  before the store that lands between the store and a load would also
  reach the load without the store).

Everything else keeps its memory.
"""

from . import mir

_CTRL = (mir.If, mir.Block, mir.Else, mir.End, mir.Break, mir.Ret)


def simplify(fn: mir.Function) -> None:
    """Fold the single-store slots of the body of ``fn`` (rewrites
    ``fn.insts`` in place)."""
    insts = fn.insts
    if len(insts) < 2:
        return

    # pass 1: the balanced block structure - for every If/Block opener
    # its own position and the (Else, End) marker positions (Else is
    # None for a Block, or for an If without an else branch)
    ends: dict[object, tuple[int, int | None, int]] = {}
    else_at: dict[object, int] = {}
    stack: list[tuple[object, int]] = []
    for i, inst in enumerate(insts):
        if isinstance(inst, (mir.If, mir.Block)):
            stack.append((inst, i))
        elif isinstance(inst, mir.Else):
            else_at[stack[-1][0]] = i
        elif isinstance(inst, mir.End):
            opener, opener_idx = stack.pop()
            ends[opener] = (opener_idx, else_at.get(opener), i)
    assert not stack, 'unbalanced block markers in the MIR'

    # pass 2: the jump target of every Break - the index just past the
    # ``End`` of the level-th enclosing block (innermost (1) first); all
    # End positions are known now
    jumps: list[tuple[int, int]] = []
    openers: list[object] = []
    for i, inst in enumerate(insts):
        if isinstance(inst, (mir.If, mir.Block)):
            openers.append(inst)
        elif isinstance(inst, mir.End):
            openers.pop()
        elif isinstance(inst, mir.Break):
            target = ends[openers[-inst.level]][2] + 1
            jumps.append((i, target))

    # every use of every alloca pointer, by its role in the using
    # instruction: 'store'/'load' - the two uses a foldable slot may
    # have - or anything else (a value use, an escaping address, ...)
    roles: dict[mir.Alloca, list[tuple[mir.Inst, str]]] = {}
    for inst in insts:
        for operand, role in _operands(inst):
            if isinstance(operand, mir.Alloca):
                roles.setdefault(operand, []).append((inst, role))
    idx_of = {inst: i for i, inst in enumerate(insts)}

    remove: set[int] = set()
    # a folded load and the value that replaces its uses
    repl: dict[mir.Load, mir.Value] = {}
    for slot, uses in roles.items():
        kinds = {role for _, role in uses}
        if kinds - {'store', 'load'}:
            # the address escapes (or the slot value is used): memory is
            # genuinely needed
            continue
        stores = [
            inst for inst, role in uses if role == 'store' and isinstance(inst, mir.Store)
        ]
        if len(stores) > 1:
            # several paths write the slot (a runtime join): not foldable
            continue
        loads = [
            inst for inst, role in uses if role == 'load' and isinstance(inst, mir.Load)
        ]
        if not stores:
            # a slot that is never written is never read either (an
            # uninitialized read is rejected at compile time): dead
            if not loads:
                remove.add(idx_of[slot])
            continue
        store = stores[0]
        store_idx = idx_of[store]
        if not all(
            _foldable(insts, ends, jumps, store_idx, idx_of[load])
            for load in loads
        ):
            continue
        # fold: the stored value replaces every load, and the slot, its
        # store and the loads disappear
        for load in loads:
            remove.add(idx_of[load])
            repl[load] = store.value
        remove.add(store_idx)
        remove.add(idx_of[slot])
    # an alloca that nothing references at all is dead
    for i, inst in enumerate(insts):
        if isinstance(inst, mir.Alloca) and inst not in roles:
            remove.add(i)
    if not remove and not repl:
        return

    def resolve(value: mir.Value) -> mir.Value:
        seen: set[object] = set()
        while isinstance(value, mir.Load) and value in repl:
            if value in seen:
                raise AssertionError('cycle in the load replacement map')
            seen.add(value)
            value = repl[value]
        return value

    def rewrite(inst: mir.Inst) -> None:
        match inst:
            case mir.Load():
                inst.ptr = resolve(inst.ptr)
            case mir.Store():
                inst.ptr = resolve(inst.ptr)
                inst.value = resolve(inst.value)
            case mir.Gep():
                inst.ptr = resolve(inst.ptr)
            case mir.Arith():
                inst.lhs = resolve(inst.lhs)
                inst.rhs = resolve(inst.rhs)
            case mir.Convert():
                inst.value = resolve(inst.value)
            case mir.Cmp():
                inst.lhs = resolve(inst.lhs)
                inst.rhs = resolve(inst.rhs)
            case mir.Call():
                inst.callee = resolve(inst.callee)
                inst.args = tuple(resolve(arg) for arg in inst.args)
            case mir.Ret():
                if inst.value is not None:
                    inst.value = resolve(inst.value)
            case mir.If():
                inst.cond = resolve(inst.cond)
            case _:
                pass

    out: list[mir.Inst] = []
    for i, inst in enumerate(insts):
        if i in remove:
            continue
        rewrite(inst)
        out.append(inst)
    fn.insts[:] = out


def _foldable(
    insts: list[mir.Inst],
    ends: dict[object, tuple[int, int | None, int]],
    jumps: list[tuple[int, int]],
    store_idx: int,
    load_idx: int,
) -> bool:
    """Whether every runtime path to the load at ``load_idx`` passes the
    store at ``store_idx`` (the conservative conditions listed in the
    module docstring)."""
    if store_idx >= load_idx:
        return False
    # between the store and the load only plain instructions and ``End``
    # markers (of blocks opened before the store) may sit: anything that
    # opens, branches or terminates a region could bypass the store
    for i in range(store_idx + 1, load_idx):
        if isinstance(insts[i], _CTRL) and not isinstance(insts[i], mir.End):
            return False
    for (opener_idx, else_idx, end_idx) in ends.values():
        if not isinstance(insts[opener_idx], mir.If):
            continue
        if not opener_idx < store_idx < end_idx:
            continue
        # the store sits inside this ``If``: a path through another arm
        # reaches the load without the store unless the load sits in the
        # same arm, after the store
        if else_idx is None:
            if not store_idx < load_idx < end_idx:
                return False
        elif store_idx < else_idx:
            if not store_idx < load_idx < else_idx:
                return False
        else:
            if not else_idx < store_idx < load_idx < end_idx:
                return False
    for break_idx, target in jumps:
        # a ``Break`` before the store that lands between the store and
        # the load reaches the load without the store
        if break_idx < store_idx < target <= load_idx:
            return False
    return True


def _operands(inst: mir.Inst) -> tuple[tuple[mir.Value, str], ...]:
    """The operand values of one instruction, tagged by their role in
    it: the ``ptr`` of a ``Load``/``Store`` counts as 'load'/'store',
    every other reference counts as 'use'."""
    match inst:
        case mir.Load():
            return ((inst.ptr, 'load'),)
        case mir.Store():
            return ((inst.ptr, 'store'), (inst.value, 'use'))
        case mir.Gep():
            return ((inst.ptr, 'use'),)
        case mir.Arith():
            return ((inst.lhs, 'use'), (inst.rhs, 'use'))
        case mir.Convert():
            return ((inst.value, 'use'),)
        case mir.Cmp():
            return ((inst.lhs, 'use'), (inst.rhs, 'use'))
        case mir.Call():
            return ((inst.callee, 'use'),) + tuple(
                (arg, 'use') for arg in inst.args
            )
        case mir.Ret():
            return ((inst.value, 'use'),) if inst.value is not None else ()
        case mir.If():
            return ((inst.cond, 'use'),)
        case _:
            return ()
