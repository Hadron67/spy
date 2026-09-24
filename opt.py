"""Cleanup passes over the typed MIR (``mir``), run once the
interpreter has finished typing a function body (see
``fn.CompileBatch.compile``).

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
  and ``Load``s of it), and
* the store's block *dominates* every load's block (for a load in the
  same block, the store precedes it): every runtime path to a load then
  passes the store, so the load can never read an uninitialized slot.

A slot that fails these conditions is left in memory and folded no
further; one that is never read (together with the store that writes it)
or is never referenced at all is deleted instead.
"""

from . import mir


def simplify(fn: mir.Function) -> None:
    """Fold the single-store slots of the body of ``fn`` (rewrites the
    blocks of ``fn`` in place)."""
    blocks = fn.entry.collect_blocks()
    all_insts = [inst for block in blocks for inst in block.insts]
    if len(all_insts) < 2:
        return

    # the block and the position in it of every instruction - the blocks
    # of one function hold each instruction exactly once
    index_of: dict[mir.Inst, tuple[mir.BasicBlock, int]] = {}
    for block in blocks:
        for i, inst in enumerate(block.insts):
            index_of[inst] = (block, i)

    # every use of every alloca pointer, by its role in the using
    # instruction: 'store'/'load' - the two uses a foldable slot may
    # have - or anything else (a value use, an escaping address, ...)
    roles: dict[mir.Alloca, list[tuple[mir.Inst, str]]] = {}
    for inst in all_insts:
        for operand, role in _operands(inst):
            if isinstance(operand, mir.Alloca):
                roles.setdefault(operand, []).append((inst, role))

    dominators = _dominators(blocks)

    remove: set[mir.Inst] = set()
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
                remove.add(slot)
            continue
        store = stores[0]
        if not all(
            _foldable(dominators, index_of, store, load) for load in loads
        ):
            continue
        # fold: the stored value replaces every load, and the slot, its
        # store and the loads disappear
        for load in loads:
            remove.add(load)
            repl[load] = store.value
        remove.add(store)
        remove.add(slot)
    # an alloca that nothing references at all is dead
    for inst in all_insts:
        if isinstance(inst, mir.Alloca) and inst not in roles:
            remove.add(inst)
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
                if not isinstance(inst.index, int):
                    inst.index = resolve(inst.index)
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
            case mir.Br():
                inst.cond = resolve(inst.cond)
            case _:
                pass

    for block in blocks:
        out: list[mir.Inst] = []
        for inst in block.insts:
            if inst in remove:
                continue
            rewrite(inst)
            out.append(inst)
        block.insts = out


def _dominators(blocks: list[mir.BasicBlock]) -> dict[mir.BasicBlock, set[mir.BasicBlock]]:
    """The dominator set of every block: ``dominators[b]`` holds ``b`` and
    every block that lies on every path from the entry to ``b``.  Computed
    by the classic iterative dataflow fixpoint (the graph is small and,
    without loops, converges quickly)."""
    entry = blocks[0]
    preds: dict[mir.BasicBlock, list[mir.BasicBlock]] = {block: [] for block in blocks}
    for block in blocks:
        for successor in block.get_outgoing_blocks():
            preds[successor].append(block)

    all_blocks = set(blocks)
    dominators: dict[mir.BasicBlock, set[mir.BasicBlock]] = {
        block: ({block} if block is entry else set(all_blocks))
        for block in blocks
    }
    changed = True
    while changed:
        changed = False
        for block in blocks:
            if block is entry:
                continue
            new: set[mir.BasicBlock] = set(all_blocks)
            for pred in preds[block]:
                new &= dominators[pred]
            new.add(block)
            if new != dominators[block]:
                dominators[block] = new
                changed = True
    return dominators


def _foldable(
    dominators: dict[mir.BasicBlock, set[mir.BasicBlock]],
    index_of: dict[mir.Inst, tuple[mir.BasicBlock, int]],
    store: mir.Store,
    load: mir.Load,
) -> bool:
    """Whether every runtime path to the load passes the store: the store
    is earlier in the same block, or its block dominates the load's."""
    store_block, store_idx = index_of[store]
    load_block, load_idx = index_of[load]
    if store_block is load_block:
        return store_idx < load_idx
    return store_block in dominators[load_block]


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
            if isinstance(inst.index, int):
                return ((inst.ptr, 'use'),)
            return ((inst.ptr, 'use'), (inst.index, 'use'))
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
        case mir.Br():
            return ((inst.cond, 'use'),)
        case _:
            return ()
