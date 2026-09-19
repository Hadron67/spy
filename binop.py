from typing import Literal

type UnaryOp = Literal['-', 'not']
type BinaryOp = Literal['+', '-', '*', '/', '//', '%', '**']
type CompareOp = Literal['==', '!=', '<', '<=', '>', '>=']
type BoolOp = Literal['and', 'or']
