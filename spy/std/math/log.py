"""The software (pure spy) natural logarithm (``std.math``).

A port of musl's ``log.c`` and ``logf.c`` (the Arm Limited implementations,
MIT), using the full ``log_data.c`` / ``logf_data.c`` tables (the non-FMA
``tab2`` path included).

* ``f64`` follows ``log.c``: inputs close to 1 are handled by the dedicated
  ``poly1`` polynomial with the ``rhi``/``rlo`` split, everything else is
  reduced to ``x = 2**k * z`` with ``z`` in ``[OFF, 2*OFF)`` and evaluated as
  ``k*ln2 + log(c) + log(z/c)`` around the center ``c`` of a 128-entry table;
  the ``tab2`` table supplies the ``chi``/``clo`` split that makes
  ``(z - chi - clo) * invc`` accurate without an FMA.
* ``f32`` follows ``logf.c``: a 16-entry table and a degree-4 polynomial,
  evaluated in ``f64`` and rounded once, at the end, to ``f32``.

The public entry point is :func:`log`, generic over ``f32``/``f64``.  It is what
``std.math`` calls when the compile-time option ``log_mode`` is ``software``
(the default).

The one deviation from musl is its floating point environment handling: spy has
no ``fenv``, so ``__math_divzero(1)`` is ``-inf``, ``__math_invalid`` is the NaN
``0/0`` and ``WANT_ROUNDING`` is 1 (so ``log(1)`` is ``+0``).  musl bounds the
error of ``log`` and ``logf`` at about 0.51 and 0.82 ULP, which the arithmetic
below reproduces (it is the non-FMA path).
"""

from ...compiler import f32, f64, i32, i64, typeof, u32, u64, usize
from ...compiler.dsl import func
from ...compiler.syntax import array, comptime, const_comptime
from ..core import Numeric, bitcast, coerce, compile_error, int_to_float_unchecked
from .pow import _hex, _invalid, _invalidf

# ``log_data.c``: the split ``ln2``, the two polynomials and the reduction
# offset.  ``_B*`` is ``poly1`` (used near 1), ``_A*`` is ``poly`` (used by the
# table path).  ``_LOG_LO``/``_LOG_HI`` are ``asuint64(1.0 - 0x1p-4)`` and
# ``asuint64(1.0 + 0x1.09p-4)``.
_LN2HI = _hex(b'0x1.62e42fefa3800p-1')
_LN2LO = _hex(b'0x1.ef35793c76730p-45')

_B0 = _hex(b'-0x1p-1')
_B1 = _hex(b'0x1.5555555555577p-2')
_B2 = _hex(b'-0x1.ffffffffffdcbp-3')
_B3 = _hex(b'0x1.999999995dd0cp-3')
_B4 = _hex(b'-0x1.55555556745a7p-3')
_B5 = _hex(b'0x1.24924a344de3p-3')
_B6 = _hex(b'-0x1.fffffa4423d65p-4')
_B7 = _hex(b'0x1.c7184282ad6cap-4')
_B8 = _hex(b'-0x1.999eb43b068ffp-4')
_B9 = _hex(b'0x1.78182f7afd085p-4')
_B10 = _hex(b'-0x1.5521375d145cdp-4')

_A0 = _hex(b'-0x1.0000000000001p-1')
_A1 = _hex(b'0x1.555555551305bp-2')
_A2 = _hex(b'-0x1.fffffffeb459p-3')
_A3 = _hex(b'0x1.999b324f10111p-3')
_A4 = _hex(b'-0x1.55575e506c89fp-3')

_LOG_OFF = 0x3FE6000000000000
_LOG_LO = 0x3FEE000000000000
_LOG_HI = 0x3FF1090000000000
_LOG_NEAR = _LOG_HI - _LOG_LO

# ``logf_data.c``: the base-2 logarithm and the degree-4 polynomial.
_LOGF_LN2 = _hex(b'0x1.62e42fefa39efp-1')
_LOGF_A0 = _hex(b'-0x1.00ea348b88334p-2')
_LOGF_A1 = _hex(b'0x1.5575b0be00b6ap-2')
_LOGF_A2 = _hex(b'-0x1.ffffef20a4123p-2')

_LOG_OFF32 = 0x3F330000

# powers of two used to normalize a subnormal argument, and the split width of
# the near-1 path
_P52 = _hex(b'0x1p52')
_P23F = _hex(b'0x1p23')
_P27 = _hex(b'0x1p27')

_NEG_INF = float('-inf')


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


@func()
def _log_tab(index: usize) -> tuple[f64, f64]:
    """``__log_data.tab[index]`` (musl's ``log_data.c``) as ``(invc, logc)``."""
    const_comptime()
    tab = array(
        _hex(b'0x1.734f0c3e0de9fp+0'), _hex(b'-0x1.7cc7f79e69000p-2'), _hex(b'0x1.713786a2ce91fp+0'), _hex(b'-0x1.76feec20d0000p-2'),
        _hex(b'0x1.6f26008fab5a0p+0'), _hex(b'-0x1.713e31351e000p-2'), _hex(b'0x1.6d1a61f138c7dp+0'), _hex(b'-0x1.6b85b38287800p-2'),
        _hex(b'0x1.6b1490bc5b4d1p+0'), _hex(b'-0x1.65d5590807800p-2'), _hex(b'0x1.69147332f0cbap+0'), _hex(b'-0x1.602d076180000p-2'),
        _hex(b'0x1.6719f18224223p+0'), _hex(b'-0x1.5a8ca86909000p-2'), _hex(b'0x1.6524f99a51ed9p+0'), _hex(b'-0x1.54f4356035000p-2'),
        _hex(b'0x1.63356aa8f24c4p+0'), _hex(b'-0x1.4f637c36b4000p-2'), _hex(b'0x1.614b36b9ddc14p+0'), _hex(b'-0x1.49da7fda85000p-2'),
        _hex(b'0x1.5f66452c65c4cp+0'), _hex(b'-0x1.445923989a800p-2'), _hex(b'0x1.5d867b5912c4fp+0'), _hex(b'-0x1.3edf439b0b800p-2'),
        _hex(b'0x1.5babccb5b90dep+0'), _hex(b'-0x1.396ce448f7000p-2'), _hex(b'0x1.59d61f2d91a78p+0'), _hex(b'-0x1.3401e17bda000p-2'),
        _hex(b'0x1.5805612465687p+0'), _hex(b'-0x1.2e9e2ef468000p-2'), _hex(b'0x1.56397cee76bd3p+0'), _hex(b'-0x1.2941b3830e000p-2'),
        _hex(b'0x1.54725e2a77f93p+0'), _hex(b'-0x1.23ec58cda8800p-2'), _hex(b'0x1.52aff42064583p+0'), _hex(b'-0x1.1e9e129279000p-2'),
        _hex(b'0x1.50f22dbb2bddfp+0'), _hex(b'-0x1.1956d2b48f800p-2'), _hex(b'0x1.4f38f4734ded7p+0'), _hex(b'-0x1.141679ab9f800p-2'),
        _hex(b'0x1.4d843cfde2840p+0'), _hex(b'-0x1.0edd094ef9800p-2'), _hex(b'0x1.4bd3ec078a3c8p+0'), _hex(b'-0x1.09aa518db1000p-2'),
        _hex(b'0x1.4a27fc3e0258ap+0'), _hex(b'-0x1.047e65263b800p-2'), _hex(b'0x1.4880524d48434p+0'), _hex(b'-0x1.feb224586f000p-3'),
        _hex(b'0x1.46dce1b192d0bp+0'), _hex(b'-0x1.f474a7517b000p-3'), _hex(b'0x1.453d9d3391854p+0'), _hex(b'-0x1.ea4443d103000p-3'),
        _hex(b'0x1.43a2744b4845ap+0'), _hex(b'-0x1.e020d44e9b000p-3'), _hex(b'0x1.420b54115f8fbp+0'), _hex(b'-0x1.d60a22977f000p-3'),
        _hex(b'0x1.40782da3ef4b1p+0'), _hex(b'-0x1.cc00104959000p-3'), _hex(b'0x1.3ee8f5d57fe8fp+0'), _hex(b'-0x1.c202956891000p-3'),
        _hex(b'0x1.3d5d9a00b4ce9p+0'), _hex(b'-0x1.b81178d811000p-3'), _hex(b'0x1.3bd60c010c12bp+0'), _hex(b'-0x1.ae2c9ccd3d000p-3'),
        _hex(b'0x1.3a5242b75dab8p+0'), _hex(b'-0x1.a45402e129000p-3'), _hex(b'0x1.38d22cd9fd002p+0'), _hex(b'-0x1.9a877681df000p-3'),
        _hex(b'0x1.3755bc5847a1cp+0'), _hex(b'-0x1.90c6d69483000p-3'), _hex(b'0x1.35dce49ad36e2p+0'), _hex(b'-0x1.87120a645c000p-3'),
        _hex(b'0x1.34679984dd440p+0'), _hex(b'-0x1.7d68fb4143000p-3'), _hex(b'0x1.32f5cceffcb24p+0'), _hex(b'-0x1.73cb83c627000p-3'),
        _hex(b'0x1.3187775a10d49p+0'), _hex(b'-0x1.6a39a9b376000p-3'), _hex(b'0x1.301c8373e3990p+0'), _hex(b'-0x1.60b3154b7a000p-3'),
        _hex(b'0x1.2eb4ebb95f841p+0'), _hex(b'-0x1.5737d76243000p-3'), _hex(b'0x1.2d50a0219a9d1p+0'), _hex(b'-0x1.4dc7b8fc23000p-3'),
        _hex(b'0x1.2bef9a8b7fd2ap+0'), _hex(b'-0x1.4462c51d20000p-3'), _hex(b'0x1.2a91c7a0c1babp+0'), _hex(b'-0x1.3b08abc830000p-3'),
        _hex(b'0x1.293726014b530p+0'), _hex(b'-0x1.31b996b490000p-3'), _hex(b'0x1.27dfa5757a1f5p+0'), _hex(b'-0x1.2875490a44000p-3'),
        _hex(b'0x1.268b39b1d3bbfp+0'), _hex(b'-0x1.1f3b9f879a000p-3'), _hex(b'0x1.2539d838ff5bdp+0'), _hex(b'-0x1.160c8252ca000p-3'),
        _hex(b'0x1.23eb7aac9083bp+0'), _hex(b'-0x1.0ce7f57f72000p-3'), _hex(b'0x1.22a012ba940b6p+0'), _hex(b'-0x1.03cdc49fea000p-3'),
        _hex(b'0x1.2157996cc4132p+0'), _hex(b'-0x1.f57bdbc4b8000p-4'), _hex(b'0x1.201201dd2fc9bp+0'), _hex(b'-0x1.e370896404000p-4'),
        _hex(b'0x1.1ecf4494d480bp+0'), _hex(b'-0x1.d17983ef94000p-4'), _hex(b'0x1.1d8f5528f6569p+0'), _hex(b'-0x1.bf9674ed8a000p-4'),
        _hex(b'0x1.1c52311577e7cp+0'), _hex(b'-0x1.adc79202f6000p-4'), _hex(b'0x1.1b17c74cb26e9p+0'), _hex(b'-0x1.9c0c3e7288000p-4'),
        _hex(b'0x1.19e010c2c1ab6p+0'), _hex(b'-0x1.8a646b372c000p-4'), _hex(b'0x1.18ab07bb670bdp+0'), _hex(b'-0x1.78d01b3ac0000p-4'),
        _hex(b'0x1.1778a25efbcb6p+0'), _hex(b'-0x1.674f145380000p-4'), _hex(b'0x1.1648d354c31dap+0'), _hex(b'-0x1.55e0e6d878000p-4'),
        _hex(b'0x1.151b990275fddp+0'), _hex(b'-0x1.4485cdea1e000p-4'), _hex(b'0x1.13f0ea432d24cp+0'), _hex(b'-0x1.333d94d6aa000p-4'),
        _hex(b'0x1.12c8b7210f9dap+0'), _hex(b'-0x1.22079f8c56000p-4'), _hex(b'0x1.11a3028ecb531p+0'), _hex(b'-0x1.10e4698622000p-4'),
        _hex(b'0x1.107fbda8434afp+0'), _hex(b'-0x1.ffa6c6ad20000p-5'), _hex(b'0x1.0f5ee0f4e6bb3p+0'), _hex(b'-0x1.dda8d4a774000p-5'),
        _hex(b'0x1.0e4065d2a9fcep+0'), _hex(b'-0x1.bbcece4850000p-5'), _hex(b'0x1.0d244632ca521p+0'), _hex(b'-0x1.9a1894012c000p-5'),
        _hex(b'0x1.0c0a77ce2981ap+0'), _hex(b'-0x1.788583302c000p-5'), _hex(b'0x1.0af2f83c636d1p+0'), _hex(b'-0x1.5715e67d68000p-5'),
        _hex(b'0x1.09ddb98a01339p+0'), _hex(b'-0x1.35c8a49658000p-5'), _hex(b'0x1.08cabaf52e7dfp+0'), _hex(b'-0x1.149e364154000p-5'),
        _hex(b'0x1.07b9f2f4e28fbp+0'), _hex(b'-0x1.e72c082eb8000p-6'), _hex(b'0x1.06ab58c358f19p+0'), _hex(b'-0x1.a55f152528000p-6'),
        _hex(b'0x1.059eea5ecf92cp+0'), _hex(b'-0x1.63d62cf818000p-6'), _hex(b'0x1.04949cdd12c90p+0'), _hex(b'-0x1.228fb8caa0000p-6'),
        _hex(b'0x1.038c6c6f0ada9p+0'), _hex(b'-0x1.c317b20f90000p-7'), _hex(b'0x1.02865137932a9p+0'), _hex(b'-0x1.419355daa0000p-7'),
        _hex(b'0x1.0182427ea7348p+0'), _hex(b'-0x1.81203c2ec0000p-8'), _hex(b'0x1.008040614b195p+0'), _hex(b'-0x1.0040979240000p-9'),
        _hex(b'0x1.fe01ff726fa1ap-1'), _hex(b'0x1.feff384900000p-9'), _hex(b'0x1.fa11cc261ea74p-1'), _hex(b'0x1.7dc41353d0000p-7'),
        _hex(b'0x1.f6310b081992ep-1'), _hex(b'0x1.3cea3c4c28000p-6'), _hex(b'0x1.f25f63ceeadcdp-1'), _hex(b'0x1.b9fc114890000p-6'),
        _hex(b'0x1.ee9c8039113e7p-1'), _hex(b'0x1.1b0d8ce110000p-5'), _hex(b'0x1.eae8078cbb1abp-1'), _hex(b'0x1.58a5bd001c000p-5'),
        _hex(b'0x1.e741aa29d0c9bp-1'), _hex(b'0x1.95c8340d88000p-5'), _hex(b'0x1.e3a91830a99b5p-1'), _hex(b'0x1.d276aef578000p-5'),
        _hex(b'0x1.e01e009609a56p-1'), _hex(b'0x1.07598e598c000p-4'), _hex(b'0x1.dca01e577bb98p-1'), _hex(b'0x1.253f5e30d2000p-4'),
        _hex(b'0x1.d92f20b7c9103p-1'), _hex(b'0x1.42edd8b380000p-4'), _hex(b'0x1.d5cac66fb5ccep-1'), _hex(b'0x1.606598757c000p-4'),
        _hex(b'0x1.d272caa5ede9dp-1'), _hex(b'0x1.7da76356a0000p-4'), _hex(b'0x1.cf26e3e6b2ccdp-1'), _hex(b'0x1.9ab434e1c6000p-4'),
        _hex(b'0x1.cbe6da2a77902p-1'), _hex(b'0x1.b78c7bb0d6000p-4'), _hex(b'0x1.c8b266d37086dp-1'), _hex(b'0x1.d431332e72000p-4'),
        _hex(b'0x1.c5894bd5d5804p-1'), _hex(b'0x1.f0a3171de6000p-4'), _hex(b'0x1.c26b533bb9f8cp-1'), _hex(b'0x1.067152b914000p-3'),
        _hex(b'0x1.bf583eeece73fp-1'), _hex(b'0x1.147858292b000p-3'), _hex(b'0x1.bc4fd75db96c1p-1'), _hex(b'0x1.2266ecdca3000p-3'),
        _hex(b'0x1.b951e0c864a28p-1'), _hex(b'0x1.303d7a6c55000p-3'), _hex(b'0x1.b65e2c5ef3e2cp-1'), _hex(b'0x1.3dfc33c331000p-3'),
        _hex(b'0x1.b374867c9888bp-1'), _hex(b'0x1.4ba366b7a8000p-3'), _hex(b'0x1.b094b211d304ap-1'), _hex(b'0x1.5933928d1f000p-3'),
        _hex(b'0x1.adbe885f2ef7ep-1'), _hex(b'0x1.66acd2418f000p-3'), _hex(b'0x1.aaf1d31603da2p-1'), _hex(b'0x1.740f8ec669000p-3'),
        _hex(b'0x1.a82e63fd358a7p-1'), _hex(b'0x1.815c0f51af000p-3'), _hex(b'0x1.a5740ef09738bp-1'), _hex(b'0x1.8e92954f68000p-3'),
        _hex(b'0x1.a2c2a90ab4b27p-1'), _hex(b'0x1.9bb3602f84000p-3'), _hex(b'0x1.a01a01393f2d1p-1'), _hex(b'0x1.a8bed1c2c0000p-3'),
        _hex(b'0x1.9d79f24db3c1bp-1'), _hex(b'0x1.b5b515c01d000p-3'), _hex(b'0x1.9ae2505c7b190p-1'), _hex(b'0x1.c2967ccbcc000p-3'),
        _hex(b'0x1.9852ef297ce2fp-1'), _hex(b'0x1.cf635d5486000p-3'), _hex(b'0x1.95cbaeea44b75p-1'), _hex(b'0x1.dc1bd3446c000p-3'),
        _hex(b'0x1.934c69de74838p-1'), _hex(b'0x1.e8c01b8cfe000p-3'), _hex(b'0x1.90d4f2f6752e6p-1'), _hex(b'0x1.f5509c0179000p-3'),
        _hex(b'0x1.8e6528effd79dp-1'), _hex(b'0x1.00e6c121fb800p-2'), _hex(b'0x1.8bfce9fcc007cp-1'), _hex(b'0x1.071b80e93d000p-2'),
        _hex(b'0x1.899c0dabec30ep-1'), _hex(b'0x1.0d46b9e867000p-2'), _hex(b'0x1.87427aa2317fbp-1'), _hex(b'0x1.13687334bd000p-2'),
        _hex(b'0x1.84f00acb39a08p-1'), _hex(b'0x1.1980d67234800p-2'), _hex(b'0x1.82a49e8653e55p-1'), _hex(b'0x1.1f8ffe0cc8000p-2'),
        _hex(b'0x1.8060195f40260p-1'), _hex(b'0x1.2595fd7636800p-2'), _hex(b'0x1.7e22563e0a329p-1'), _hex(b'0x1.2b9300914a800p-2'),
        _hex(b'0x1.7beb377dcb5adp-1'), _hex(b'0x1.3187210436000p-2'), _hex(b'0x1.79baa679725c2p-1'), _hex(b'0x1.377266dec1800p-2'),
        _hex(b'0x1.77907f2170657p-1'), _hex(b'0x1.3d54ffbaf3000p-2'), _hex(b'0x1.756cadbd6130cp-1'), _hex(b'0x1.432eee32fe000p-2'),
    )
    return tab[2 * index], tab[2 * index + 1]


@func()
def _log_tab2(index: usize) -> tuple[f64, f64]:
    """``__log_data.tab2[index]`` (musl's ``log_data.c``, the non-FMA path) as
    ``(chi, clo)``."""
    const_comptime()
    tab2 = array(
        _hex(b'0x1.61000014fb66bp-1'), _hex(b'0x1.e026c91425b3cp-56'), _hex(b'0x1.63000034db495p-1'), _hex(b'0x1.dbfea48005d41p-55'),
        _hex(b'0x1.650000d94d478p-1'), _hex(b'0x1.e7fa786d6a5b7p-55'), _hex(b'0x1.67000074e6fadp-1'), _hex(b'0x1.1fcea6b54254cp-57'),
        _hex(b'0x1.68ffffedf0faep-1'), _hex(b'-0x1.c7e274c590efdp-56'), _hex(b'0x1.6b0000763c5bcp-1'), _hex(b'-0x1.ac16848dcda01p-55'),
        _hex(b'0x1.6d0001e5cc1f6p-1'), _hex(b'0x1.33f1c9d499311p-55'), _hex(b'0x1.6efffeb05f63ep-1'), _hex(b'-0x1.e80041ae22d53p-56'),
        _hex(b'0x1.710000e86978p-1'), _hex(b'0x1.bff6671097952p-56'), _hex(b'0x1.72ffffc67e912p-1'), _hex(b'0x1.c00e226bd8724p-55'),
        _hex(b'0x1.74fffdf81116ap-1'), _hex(b'-0x1.e02916ef101d2p-57'), _hex(b'0x1.770000f679c9p-1'), _hex(b'-0x1.7fc71cd549c74p-57'),
        _hex(b'0x1.78ffffa7ec835p-1'), _hex(b'0x1.1bec19ef50483p-55'), _hex(b'0x1.7affffe20c2e6p-1'), _hex(b'-0x1.07e1729cc6465p-56'),
        _hex(b'0x1.7cfffed3fc9p-1'), _hex(b'-0x1.08072087b8b1cp-55'), _hex(b'0x1.7efffe9261a76p-1'), _hex(b'0x1.dc0286d9df9aep-55'),
        _hex(b'0x1.81000049ca3e8p-1'), _hex(b'0x1.97fd251e54c33p-55'), _hex(b'0x1.8300017932c8fp-1'), _hex(b'-0x1.afee9b630f381p-55'),
        _hex(b'0x1.850000633739cp-1'), _hex(b'0x1.9bfbf6b6535bcp-55'), _hex(b'0x1.87000204289c6p-1'), _hex(b'-0x1.bbf65f3117b75p-55'),
        _hex(b'0x1.88fffebf57904p-1'), _hex(b'-0x1.9006ea23dcb57p-55'), _hex(b'0x1.8b00022bc04dfp-1'), _hex(b'-0x1.d00df38e04b0ap-56'),
        _hex(b'0x1.8cfffe50c1b8ap-1'), _hex(b'-0x1.8007146ff9f05p-55'), _hex(b'0x1.8effffc918e43p-1'), _hex(b'0x1.3817bd07a7038p-55'),
        _hex(b'0x1.910001efa5fc7p-1'), _hex(b'0x1.93e9176dfb403p-55'), _hex(b'0x1.9300013467bb9p-1'), _hex(b'0x1.f804e4b980276p-56'),
        _hex(b'0x1.94fffe6ee076fp-1'), _hex(b'-0x1.f7ef0d9ff622ep-55'), _hex(b'0x1.96fffde3c12d1p-1'), _hex(b'-0x1.082aa962638bap-56'),
        _hex(b'0x1.98ffff4458a0dp-1'), _hex(b'-0x1.7801b9164a8efp-55'), _hex(b'0x1.9afffdd982e3ep-1'), _hex(b'-0x1.740e08a5a9337p-55'),
        _hex(b'0x1.9cfffed49fb66p-1'), _hex(b'0x1.fce08c19bep-60'), _hex(b'0x1.9f00020f19c51p-1'), _hex(b'-0x1.a3faa27885b0ap-55'),
        _hex(b'0x1.a10001145b006p-1'), _hex(b'0x1.4ff489958da56p-56'), _hex(b'0x1.a300007bbf6fap-1'), _hex(b'0x1.cbeab8a2b6d18p-55'),
        _hex(b'0x1.a500010971d79p-1'), _hex(b'0x1.8fecadd78793p-55'), _hex(b'0x1.a70001df52e48p-1'), _hex(b'-0x1.f41763dd8abdbp-55'),
        _hex(b'0x1.a90001c593352p-1'), _hex(b'-0x1.ebf0284c27612p-55'), _hex(b'0x1.ab0002a4f3e4bp-1'), _hex(b'-0x1.9fd043cff3f5fp-57'),
        _hex(b'0x1.acfffd7ae1ed1p-1'), _hex(b'-0x1.23ee7129070b4p-55'), _hex(b'0x1.aefffee510478p-1'), _hex(b'0x1.a063ee00edea3p-57'),
        _hex(b'0x1.b0fffdb650d5bp-1'), _hex(b'0x1.a06c8381f0ab9p-58'), _hex(b'0x1.b2ffffeaaca57p-1'), _hex(b'-0x1.9011e74233c1dp-56'),
        _hex(b'0x1.b4fffd995badcp-1'), _hex(b'-0x1.9ff1068862a9fp-56'), _hex(b'0x1.b7000249e659cp-1'), _hex(b'0x1.aff45d0864f3ep-55'),
        _hex(b'0x1.b8ffff987164p-1'), _hex(b'0x1.cfe7796c2c3f9p-56'), _hex(b'0x1.bafffd204cb4fp-1'), _hex(b'-0x1.3ff27eef22bc4p-57'),
        _hex(b'0x1.bcfffd2415c45p-1'), _hex(b'-0x1.cffb7ee3bea21p-57'), _hex(b'0x1.beffff86309dfp-1'), _hex(b'-0x1.14103972e0b5cp-55'),
        _hex(b'0x1.c0fffe1b57653p-1'), _hex(b'0x1.bc16494b76a19p-55'), _hex(b'0x1.c2ffff1fa57e3p-1'), _hex(b'-0x1.4feef8d30c6edp-57'),
        _hex(b'0x1.c4fffdcbfe424p-1'), _hex(b'-0x1.43f68bcec4775p-55'), _hex(b'0x1.c6fffed54b9f7p-1'), _hex(b'0x1.47ea3f053e0ecp-55'),
        _hex(b'0x1.c8fffeb998fd5p-1'), _hex(b'0x1.383068df992f1p-56'), _hex(b'0x1.cb0002125219ap-1'), _hex(b'-0x1.8fd8e64180e04p-57'),
        _hex(b'0x1.ccfffdd94469cp-1'), _hex(b'0x1.e7ebe1cc7ea72p-55'), _hex(b'0x1.cefffeafdc476p-1'), _hex(b'0x1.ebe39ad9f88fep-55'),
        _hex(b'0x1.d1000169af82bp-1'), _hex(b'0x1.57d91a8b95a71p-56'), _hex(b'0x1.d30000d0ff71dp-1'), _hex(b'0x1.9c1906970c7dap-55'),
        _hex(b'0x1.d4fffea790fc4p-1'), _hex(b'-0x1.80e37c558fe0cp-58'), _hex(b'0x1.d70002edc87e5p-1'), _hex(b'-0x1.f80d64dc10f44p-56'),
        _hex(b'0x1.d900021dc82aap-1'), _hex(b'-0x1.47c8f94fd5c5cp-56'), _hex(b'0x1.dafffd86b0283p-1'), _hex(b'0x1.c7f1dc521617ep-55'),
        _hex(b'0x1.dd000296c4739p-1'), _hex(b'0x1.8019eb2ffb153p-55'), _hex(b'0x1.defffe54490f5p-1'), _hex(b'0x1.e00d2c652cc89p-57'),
        _hex(b'0x1.e0fffcdabf694p-1'), _hex(b'-0x1.f8340202d69d2p-56'), _hex(b'0x1.e2fffdb52c8ddp-1'), _hex(b'0x1.b00c1ca1b0864p-56'),
        _hex(b'0x1.e4ffff24216efp-1'), _hex(b'0x1.2ffa8b094ab51p-56'), _hex(b'0x1.e6fffe88a5e11p-1'), _hex(b'-0x1.7f673b1efbe59p-58'),
        _hex(b'0x1.e9000119eff0dp-1'), _hex(b'-0x1.4808d5e0bc801p-55'), _hex(b'0x1.eafffdfa51744p-1'), _hex(b'0x1.80006d54320b5p-56'),
        _hex(b'0x1.ed0001a127fa1p-1'), _hex(b'-0x1.002f860565c92p-58'), _hex(b'0x1.ef00007babcc4p-1'), _hex(b'-0x1.540445d35e611p-55'),
        _hex(b'0x1.f0ffff57a8d02p-1'), _hex(b'-0x1.ffb3139ef9105p-59'), _hex(b'0x1.f30001ee58ac7p-1'), _hex(b'0x1.a81acf2731155p-55'),
        _hex(b'0x1.f4ffff5823494p-1'), _hex(b'0x1.a3f41d4d7c743p-55'), _hex(b'0x1.f6ffffca94c6bp-1'), _hex(b'-0x1.202f41c987875p-57'),
        _hex(b'0x1.f8fffe1f9c441p-1'), _hex(b'0x1.77dd1f477e74bp-56'), _hex(b'0x1.fafffd2e0e37ep-1'), _hex(b'-0x1.f01199a7ca331p-57'),
        _hex(b'0x1.fd0001c77e49ep-1'), _hex(b'0x1.181ee4bceacb1p-56'), _hex(b'0x1.feffff7e0c331p-1'), _hex(b'-0x1.e05370170875ap-57'),
        _hex(b'0x1.00ffff465606ep+0'), _hex(b'-0x1.a7ead491c0adap-55'), _hex(b'0x1.02ffff3867a58p+0'), _hex(b'-0x1.77f69c3fcb2ep-54'),
        _hex(b'0x1.04ffffdfc0d17p+0'), _hex(b'0x1.7bffe34cb945bp-54'), _hex(b'0x1.0700003cd4d82p+0'), _hex(b'0x1.20083c0e456cbp-55'),
        _hex(b'0x1.08ffff9f2cbe8p+0'), _hex(b'-0x1.dffdfbe37751ap-57'), _hex(b'0x1.0b000010cda65p+0'), _hex(b'-0x1.13f7faee626ebp-54'),
        _hex(b'0x1.0d00001a4d338p+0'), _hex(b'0x1.07dfa79489ff7p-55'), _hex(b'0x1.0effffadafdfdp+0'), _hex(b'-0x1.7040570d66bcp-56'),
        _hex(b'0x1.110000bbafd96p+0'), _hex(b'0x1.e80d4846d0b62p-55'), _hex(b'0x1.12ffffae5f45dp+0'), _hex(b'0x1.dbffa64fd36efp-54'),
        _hex(b'0x1.150000dd59ad9p+0'), _hex(b'0x1.a0077701250aep-54'), _hex(b'0x1.170000f21559ap+0'), _hex(b'0x1.dfdf9e2e3deeep-55'),
        _hex(b'0x1.18ffffc275426p+0'), _hex(b'0x1.10030dc3b7273p-54'), _hex(b'0x1.1b000123d3c59p+0'), _hex(b'0x1.97f7980030188p-54'),
        _hex(b'0x1.1cffff8299eb7p+0'), _hex(b'-0x1.5f932ab9f8c67p-57'), _hex(b'0x1.1effff48ad4p+0'), _hex(b'0x1.37fbf9da75bebp-54'),
        _hex(b'0x1.210000c8b86a4p+0'), _hex(b'0x1.f806b91fd5b22p-54'), _hex(b'0x1.2300003854303p+0'), _hex(b'0x1.3ffc2eb9fbf33p-54'),
        _hex(b'0x1.24fffffbcf684p+0'), _hex(b'0x1.601e77e2e2e72p-56'), _hex(b'0x1.26ffff52921d9p+0'), _hex(b'0x1.ffcbb767f0c61p-56'),
        _hex(b'0x1.2900014933a3cp+0'), _hex(b'-0x1.202ca3c02412bp-56'), _hex(b'0x1.2b00014556313p+0'), _hex(b'-0x1.2808233f21f02p-54'),
        _hex(b'0x1.2cfffebfe523bp+0'), _hex(b'-0x1.8ff7e384fdcf2p-55'), _hex(b'0x1.2f0000bb8ad96p+0'), _hex(b'-0x1.5ff51503041c5p-55'),
        _hex(b'0x1.30ffffb7ae2afp+0'), _hex(b'-0x1.10071885e289dp-55'), _hex(b'0x1.32ffffeac5f7fp+0'), _hex(b'-0x1.1ff5d3fb7b715p-54'),
        _hex(b'0x1.350000ca66756p+0'), _hex(b'0x1.57f82228b82bdp-54'), _hex(b'0x1.3700011fbf721p+0'), _hex(b'0x1.000bac40dd5ccp-55'),
        _hex(b'0x1.38ffff9592fb9p+0'), _hex(b'-0x1.43f9d2db2a751p-54'), _hex(b'0x1.3b00004ddd242p+0'), _hex(b'0x1.57f6b707638e1p-55'),
        _hex(b'0x1.3cffff5b2c957p+0'), _hex(b'0x1.a023a10bf1231p-56'), _hex(b'0x1.3efffeab0b418p+0'), _hex(b'0x1.87f6d66b152bp-54'),
        _hex(b'0x1.410001532aff4p+0'), _hex(b'0x1.7f8375f198524p-57'), _hex(b'0x1.4300017478b29p+0'), _hex(b'0x1.301e672dc5143p-55'),
        _hex(b'0x1.44fffe795b463p+0'), _hex(b'0x1.9ff69b8b2895ap-55'), _hex(b'0x1.46fffe80475ep+0'), _hex(b'-0x1.5c0b19bc2f254p-54'),
        _hex(b'0x1.48fffef6fc1e7p+0'), _hex(b'0x1.b4009f23a2a72p-54'), _hex(b'0x1.4afffe5bea704p+0'), _hex(b'-0x1.4ffb7bf0d7d45p-54'),
        _hex(b'0x1.4d000171027dep+0'), _hex(b'-0x1.9c06471dc6a3dp-54'), _hex(b'0x1.4f0000ff03ee2p+0'), _hex(b'0x1.77f890b85531cp-54'),
        _hex(b'0x1.5100012dc4bd1p+0'), _hex(b'0x1.004657166a436p-57'), _hex(b'0x1.530001605277ap+0'), _hex(b'-0x1.6bfcece233209p-54'),
        _hex(b'0x1.54fffecdb704cp+0'), _hex(b'-0x1.902720505a1d7p-55'), _hex(b'0x1.56fffef5f54a9p+0'), _hex(b'0x1.bbfe60ec96412p-54'),
        _hex(b'0x1.5900017e61012p+0'), _hex(b'0x1.87ec581afef9p-55'), _hex(b'0x1.5b00003c93e92p+0'), _hex(b'-0x1.f41080abf0ccp-54'),
        _hex(b'0x1.5d0001d4919bcp+0'), _hex(b'-0x1.8812afb254729p-54'), _hex(b'0x1.5efffe7b87a89p+0'), _hex(b'-0x1.47eb780ed6904p-54'),
    )
    return tab2[2 * index], tab2[2 * index + 1]


@func()
def _logf_tab(index: usize) -> tuple[f64, f64]:
    """``__logf_data.tab[index]`` (musl's ``logf_data.c``) as
    ``(invc, logc)``."""
    const_comptime()
    tab = array(
        _hex(b'0x1.661ec79f8f3bep+0'), _hex(b'-0x1.57bf7808caadep-2'), _hex(b'0x1.571ed4aaf883dp+0'), _hex(b'-0x1.2bef0a7c06ddbp-2'),
        _hex(b'0x1.49539f0f010bp+0'), _hex(b'-0x1.01eae7f513a67p-2'), _hex(b'0x1.3c995b0b80385p+0'), _hex(b'-0x1.b31d8a68224e9p-3'),
        _hex(b'0x1.30d190c8864a5p+0'), _hex(b'-0x1.6574f0ac07758p-3'), _hex(b'0x1.25e227b0b8eap+0'), _hex(b'-0x1.1aa2bc79c81p-3'),
        _hex(b'0x1.1bb4a4a1a343fp+0'), _hex(b'-0x1.a4e76ce8c0e5ep-4'), _hex(b'0x1.12358f08ae5bap+0'), _hex(b'-0x1.1973c5a611cccp-4'),
        _hex(b'0x1.0953f419900a7p+0'), _hex(b'-0x1.252f438e10c1ep-5'), _hex(b'0x1p+0'), _hex(b'0x0p+0'),
        _hex(b'0x1.e608cfd9a47acp-1'), _hex(b'0x1.aa5aa5df25984p-5'), _hex(b'0x1.ca4b31f026aap-1'), _hex(b'0x1.c5e53aa362eb4p-4'),
        _hex(b'0x1.b2036576afce6p-1'), _hex(b'0x1.526e57720db08p-3'), _hex(b'0x1.9c2d163a1aa2dp-1'), _hex(b'0x1.bc2860d22477p-3'),
        _hex(b'0x1.886e6037841edp-1'), _hex(b'0x1.1058bc8a07ee1p-2'), _hex(b'0x1.767dcf5534862p-1'), _hex(b'0x1.4043057b6ee09p-2'),
    )
    return tab[2 * index], tab[2 * index + 1]


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


@func()
def log[T: Numeric](x: T) -> T:
    """The natural logarithm of ``x`` (``f32`` or ``f64``), the pure spy
    implementation: a port of musl's ``logf``/``log``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        xf = coerce(x, f32)
        ix = bitcast(xf, u32)
        if ix == 0x3F800000:
            return coerce(0.0, typ)  # log(1) is +0
        if (ix - 0x00800000) >= (0x7F800000 - 0x00800000):
            # x < 0x1p-126 or x is an inf/NaN
            if ix * 2 == 0:
                return coerce(_NEG_INF, typ)  # x is +-0
            if ix == 0x7F800000:
                return coerce(xf, typ)  # x is +inf
            if (ix & 0x80000000) != 0 or ix * 2 >= 0xFF000000:
                return coerce(_invalidf(xf), typ)  # x < 0 or NaN
            # normalize a subnormal x
            ix = bitcast(xf * coerce(_P23F, f32), u32)
            ix = ix - (23 << 23)
        tmp = ix - _LOG_OFF32
        i = coerce((tmp >> 19) % 16, usize)
        k = bitcast(tmp, i32) >> 23
        iz = ix - (tmp & 0xFF800000)
        invc, logc = _logf_tab(i)
        z = coerce(bitcast(iz, f32), f64)
        # log(x) = log1p(z/c - 1) + log(c) + k*ln2
        r = z * invc - 1.0
        y0 = logc + int_to_float_unchecked(k, f64) * _LOGF_LN2
        r2 = r * r
        y = _LOGF_A1 * r + _LOGF_A2
        y = _LOGF_A0 * r2 + y
        y = y * r2 + (y0 + r)
        return coerce(y, typ)
    elif typ == f64:
        xd = coerce(x, f64)
        ix = bitcast(xd, u64)
        top = coerce(ix >> 48, u32)
        if ix - _LOG_LO < _LOG_NEAR:
            # inputs close to 1.0
            if ix == 0x3FF0000000000000:
                return coerce(0.0, typ)  # log(1) is +0
            r = xd - 1.0
            r2 = r * r
            r3 = r * r2
            y = r3 * (
                _B1 + r * _B2 + r2 * _B3
                + r3 * (_B4 + r * _B5 + r2 * _B6 + r3 * (_B7 + r * _B8 + r2 * _B9 + r3 * _B10))
            )
            w = r * _P27
            rhi = r + w - w
            rlo = r - rhi
            w = rhi * rhi * _B0
            hi = r + w
            lo = r - hi + w
            lo = lo + _B0 * rlo * (rhi + r)
            y = y + lo
            y = y + hi
            return coerce(y, typ)
        if (top - 0x0010) >= (0x7FF0 - 0x0010):
            # x < 0x1p-1022 or x is an inf/NaN
            if ix * 2 == 0:
                return coerce(_NEG_INF, typ)  # x is +-0
            if ix == 0x7FF0000000000000:
                return coerce(xd, typ)  # x is +inf
            if (top & 0x8000) != 0 or (top & 0x7FF0) == 0x7FF0:
                return coerce(_invalid(xd), typ)  # x < 0 or NaN
            # normalize a subnormal x
            ix = bitcast(xd * _P52, u64)
            ix = ix - (52 << 52)
        tmp = ix - _LOG_OFF
        i = coerce((tmp >> 45) % 128, usize)
        k = bitcast(tmp, i64) >> 52
        iz = ix - (tmp & 0xFFF0000000000000)
        invc, logc = _log_tab(i)
        chi, clo = _log_tab2(i)
        z = bitcast(iz, f64)
        # log(x) = log1p(z/c - 1) + log(c) + k*ln2
        r = (z - chi - clo) * invc
        kd = int_to_float_unchecked(k, f64)
        w = kd * _LN2HI + logc
        hi = w + r
        lo = w - hi + r + kd * _LN2LO
        r2 = r * r
        y = lo + r2 * _A0 + r * r2 * (_A1 + r * _A2 + r2 * (_A3 + r * _A4)) + hi
        return coerce(y, typ)
    compile_error("only f32 or f64")
