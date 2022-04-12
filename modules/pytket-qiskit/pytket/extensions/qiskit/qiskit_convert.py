# Copyright 2019-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Methods to allow conversion between Qiskit and pytket circuit classes
"""
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Any,
    Iterable,
    cast,
    Set,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)
from inspect import signature
from uuid import UUID

import numpy as np

import sympy  # type: ignore
import qiskit.circuit.library.standard_gates as qiskit_gates  # type: ignore
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit import (
    Barrier,
    Instruction,
    InstructionSet,
    Gate,
    ControlledGate,
    Measure,
    Parameter,
    ParameterExpression,
    Reset,
)
from qiskit.circuit.library import CRYGate, RYGate, MCMT, PauliEvolutionGate  # type: ignore

from qiskit.extensions.unitary import UnitaryGate  # type: ignore
from pytket.circuit import (  # type: ignore
    CircBox,
    Circuit,
    Node,
    Op,
    OpType,
    Unitary2qBox,
    UnitType,
    CustomGateDef,
    Bit,
    Qubit,
)
from pytket._tket.circuit import _TEMP_BIT_NAME  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.architecture import Architecture, FullyConnected  # type: ignore
from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit

if TYPE_CHECKING:
    from qiskit.providers.backend import BackendV1 as QiskitBackend  # type: ignore
    from qiskit.providers.models.backendproperties import (  # type: ignore
        BackendProperties,
        Nduv,
    )
    from qiskit.circuit.quantumcircuitdata import QuantumCircuitData  # type: ignore
    from pytket.circuit import Op, UnitID  # type: ignore

_qiskit_gates_1q = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.HGate: OpType.H,
    qiskit_gates.IGate: OpType.noop,
    qiskit_gates.PhaseGate: OpType.U1,
    qiskit_gates.RGate: OpType.PhasedX,
    qiskit_gates.RXGate: OpType.Rx,
    qiskit_gates.RYGate: OpType.Ry,
    qiskit_gates.RZGate: OpType.Rz,
    qiskit_gates.SdgGate: OpType.Sdg,
    qiskit_gates.SGate: OpType.S,
    qiskit_gates.SXdgGate: OpType.SXdg,
    qiskit_gates.SXGate: OpType.SX,
    qiskit_gates.TdgGate: OpType.Tdg,
    qiskit_gates.TGate: OpType.T,
    qiskit_gates.U1Gate: OpType.U1,
    qiskit_gates.U2Gate: OpType.U2,
    qiskit_gates.U3Gate: OpType.U3,
    qiskit_gates.UGate: OpType.U3,
    qiskit_gates.XGate: OpType.X,
    qiskit_gates.YGate: OpType.Y,
    qiskit_gates.ZGate: OpType.Z,
}

_qiskit_gates_2q = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.CHGate: OpType.CH,
    qiskit_gates.CPhaseGate: OpType.CU1,
    qiskit_gates.CRZGate: OpType.CRz,
    qiskit_gates.CUGate: OpType.CU3,
    qiskit_gates.CU1Gate: OpType.CU1,
    qiskit_gates.CU3Gate: OpType.CU3,
    qiskit_gates.CXGate: OpType.CX,
    qiskit_gates.CYGate: OpType.CY,
    qiskit_gates.CZGate: OpType.CZ,
    qiskit_gates.iSwapGate: OpType.ISWAPMax,
    qiskit_gates.RXXGate: OpType.XXPhase,
    qiskit_gates.RYYGate: OpType.YYPhase,
    qiskit_gates.RZZGate: OpType.ZZPhase,
    qiskit_gates.SwapGate: OpType.SWAP,
}

_qiskit_gates_other = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.C3XGate: OpType.CnX,
    qiskit_gates.C4XGate: OpType.CnX,
    qiskit_gates.CCXGate: OpType.CCX,
    qiskit_gates.CSwapGate: OpType.CSWAP,
    # Multi-controlled gates (qiskit expects a list of controls followed by the target):
    qiskit_gates.MCXGate: OpType.CnX,
    qiskit_gates.MCXGrayCode: OpType.CnX,
    qiskit_gates.MCXRecursive: OpType.CnX,
    qiskit_gates.MCXVChain: OpType.CnX,
    # Note: should be OpType.CRy, but not currently available
    qiskit_gates.CRYGate: OpType.CnRy,
    # Special types:
    Barrier: OpType.Barrier,
    Instruction: OpType.CircBox,
    Gate: OpType.CustomGate,
    Measure: OpType.Measure,
    Reset: OpType.Reset,
    UnitaryGate: OpType.Unitary2qBox,
}

_known_qiskit_gate = {**_qiskit_gates_1q, **_qiskit_gates_2q, **_qiskit_gates_other}

# Some qiskit gates are aliases (e.g. UGate and U3Gate).
# In such cases this reversal will select one or the other.
_known_qiskit_gate_rev = {v: k for k, v in _known_qiskit_gate.items()}

# One way mapping: CRY is a special case of CnRy, but not vice versa.
# Constructing a qiskit CnRy gate is not just a single step,
# so treat as a special case.
del _known_qiskit_gate_rev[OpType.CnRy]

# Ensure U3 maps to UGate. (U3Gate deprecated in Qiskit but equivalent.)
_known_qiskit_gate_rev[OpType.U3] = qiskit_gates.UGate

# There is a bijective mapping, but requires some special parameter conversions
# tk1(a, b, c) = U(b, a-1/2, c+1/2) + phase(-(a+c)/2)
_known_qiskit_gate_rev[OpType.TK1] = qiskit_gates.UGate

# some gates are only equal up to global phase, support their conversion
# from tket -> qiskit
_known_gate_rev_phase = {
    optype: (qgate, 0.0) for optype, qgate in _known_qiskit_gate_rev.items()
}

_known_gate_rev_phase[OpType.V] = (qiskit_gates.SXGate, -0.25)
_known_gate_rev_phase[OpType.Vdg] = (qiskit_gates.SXdgGate, 0.25)

# use minor signature hacks to figure out the string names of qiskit Gate objects
_gate_str_2_optype: Dict[str, OpType] = dict()
for gate, optype in _known_qiskit_gate.items():
    if gate in (
        UnitaryGate,
        Instruction,
        Gate,
        qiskit_gates.MCXGate,  # all of these have special (c*n)x names
        qiskit_gates.MCXGrayCode,
        qiskit_gates.MCXRecursive,
        qiskit_gates.MCXVChain,
    ):
        continue
    sig = signature(gate.__init__)
    # name is only a property of the instance, not the class
    # so initialize with the correct number of dummy variables
    n_params = len([p for p in sig.parameters.values() if p.default is p.empty]) - 1
    name = gate(*([1] * n_params)).name
    _gate_str_2_optype[name] = optype

# assumption that unitary will refer to unitary2qbox, even though it means any unitary
_gate_str_2_optype["unitary"] = OpType.Unitary2qBox
_gate_str_2_optype_rev = {v: k for k, v in _gate_str_2_optype.items()}
# the aliasing of the name is ok in the reverse map
_gate_str_2_optype_rev[OpType.Unitary1qBox] = "unitary"


def _tk_gate_set(backend: "QiskitBackend") -> Set[OpType]:
    """Set of tket gate types supported by the qiskit backend"""
    config = backend.configuration()
    if config.simulator:
        return {
            _gate_str_2_optype[gate_str]
            for gate_str in config.basis_gates
            if gate_str in _gate_str_2_optype
        }.union({OpType.Measure, OpType.Reset, OpType.Barrier})
    else:
        return {
            _gate_str_2_optype[gate_str]
            for gate_str in config.supported_instructions
            if gate_str in _gate_str_2_optype
        }


def _qpo_from_peg(peg: PauliEvolutionGate, qubits: List[Qubit]) -> QubitPauliOperator:
    op = peg.operator
    qpodict = {}
    for p, c in zip(op.paulis, op.coeffs):
        if np.iscomplex(c):
            raise ValueError("Coefficient for Pauli {} is non-real.".format(p))
        qpslist = []
        pstr = p.to_label()
        for a in pstr:
            if a == "X":
                qpslist.append(Pauli.X)
            elif a == "Y":
                qpslist.append(Pauli.Y)
            elif a == "Z":
                qpslist.append(Pauli.Z)
            else:
                assert a == "I"
                qpslist.append(Pauli.I)
        qpodict[QubitPauliString(qubits, qpslist)] = c
    return QubitPauliOperator(qpodict)


class CircuitBuilder:
    def __init__(
        self,
        qregs: List[QuantumRegister],
        cregs: Optional[List[ClassicalRegister]] = None,
        name: Optional[str] = None,
        phase: Optional[sympy.Expr] = 0 * sympy.pi,
    ):
        self.qregs = qregs
        self.cregs = [] if cregs is None else cregs
        self.qbmap = {}
        self.cbmap = {}
        if name is not None:
            self.tkc = Circuit(name=name)
        else:
            self.tkc = Circuit()
        self.tkc.add_phase(phase)
        for reg in qregs:
            tk_reg = self.tkc.add_q_register(reg.name, len(reg))
            for i, qb in enumerate(reg):
                self.qbmap[qb] = Qubit(reg.name, i)
        self.cregmap = {}
        for reg in self.cregs:
            tk_reg = self.tkc.add_c_register(reg.name, len(reg))
            self.cregmap.update({reg: tk_reg})
            for i, cb in enumerate(reg):
                self.cbmap[cb] = Bit(reg.name, i)

    def circuit(self) -> Circuit:
        return self.tkc

    def add_qiskit_data(self, data: "QuantumCircuitData") -> None:
        for i, qargs, cargs in data:
            condition_kwargs = {}
            if i.condition is not None:
                cond_reg = self.cregmap[i.condition[0]]
                condition_kwargs = {
                    "condition_bits": [cond_reg[k] for k in range(len(cond_reg))],
                    "condition_value": i.condition[1],
                }
            optype = None
            if type(i) == ControlledGate:
                if type(i.base_gate) == qiskit_gates.RYGate:
                    optype = OpType.CnRy
                else:
                    # Maybe handle multicontrolled gates in a more general way,
                    # but for now just do CnRy
                    raise NotImplementedError(
                        "qiskit ControlledGate with "
                        + "base gate {} not implemented".format(i.base_gate)
                    )
            elif type(i) == PauliEvolutionGate:
                pass  # Special handling below
            else:
                optype = _known_qiskit_gate[type(i)]
            qubits = [self.qbmap[qbit] for qbit in qargs]
            bits = [self.cbmap[bit] for bit in cargs]

            if optype == OpType.Unitary2qBox:
                u = i.to_matrix()
                ubox = Unitary2qBox(u)
                # Note reversal of qubits, to account for endianness (pytket unitaries
                # are ILO-BE == DLO-LE; qiskit unitaries are ILO-LE == DLO-BE).
                self.tkc.add_unitary2qbox(
                    ubox, qubits[1], qubits[0], **condition_kwargs
                )
            elif type(i) == PauliEvolutionGate:
                qpo = _qpo_from_peg(i, qubits)
                empty_circ = Circuit(len(qargs))
                circ = gen_term_sequence_circuit(qpo, empty_circ)
                ccbox = CircBox(circ)
                self.tkc.add_circbox(ccbox, qubits)
            elif optype == OpType.Barrier:
                self.tkc.add_barrier(qubits)
            elif optype in (OpType.CircBox, OpType.CustomGate):
                qregs = [QuantumRegister(i.num_qubits, "q")] if i.num_qubits > 0 else []
                cregs = (
                    [ClassicalRegister(i.num_clbits, "c")] if i.num_clbits > 0 else []
                )
                builder = CircuitBuilder(qregs, cregs)
                builder.add_qiskit_data(i.definition)
                subc = builder.circuit()
                if optype == OpType.CircBox:
                    cbox = CircBox(subc)
                    self.tkc.add_circbox(cbox, qubits + bits, **condition_kwargs)
                else:
                    # warning, this will catch all `Gate` instances
                    # that were not picked up as a subclass in _known_qiskit_gate
                    params = [param_to_tk(p) for p in i.params]
                    gate_def = CustomGateDef.define(
                        i.name, subc, list(subc.free_symbols())
                    )
                    self.tkc.add_custom_gate(gate_def, params, qubits + bits)
            elif optype == OpType.CU3 and type(i) == qiskit_gates.CUGate:
                if i.params[-1] == 0:
                    self.tkc.add_gate(
                        optype,
                        [param_to_tk(p) for p in i.params[:-1]],
                        qubits,
                        **condition_kwargs,
                    )
                else:
                    raise NotImplementedError("CUGate with nonzero phase")
            else:
                params = [param_to_tk(p) for p in i.params]
                self.tkc.add_gate(optype, params, qubits + bits, **condition_kwargs)


def qiskit_to_tk(qcirc: QuantumCircuit, preserve_param_uuid: bool = False) -> Circuit:
    """Convert a :py:class:`qiskit.QuantumCircuit` to a :py:class:`Circuit`.

    :param qcirc: A circuit to be converted
    :type qcirc: QuantumCircuit
    :param preserve_param_uuid: Whether to preserve symbolic Parameter uuids
        by appending them to the tket Circuit symbol names as "_UUID:<uuid>".
        This can be useful if you want to reassign Parameters after conversion
        to tket and back, as it is necessary for Parameter object equality
        to be preserved.
    :type preserve_param_uuid: bool
    :return: The converted circuit
    :rtype: Circuit
    """
    circ_name = qcirc.name
    # Parameter uses a hidden _uuid for equality check
    # we optionally preserve this in parameter name for later use
    if preserve_param_uuid:
        updates = {p: Parameter(f"{p.name}_UUID:{p._uuid}") for p in qcirc.parameters}
        qcirc = cast(QuantumCircuit, qcirc.assign_parameters(updates))

    builder = CircuitBuilder(
        qregs=qcirc.qregs,
        cregs=qcirc.cregs,
        name=circ_name,
        phase=param_to_tk(qcirc.global_phase),
    )
    builder.add_qiskit_data(qcirc.data)
    return builder.circuit()


def param_to_tk(p: Union[float, ParameterExpression]) -> sympy.Expr:
    if isinstance(p, ParameterExpression):
        symexpr = p._symbol_expr
        try:
            return symexpr._sympy_() / sympy.pi  # type: ignore
        except AttributeError:
            return symexpr / sympy.pi  # type: ignore
    else:
        return p / sympy.pi  # type: ignore


def param_to_qiskit(
    p: sympy.Expr, symb_map: Dict[Parameter, sympy.Symbol]
) -> Union[float, ParameterExpression]:
    ppi = p * sympy.pi
    if len(ppi.free_symbols) == 0:
        return float(ppi.evalf())
    else:
        return ParameterExpression(symb_map, ppi)


def _get_params(
    op: Op, symb_map: Dict[Parameter, sympy.Symbol]
) -> List[Union[float, ParameterExpression]]:
    return [param_to_qiskit(p, symb_map) for p in op.params]


def append_tk_command_to_qiskit(
    op: "Op",
    args: List["UnitID"],
    qcirc: QuantumCircuit,
    qregmap: Dict[str, QuantumRegister],
    cregmap: Dict[str, ClassicalRegister],
    symb_map: Dict[Parameter, sympy.Symbol],
    range_preds: Dict[Bit, Tuple[List["UnitID"], int]],
) -> InstructionSet:
    optype = op.type
    if optype == OpType.Measure:
        qubit = args[0]
        bit = args[1]
        qb = qregmap[qubit.reg_name][qubit.index[0]]
        b = cregmap[bit.reg_name][bit.index[0]]
        return qcirc.measure(qb, b)

    if optype == OpType.Reset:
        qb = qregmap[args[0].reg_name][args[0].index[0]]
        return qcirc.reset(qb)

    if optype in [OpType.CircBox, OpType.ExpBox, OpType.PauliExpBox, OpType.CustomGate]:
        subcircuit = op.get_circuit()
        subqc = tk_to_qiskit(subcircuit)
        qargs = []
        cargs = []
        for a in args:
            if a.type == UnitType.qubit:
                qargs.append(qregmap[a.reg_name][a.index[0]])
            else:
                cargs.append(cregmap[a.reg_name][a.index[0]])
        if optype == OpType.CustomGate:
            instruc = subqc.to_gate()
            instruc.name = op.get_name()
        else:
            instruc = subqc.to_instruction()
        return qcirc.append(instruc, qargs, cargs)
    if optype == OpType.Unitary2qBox:
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        u = op.get_matrix()
        g = UnitaryGate(u, label="unitary")
        # Note reversal of qubits, to account for endianness (pytket unitaries are
        # ILO-BE == DLO-LE; qiskit unitaries are ILO-LE == DLO-BE).
        return qcirc.append(g, qargs=list(reversed(qargs)))
    if optype == OpType.Barrier:
        if any(q.type == UnitType.bit for q in args):
            raise NotImplementedError(
                "Qiskit Barriers are not defined for classical bits."
            )
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        g = Barrier(len(args))
        return qcirc.append(g, qargs=qargs)
    if optype == OpType.RangePredicate:
        if op.lower != op.upper:
            raise NotImplementedError
        range_preds[args[-1]] = (args[:-1], op.lower)
        # attach predicate to bit,
        # subsequent conditional will handle it
        return Instruction("", 0, 0, [])
    if optype == OpType.Conditional:
        if args[0] in range_preds:
            assert op.value == 1
            condition_bits, value = range_preds[args[0]]
            del range_preds[args[0]]
            args = condition_bits + args[1:]
            width = len(condition_bits)
        else:
            width = op.width
            value = op.value
        regname = args[0].reg_name
        if len(cregmap[regname]) != width:
            raise NotImplementedError("OpenQASM conditions must be an entire register")
        for i, a in enumerate(args[:width]):
            if a.reg_name != regname:
                raise NotImplementedError(
                    "OpenQASM conditions can only use a single register"
                )
            if a.index != [i]:
                raise NotImplementedError(
                    "OpenQASM conditions must be an entire register in order"
                )
        instruction = append_tk_command_to_qiskit(
            op.op, args[width:], qcirc, qregmap, cregmap, symb_map, range_preds
        )

        instruction.c_if(cregmap[regname], value)
        return instruction
    # normal gates
    qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
    if optype == OpType.CnX:
        return qcirc.mcx(qargs[:-1], qargs[-1])

    # special case
    if optype == OpType.CnRy:
        # might as well do a bit more checking
        assert len(op.params) == 1
        alpha = param_to_qiskit(op.params[0], symb_map)
        assert len(qargs) >= 2
        if len(qargs) == 2:
            # presumably more efficient; single control only
            new_gate = CRYGate(alpha)
        else:
            new_ry_gate = RYGate(alpha)
            new_gate = MCMT(
                gate=new_ry_gate, num_ctrl_qubits=len(qargs) - 1, num_target_qubits=1
            )
        qcirc.append(new_gate, qargs)
        return qcirc

    if optype == OpType.CU3:
        params = _get_params(op, symb_map) + [0]
        return qcirc.append(qiskit_gates.CUGate(*params), qargs=qargs)

    if optype == OpType.TK1:
        params = _get_params(op, symb_map)
        half = ParameterExpression(symb_map, sympy.pi / 2)
        qcirc.global_phase += -params[0] / 2 - params[2] / 2
        return qcirc.append(
            qiskit_gates.UGate(params[1], params[0] - half, params[2] + half),
            qargs=qargs,
        )
    # others are direct translations
    try:
        gatetype, phase = _known_gate_rev_phase[optype]
    except KeyError as error:
        raise NotImplementedError(
            "Cannot convert tket Op to Qiskit gate: " + op.get_name()
        ) from error
    params = _get_params(op, symb_map)
    g = gatetype(*params)
    qcirc.global_phase += phase * sympy.pi
    return qcirc.append(g, qargs=qargs)


def tk_to_qiskit(tkcirc: Circuit) -> QuantumCircuit:
    """Convert back

    :param tkcirc: A circuit to be converted
    :type tkcirc: Circuit
    :return: The converted circuit
    :rtype: QuantumCircuit
    """
    tkc = tkcirc
    qcirc = QuantumCircuit(name=tkc.name)
    qreg_sizes: Dict[str, int] = {}
    for qb in tkc.qubits:
        if len(qb.index) != 1:
            raise NotImplementedError("Qiskit registers must use a single index")
        if (qb.reg_name not in qreg_sizes) or (qb.index[0] >= qreg_sizes[qb.reg_name]):
            qreg_sizes.update({qb.reg_name: qb.index[0] + 1})
    creg_sizes: Dict[str, int] = {}
    for b in tkc.bits:
        if len(b.index) != 1:
            raise NotImplementedError("Qiskit registers must use a single index")
        # names with underscore not supported, and _TEMP_BIT_NAME should not be needed
        # for qiskit compatible classical control circuits
        if b.reg_name != _TEMP_BIT_NAME and (
            (b.reg_name not in creg_sizes) or (b.index[0] >= creg_sizes[b.reg_name])
        ):
            creg_sizes.update({b.reg_name: b.index[0] + 1})
    qregmap = {}
    for reg_name, size in qreg_sizes.items():
        qis_reg = QuantumRegister(size, reg_name)
        qregmap.update({reg_name: qis_reg})
        qcirc.add_register(qis_reg)
    cregmap = {}
    for reg_name, size in creg_sizes.items():
        qis_reg = ClassicalRegister(size, reg_name)
        cregmap.update({reg_name: qis_reg})
        qcirc.add_register(qis_reg)
    symb_map = {Parameter(str(s)): s for s in tkc.free_symbols()}
    range_preds: Dict[Bit, Tuple[List["UnitID"], int]] = dict()
    for command in tkc:
        append_tk_command_to_qiskit(
            command.op, command.args, qcirc, qregmap, cregmap, symb_map, range_preds
        )
    qcirc.global_phase += param_to_qiskit(tkc.phase, symb_map)

    # if UUID stored in name, set parameter uuids accordingly (see qiskit_to_tk)
    updates = dict()
    for p in qcirc.parameters:
        name_spl = p.name.split("_UUID:", 2)
        if len(name_spl) == 2:
            p_name, uuid = name_spl
            new_p = Parameter.__new__(Parameter, p_name, UUID(uuid))
            new_p.__init__(p_name)
            updates[p] = new_p
    qcirc.assign_parameters(updates, inplace=True)
    return qcirc


def process_characterisation(backend: "QiskitBackend") -> Dict[str, Any]:
    """Convert a :py:class:`qiskit.providers.backend.Backendv1` to a dictionary
     containing device Characteristics

    :param backend: A backend to be converted
    :type backend: Backendv1
    :return: A dictionary containing device characteristics
    :rtype: dict
    """

    # TODO explicitly check for and separate 1 and 2 qubit gates
    properties = cast("BackendProperties", backend.properties())

    def return_value_if_found(iterator: Iterable["Nduv"], name: str) -> Optional[Any]:
        try:
            first_found = next(filter(lambda item: item.name == name, iterator))
        except StopIteration:
            return None
        if hasattr(first_found, "value"):
            return first_found.value
        return None

    config = backend.configuration()
    coupling_map = config.coupling_map
    n_qubits = config.n_qubits
    if coupling_map is None:
        # Assume full connectivity
        arc = FullyConnected(n_qubits)
    else:
        arc = Architecture(coupling_map)

    link_errors: dict = defaultdict(dict)
    node_errors: dict = defaultdict(dict)
    readout_errors: dict = {}

    t1_times = []
    t2_times = []
    frequencies = []
    gate_times = []

    if properties is not None:
        for index, qubit_info in enumerate(properties.qubits):
            t1_times.append([index, return_value_if_found(qubit_info, "T1")])
            t2_times.append([index, return_value_if_found(qubit_info, "T2")])
            frequencies.append([index, return_value_if_found(qubit_info, "frequency")])
            # readout error as a symmetric 2x2 matrix
            offdiag = return_value_if_found(qubit_info, "readout_error")
            if offdiag:
                diag = 1.0 - offdiag
                readout_errors[index] = [[diag, offdiag], [offdiag, diag]]
            else:
                readout_errors[index] = None

        for gate in properties.gates:
            name = gate.gate
            if name in _gate_str_2_optype:
                optype = _gate_str_2_optype[name]
                qubits = gate.qubits
                gate_error = return_value_if_found(gate.parameters, "gate_error")
                gate_error = gate_error if gate_error else 0.0
                gate_length = return_value_if_found(gate.parameters, "gate_length")
                gate_length = gate_length if gate_length else 0.0
                gate_times.append([name, qubits, gate_length])
                # add gate fidelities to their relevant lists
                if len(qubits) == 1:
                    node_errors[qubits[0]].update({optype: gate_error})
                elif len(qubits) == 2:
                    link_errors[tuple(qubits)].update({optype: gate_error})
                    opposite_link = tuple(qubits[::-1])
                    if opposite_link not in coupling_map:
                        # to simulate a worse reverse direction square the fidelity
                        link_errors[opposite_link].update({optype: 2 * gate_error})

    # map type (k1 -> k2) -> v[k1] -> v[k2]
    K1 = TypeVar("K1")
    K2 = TypeVar("K2")
    V = TypeVar("V")
    convert_keys_t = Callable[[Callable[[K1], K2], Dict[K1, V]], Dict[K2, V]]
    # convert qubits to architecture Nodes
    convert_keys: convert_keys_t = lambda f, d: {f(k): v for k, v in d.items()}
    node_errors = convert_keys(lambda q: Node(q), node_errors)
    link_errors = convert_keys(lambda p: (Node(p[0]), Node(p[1])), link_errors)
    readout_errors = convert_keys(lambda q: Node(q), readout_errors)

    characterisation: Dict[str, Any] = dict()
    characterisation["NodeErrors"] = node_errors
    characterisation["EdgeErrors"] = link_errors
    characterisation["ReadoutErrors"] = readout_errors
    characterisation["Architecture"] = arc
    characterisation["t1times"] = t1_times
    characterisation["t2times"] = t2_times
    characterisation["Frequencies"] = frequencies
    characterisation["GateTimes"] = gate_times

    return characterisation


def get_avg_characterisation(
    characterisation: Dict[str, Any]
) -> Dict[str, Dict[Node, float]]:
    """
    Convert gate-specific characterisation into readout, one- and two-qubit errors

    Used to convert a typical output from `process_characterisation` into an input
    noise characterisation for NoiseAwarePlacement
    """

    K = TypeVar("K")
    V1 = TypeVar("V1")
    V2 = TypeVar("V2")
    map_values_t = Callable[[Callable[[V1], V2], Dict[K, V1]], Dict[K, V2]]
    map_values: map_values_t = lambda f, d: {k: f(v) for k, v in d.items()}

    node_errors = cast(Dict[Node, Dict[OpType, float]], characterisation["NodeErrors"])
    link_errors = cast(
        Dict[Tuple[Node, Node], Dict[OpType, float]], characterisation["EdgeErrors"]
    )
    readout_errors = cast(
        Dict[Node, List[List[float]]], characterisation["ReadoutErrors"]
    )

    avg: Callable[[Dict[Any, float]], float] = lambda xs: sum(xs.values()) / len(xs)
    avg_mat: Callable[[List[List[float]]], float] = (
        lambda xs: (xs[0][1] + xs[1][0]) / 2.0
    )
    avg_readout_errors = map_values(avg_mat, readout_errors)
    avg_node_errors = map_values(avg, node_errors)
    avg_link_errors = map_values(avg, link_errors)

    return {
        "node_errors": avg_node_errors,
        "edge_errors": avg_link_errors,
        "readout_errors": avg_readout_errors,
    }
