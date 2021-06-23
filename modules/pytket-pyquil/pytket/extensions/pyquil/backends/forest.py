# Copyright 2019-2021 Cambridge Quantum Computing
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

import json
from copy import copy
from typing import cast, Any, Dict, Iterable, List, Optional, Sequence, Union
from uuid import uuid4
from logging import warning

import numpy as np
from pyquil import get_qc
from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.gates import I
from pyquil.paulis import ID, PauliSum, PauliTerm
from pyquil.quilatom import Qubit as Qubit_

from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.pyquil._metadata import __extension_version__
from pytket.passes import (  # type: ignore
    BasePass,
    EulerAngleReduction,
    CXMappingPass,
    RebaseQuil,
    SequencePass,
    SynthesiseIBM,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    FlattenRegisters,
    SimplifyInitial,
)
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.predicates import (  # type: ignore
    NoSymbolsPredicate,
    ConnectivityPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    DefaultRegisterPredicate,
    Predicate,
)
from pytket.extensions.pyquil.pyquil_convert import (
    process_characterisation,
    get_avg_characterisation,
    tk_to_pyquil,
)
from pytket.routing import NoiseAwarePlacement, Architecture  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray

_STATUS_MAP = {
    "done": StatusEnum.COMPLETED,
    "running": StatusEnum.RUNNING,
    "loaded": StatusEnum.SUBMITTED,
    "connected": StatusEnum.SUBMITTED,
}


def _default_q_index(q: Qubit) -> int:
    if q.reg_name != "q" or len(q.index) != 1:
        raise ValueError("Non-default qubit register")
    return int(q.index[0])


class ForestBackend(Backend):
    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True
    _GATE_SET = {OpType.CZ, OpType.Rx, OpType.Rz, OpType.Measure, OpType.Barrier}

    def __init__(self, qc_name: str, simulator: bool = True):
        """Backend for running circuits on a Rigetti QCS device or simulating with the
        QVM.

        :param qc_name: The name of the particular QuantumComputer to use. See the
            pyQuil docs for more details.
        :type qc_name: str
        :param simulator: Simulate the device with the QVM (True), or run on the QCS
            (False). Defaults to True.
        :type simulator: bool, optional
        """
        super().__init__()
        self._qc: QuantumComputer = get_qc(qc_name, as_qvm=simulator)

        char_dict: dict = process_characterisation(self._qc)
        arch = char_dict.get("Architecture", Architecture([]))
        node_errors = char_dict.get("NodeErrors")
        link_errors = char_dict.get("EdgeErrors")
        self._backend_info = BackendInfo(
            type(self).__name__,
            qc_name,
            __extension_version__,
            arch,
            self._GATE_SET,
            misc={
                "characterisation": {
                    "NodeErrors": node_errors,
                    "EdgeErrors": link_errors,
                }
            },
        )

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            GateSetPredicate(self.backend_info.gate_set),
            ConnectivityPredicate(self.backend_info.architecture),
        ]

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [
            DecomposeBoxes(),
            FlattenRegisters(),
        ]
        if optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        passlist.append(
            CXMappingPass(
                self.backend_info.architecture,
                NoiseAwarePlacement(
                    self.backend_info.architecture,
                    **get_avg_characterisation(self.characterisation),
                ),
                directed_cx=False,
                delay_measures=True,
            )
        )
        if optimisation_level == 2:
            passlist.append(CliffordSimp(False))
        if optimisation_level > 0:
            passlist.append(SynthesiseIBM())
        passlist.append(RebaseQuil())
        if optimisation_level > 0:
            passlist.extend(
                [
                    EulerAngleReduction(OpType.Rx, OpType.Rz),
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )
        return SequencePass(passlist)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (int, str)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`.
        """
        circuits = list(circuits)
        n_shots_list: List[int] = []
        if hasattr(n_shots, "__iter__"):
            for n in cast(Sequence[Optional[int]], n_shots):
                if n is None or n < 1:
                    raise ValueError(
                        "n_shots values are required for all circuits for this backend"
                    )
                n_shots_list.append(n)
            if len(n_shots_list) != len(circuits):
                raise ValueError("The length of n_shots and circuits must match")
        else:
            if n_shots is None:
                raise ValueError("Parameter n_shots is required for this backend")
            # convert n_shots to a list
            n_shots_list = [cast(int, n_shots)] * len(circuits)

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = kwargs.get("postprocess", False)

        handle_list = []
        for circuit, n_shots in zip(circuits, n_shots_list):
            if postprocess:
                c0, ppcirc = prepare_circuit(circuit, allow_classical=False)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = circuit, None
            p, bits = tk_to_pyquil(c0, return_used_bits=True)
            p.wrap_in_numshots_loop(n_shots)
            ex = self._qc.compiler.native_quil_to_executable(p)
            qam = copy(self._qc.qam)
            qam.load(ex)
            qam.random_seed = kwargs.get("seed")  # type: ignore
            qam.run()
            handle = ResultHandle(uuid4().int, json.dumps(ppcirc_rep))
            measures = circuit.n_gates_of_type(OpType.Measure)
            if measures == 0:
                self._cache[handle] = {
                    "qam": qam,
                    "c_bits": sorted(bits),
                    "result": self.empty_result(circuit, n_shots=n_shots),
                }
            else:
                self._cache[handle] = {"qam": qam, "c_bits": sorted(bits)}
            handle_list.append(handle)
        return handle_list

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache and "result" in self._cache[handle]:
            return CircuitStatus(StatusEnum.COMPLETED)
        if handle in self._cache:
            qamstatus = self._cache[handle]["qam"].status
            tkstat = _STATUS_MAP.get(qamstatus, StatusEnum.ERROR)
            return CircuitStatus(tkstat, qamstatus)
        raise CircuitNotRunError(handle)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: none.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            if handle not in self._cache:
                raise CircuitNotRunError(handle)

            qam = self._cache[handle]["qam"]
            shots = qam.wait().read_memory(region_name="ro")
            shots = OutcomeArray.from_readouts(shots)
            ppcirc_rep = json.loads(cast(str, handle[1]))
            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
            res = BackendResult(
                shots=shots, c_bits=self._cache[handle]["c_bits"], ppcirc=ppcirc
            )
            self._cache[handle].update({"result": res})
            return res

    @property
    def characterisation(self) -> Dict[str, Any]:
        char = self._backend_info.get_misc("characterisation")
        return cast(Dict[str, Any], char)

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info


class ForestStateBackend(Backend):
    _supports_state = True
    _supports_expectation = True
    _expectation_allows_nonhermitian = False
    _persistent_handles = False
    _GATE_SET = {
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.H,
        OpType.S,
        OpType.T,
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
        OpType.CZ,
        OpType.CX,
        OpType.CCX,
        OpType.CU1,
        OpType.U1,
        OpType.SWAP,
    }

    def __init__(self) -> None:
        """Backend for running simulations on the Rigetti QVM Wavefunction Simulator."""
        super().__init__()
        self._sim = WavefunctionSimulator()

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._GATE_SET),
            DefaultRegisterPredicate(),
        ]

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes(), FlattenRegisters()]
        if optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        passlist.append(RebaseQuil())
        if optimisation_level > 0:
            passlist.append(EulerAngleReduction(OpType.Rx, OpType.Rz))
        return SequencePass(passlist)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (int,)

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        handle_list = []
        if valid_check:
            self._check_all_circuits(circuits)
        for circuit in circuits:
            p = tk_to_pyquil(circuit)
            for qb in circuit.qubits:
                # Qubits with no gates will not be included in the Program
                # Add identities to ensure all qubits are present and dimension
                # is as expected
                p += I(Qubit_(qb.index[0]))
            handle = ResultHandle(uuid4().int)
            state = np.array(self._sim.wavefunction(p).amplitudes)
            try:
                phase = float(circuit.phase)
                coeff = np.exp(phase * np.pi * 1j)
                state *= coeff
            except ValueError:
                warning(
                    "Global phase is dependent on a symbolic parameter, so cannot "
                    "adjust for phase"
                )
            implicit_perm = circuit.implicit_qubit_permutation()
            res_qubits = [
                implicit_perm[qb] for qb in sorted(circuit.qubits, reverse=True)
            ]
            res = BackendResult(q_bits=res_qubits, state=state)
            self._cache[handle] = {"result": res}
            handle_list.append(handle)
        return handle_list

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def _gen_PauliTerm(self, term: QubitPauliString, coeff: complex = 1.0) -> PauliTerm:
        pauli_term = ID() * coeff
        for q, p in term.to_dict().items():
            pauli_term *= PauliTerm(p.name, _default_q_index(q))
        return pauli_term  # type: ignore

    def get_pauli_expectation_value(
        self, state_circuit: Circuit, pauli: QubitPauliString
    ) -> complex:
        """Calculates the expectation value of the given circuit using the built-in QVM
        functionality

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param pauli: Pauli operator
        :type pauli: QubitPauliString
        :return: :math:`\\left<\\psi | P | \\psi \\right>`
        :rtype: complex
        """
        prog = tk_to_pyquil(state_circuit)
        pauli_term = self._gen_PauliTerm(pauli)
        return complex(self._sim.expectation(prog, [pauli_term]))

    def get_operator_expectation_value(
        self, state_circuit: Circuit, operator: QubitPauliOperator
    ) -> complex:
        """Calculates the expectation value of the given circuit with respect to the
        operator using the built-in QVM functionality

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param operator: Operator :math:`H`.
        :type operator: QubitPauliOperator
        :return: :math:`\\left<\\psi | H | \\psi \\right>`
        :rtype: complex
        """
        prog = tk_to_pyquil(state_circuit)
        pauli_sum = PauliSum(
            [self._gen_PauliTerm(term, coeff) for term, coeff in operator._dict.items()]
        )
        return complex(self._sim.expectation(prog, pauli_sum))


_xcirc = Circuit(1).Rx(1, 0)
_xcirc.add_phase(0.5)
