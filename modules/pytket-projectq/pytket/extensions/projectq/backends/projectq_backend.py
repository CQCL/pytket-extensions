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

"""Methods to allow tket circuits to be ran on ProjectQ simulator
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)
from uuid import uuid4
from logging import warning

import numpy as np
import projectq  # type: ignore
from projectq import MainEngine  # type: ignore
from projectq.backends import Simulator  # type: ignore
from projectq.cengines import ForwarderEngine  # type: ignore
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.circuit import Qubit  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    ResultHandle,
    CircuitStatus,
    StatusEnum,
)
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backendresult import BackendResult
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseTket,
    FullPeepholeOptimise,
    DecomposeBoxes,
    FlattenRegisters,
)
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.predicates import (  # type: ignore
    NoSymbolsPredicate,
    NoMidMeasurePredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    DefaultRegisterPredicate,
    Predicate,
)
from pytket.architecture import Architecture  # type: ignore
from pytket.extensions.projectq.projectq_convert import tk_to_projectq, _REBASE  # type: ignore
from pytket.extensions.projectq._metadata import __extension_version__  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import KwargTypes


def _default_q_index(q: Qubit) -> int:
    if q.reg_name != "q" or len(q.index) != 1:
        raise ValueError("Non-default qubit register")
    return int(q.index[0])


_GATE_SET = {
    OpType.SWAP,
    OpType.CRz,
    OpType.CX,
    OpType.CZ,
    OpType.H,
    OpType.X,
    OpType.Y,
    OpType.Z,
    OpType.S,
    OpType.T,
    OpType.V,
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.Barrier,
    OpType.Measure,
}


class ProjectQBackend(Backend):
    """Backend for running statevector simulations on the ProjectQ simulator."""

    _supports_state = True
    _supports_expectation = True
    _expectation_allows_nonhermitian = False
    _persistent_handles = False

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def characterisation(self) -> Dict[str, Any]:
        return dict()

    @property
    def backend_info(self) -> BackendInfo:
        backend_info = BackendInfo(
            type(self).__name__,
            None,
            __extension_version__,
            Architecture([]),
            _GATE_SET,
        )
        return backend_info

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoSymbolsPredicate(),
            NoMidMeasurePredicate(),
            GateSetPredicate(_GATE_SET),
            DefaultRegisterPredicate(),
        ]

    def rebase_pass(self) -> BasePass:
        return _REBASE

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [DecomposeBoxes(), FlattenRegisters(), self.rebase_pass()]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FlattenRegisters(),
                    SynthesiseTket(),
                    self.rebase_pass(),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FlattenRegisters(),
                    FullPeepholeOptimise(),
                    self.rebase_pass(),
                ]
            )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list = []
        for circuit, n_shots_circ in zip(circuits, n_shots_list):
            sim = Simulator(rnd_seed=kwargs.get("seed"))
            fwd = ForwarderEngine(sim)
            eng = MainEngine(backend=sim, engine_list=[fwd])
            qureg = eng.allocate_qureg(circuit.n_qubits)
            tk_to_projectq(eng, qureg, circuit, True)
            eng.flush()
            state = np.array(
                eng.backend.cheat()[1], dtype=complex
            )  # `cheat()` returns tuple:(a dictionary of qubit indices, statevector)
            handle = ResultHandle(str(uuid4()))
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
            # reverse qubits as projectq state is dlo
            res_qubits = [
                implicit_perm[qb] for qb in sorted(circuit.qubits, reverse=True)
            ]
            measures = circuit.n_gates_of_type(OpType.Measure)
            if measures == 0 and n_shots_circ is not None:
                backres = self.empty_result(circuit, n_shots=n_shots_circ)
            else:
                backres = BackendResult(q_bits=res_qubits, state=state)
            self._cache[handle] = {"result": backres}
            handle_list.append(handle)
        return handle_list

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def _expectation_value(
        self,
        circuit: Circuit,
        hamiltonian: projectq.ops.QubitOperator,  # type: ignore
        valid_check: bool = True,
    ) -> complex:
        if valid_check and not self.valid_circuit(circuit):
            raise ValueError(
                "Circuits do not satisfy all required predicates for this backend"
            )
        sim = Simulator()
        fwd = ForwarderEngine(sim)
        eng = MainEngine(backend=sim, engine_list=[fwd])
        qureg = eng.allocate_qureg(circuit.n_qubits)
        tk_to_projectq(eng, qureg, circuit)
        eng.flush()
        energy = eng.backend.get_expectation_value(hamiltonian, qureg)
        return complex(energy)

    def get_pauli_expectation_value(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit using the built-in
        ProjectQ functionality

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param pauli: Pauli operator
        :type pauli: QubitPauliString
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | P | \\psi \\right>`
        :rtype: complex
        """
        pauli_tuple = tuple((_default_q_index(q), p.name) for q, p in pauli.map.items())
        return self._expectation_value(
            state_circuit, projectq.ops.QubitOperator(pauli_tuple), valid_check  # type: ignore
        )

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit with respect to the
        operator using the built-in ProjectQ functionality

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param operator: Operator :math:`H`. Must be Hermitian.
        :type operator: QubitPauliOperator
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | H | \\psi \\right>`
        :rtype: complex
        """
        ham = projectq.ops.QubitOperator()  # type: ignore
        for term, coeff in operator._dict.items():
            if type(coeff) is complex and abs(coeff.imag) > 1e-12:
                raise ValueError(
                    "Operator is not Hermitian and cannot be converted to "
                    "`projectq.ops.QubitOperator`."
                )
            ham += projectq.ops.QubitOperator(  # type: ignore
                tuple((_default_q_index(q), p.name) for q, p in term.map.items()),
                float(coeff),
            )
        return self._expectation_value(state_circuit, ham, valid_check)
