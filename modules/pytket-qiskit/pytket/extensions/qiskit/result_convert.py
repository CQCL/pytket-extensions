from typing import (
    List,
    Iterator,
    Sequence,
    Type,
    Tuple,
    TYPE_CHECKING,
    Optional,
    Dict,
    Any,
)
from collections import Counter

import numpy as np  # type: ignore

from qiskit.result import Result  # type: ignore
from qiskit.result.models import ExperimentResultData  # type: ignore

from pytket.circuit import Bit, Qubit, UnitID  # type: ignore

from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray

if TYPE_CHECKING:
    from qiskit.qobj import QobjExperimentHeader  # type: ignore


def _gen_uids(labels: Sequence[Tuple[str, int]], derived: Type[UnitID]) -> List[UnitID]:
    # sorted_labels = sorted(labels, key=lambda x: x[0])
    # see
    # https://github.com/Qiskit/qiskit-terra/blob/6148588d25a5a0e96744b541bb0da676779f3204/qiskit/result/postprocess.py#L36
    return [
        derived(name, index)
        for name, size in reversed(labels)
        for index in reversed(range(size))  # reversed to account for little-endian
    ]


def _hex_to_outar(hexes: Sequence[str], width: int) -> OutcomeArray:
    ints = [int(hexst, 16) for hexst in hexes]
    return OutcomeArray.from_ints(ints, width)


def qiskit_result_to_backendresult(res: Result) -> Iterator[BackendResult]:
    for result in res.results:
        header = result.header
        width = header.memory_slots

        c_bits = (
            _gen_uids(header.creg_sizes, Bit) if hasattr(header, "creg_sizes") else None
        )
        q_bits = (
            _gen_uids(header.qreg_sizes, Qubit)
            if hasattr(header, "qreg_sizes")
            else None
        )
        shots, counts, state, unitary = (None,) * 4
        datadict = result.data.to_dict()
        if len(datadict) == 0 and result.shots > 0:
            n_bits = len(c_bits) if c_bits else 0
            shots = OutcomeArray.from_readouts(
                np.zeros((result.shots, n_bits), dtype=np.uint8)  #  type: ignore
            )
        else:
            if "memory" in datadict:
                memory = datadict["memory"]
                shots = _hex_to_outar(memory, width)
            elif "counts" in datadict:
                qis_counts = datadict["counts"]
                counts = Counter(
                    dict(
                        (_hex_to_outar([hexst], width), count)
                        for hexst, count in qis_counts.items()
                    )
                )

            if "statevector" in datadict:
                state = datadict["statevector"]

            if "unitary" in datadict:
                unitary = datadict["unitary"]

        yield BackendResult(
            c_bits=c_bits,
            q_bits=q_bits,
            shots=shots,
            counts=counts,
            state=state,
            unitary=unitary,
        )


def backendresult_to_qiskit_resultdata(
    res: BackendResult,
    header: "QobjExperimentHeader",
    final_map: Optional[Dict[UnitID, UnitID]],
) -> ExperimentResultData:
    data: Dict[str, Any] = dict()
    if res.contains_state_results:
        qbits = (
            _gen_uids(header.qreg_sizes, Qubit) if hasattr(header, "qreg_sizes") else []
        )
        if final_map:
            qbits = [final_map[q] for q in qbits]
        stored_res = res.get_result(qbits)
        if stored_res.state is not None:
            data["statevector"] = stored_res.state
        if stored_res.unitary is not None:
            data["unitary"] = stored_res.unitary
    if res.contains_measured_results:
        cbits = (
            _gen_uids(header.creg_sizes, Bit) if hasattr(header, "creg_sizes") else []
        )
        if final_map:
            cbits = [final_map[c] for c in cbits]
        stored_res = res.get_result(cbits)
        if stored_res.shots is not None:
            data["memory"] = [hex(i) for i in stored_res.shots.to_intlist()]
            data["counts"] = dict(Counter(data["memory"]))
        elif stored_res.counts is not None:
            data["counts"] = {
                hex(i.to_intlist()[0]): f for i, f in stored_res.counts.items()
            }
    return ExperimentResultData(**data)
