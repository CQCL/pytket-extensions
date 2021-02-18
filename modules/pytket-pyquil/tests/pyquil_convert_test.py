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

import time
from shutil import which
import platform

import docker  # type: ignore
import numpy as np  # type: ignore
import pytest
from pyquil import Program
from pyquil.api import WavefunctionSimulator
from pyquil.gates import (
    X,
    Y,
    Z,
    H,
    S,
    T,
    RX,
    RY,
    RZ,
    CZ,
    CNOT,
    CCNOT,
    CPHASE,
    SWAP,
    MEASURE,
)
from pyquil.quilbase import Measurement
from sympy import pi, Symbol  # type: ignore

from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.pyquil import pyquil_to_tk, tk_to_pyquil
from pytket.extensions.backends.forest import ForestStateBackend
from pytket.passes import RemoveRedundancies  # type: ignore

skip_qvm_tests = (which("docker") is None) or (platform.system() == "Windows")


@pytest.fixture(scope="module")
def qvm(request) -> None:  # type: ignore
    print("running qvm container")
    dock = docker.from_env()
    container = dock.containers.run(
        image="rigetti/qvm", command="-S", detach=True, ports={5000: 5000}, remove=True
    )
    time.sleep(1)  # Wait for container to start running the server.
    request.addfinalizer(container.stop)
    return None


def get_test_program(measure: bool = False) -> Program:
    PI = float(pi.evalf())
    p = Program()
    p += X(0)
    p += Y(1)
    p += Z(2)
    p += H(3)
    p += S(0)
    p += T(1)
    p += RX(PI / 2, 2)
    p += RY(PI / 2, 3)
    p += RZ(PI / 2, 0)
    p += CZ(0, 1)
    p += CNOT(2, 3)
    p += CCNOT(0, 1, 2)
    p += CPHASE(PI / 4, 2, 1)
    p += SWAP(0, 3)
    if measure:
        ro = p.declare("ro", "BIT", 4)
        p += MEASURE(0, ro[0])
        p += MEASURE(3, ro[1])
        p += MEASURE(2, ro[2])
        p += MEASURE(1, ro[3])
    return p


def adjust_for_relative_phase(state0, state1) -> tuple:  # type: ignore
    maxval = 0
    phase0 = 1
    phase1 = 1
    for s0, s1 in zip(state0, state1):
        if abs(s0) > maxval:
            maxval = abs(s0)
            phase0 = s0 / abs(s0)
            phase1 = s1 / abs(s1)
    return (state0 / phase0), (state1 / phase1)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"  # type: ignore
)
def test_convert(qvm):
    wf_sim = WavefunctionSimulator()
    p = get_test_program()
    initial_wf = wf_sim.wavefunction(p)
    tkc = pyquil_to_tk(p)
    p2 = tk_to_pyquil(tkc)
    final_wf = wf_sim.wavefunction(p2)
    initial_phaseless, final_phaseless = adjust_for_relative_phase(
        initial_wf.amplitudes, final_wf.amplitudes
    )
    assert np.allclose(initial_phaseless, final_phaseless, atol=1e-10)


def test_from_tket() -> None:
    c = Circuit(4, 2)
    c.X(0)
    c.H(1)
    c.S(1)
    c.CX(2, 0)
    c.Ry(0.5, 3)
    c.Measure(3, 0)
    c.Measure(1, 1)
    p = tk_to_pyquil(c)
    assert (
        len(p.instructions) == 8
    )  # 5 gates, 2 measures, and an initial declaration of classical register


def test_measure() -> None:
    p = get_test_program(True)
    m_map = {}
    for i in p.instructions:
        if isinstance(i, Measurement):
            m_map[i.qubit] = i.classical_reg.offset  # type: ignore
    tkc = pyquil_to_tk(p)
    p2 = tk_to_pyquil(tkc)
    m_map2 = {}
    for i in p2.instructions:
        if isinstance(i, Measurement):
            m_map2[i.qubit] = i.classical_reg.offset  # type: ignore
    assert m_map == m_map2


def test_measures_are_last() -> None:
    c = Circuit(2, 2)
    c.Measure(0, 0).X(1).Y(1).Measure(1, 1)
    p = tk_to_pyquil(c)
    assert isinstance(p.instructions[3], Measurement)
    assert isinstance(p.instructions[4], Measurement)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"  # type: ignore
)
def test_symbolic(qvm) -> None:
    pi2 = Symbol("pi2")
    pi3 = Symbol("pi3")

    tkc = Circuit(2).Rx(pi2, 1).Rx(-pi3, 1).CX(1, 0)
    RemoveRedundancies().apply(tkc)

    prog = tk_to_pyquil(tkc)
    tkc2 = pyquil_to_tk(prog)

    assert tkc2.free_symbols() == {pi2, pi3}
    tkc2.symbol_substitution({pi2: pi / 2, pi3: -pi / 3})

    backend = ForestStateBackend()
    state1 = backend.get_state(tkc2)
    state0 = np.array([-0.56468689 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.82530523j])
    assert np.allclose(state0, state1)
