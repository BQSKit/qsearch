#!/usr/bin/env python
# coding: utf-8


import numpy as np
from search_compiler_native import *
from search_compiler.gatesets import QubitCNOTLinear
from search_compiler.circuits import *
from timeit import timeit


def native_fill_row(gate, n):
    return GateKronecker(np.repeat(gate, n))

def native_linear_topo(double_step, single_step, n, d, identity_step=None):
    if not identity_step:
        identity_step = GateIdentity(n, d)
    return [GateProduct(np.array([GateKronecker(np.array([*[identity_step]*i, double_step, *[identity_step]*(n-i-2)])),
                        GateKronecker(np.array([*[identity_step]*i, single_step, single_step, *[identity_step]*(n-i-2)]))])
            ) for i in range(0,n-1)]
qubits = 4

print("Native initial layer")
print(timeit(lambda: native_fill_row(GateSingleQubit(1), qubits), number=10000))
print("Native search layer")
print(timeit(lambda: native_linear_topo(GateCNOT(), GateSingleQubit(1), qubits, 1), number=10000))

qct = QubitCNOTLinear()
print("Python initial layer")
print(timeit(lambda: qct.initial_layer(qubits, 1), number=10000))
print("Python search layer")
print(timeit(lambda: qct.search_layers(qubits, 1), number=10000))

py = qct.search_layers(qubits, 1)
print("Python .matrix")
print(timeit(lambda: py[0].matrix(np.repeat(np.pi, 100)), number=10000))

native = native_linear_topo(GateCNOT(), GateSingleQubit(1), qubits, 1)
print("Native .matrix")
print(timeit(lambda: native[0].matrix(np.array([np.pi]*100)), number=10000))

