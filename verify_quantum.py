import numpy as np
import qutip as qt
from quantum_system import QuantumSystem
from quantum_env import QuantumEnv

def test_initial_unitary():
    sys = QuantumSystem()
    U = sys.get_initial_unitary()
    eye = qt.tensor(qt.qeye(3), qt.qeye(3))
    assert U == eye, "Initial unitary is not identity"
    print("[PASS] Initial Unitary")

def test_fidelity():
    sys = QuantumSystem()
    # Test Identity fidelity
    eye = qt.tensor(qt.qeye(3), qt.qeye(3))
    # Target identity (4x4)
    target_eye = np.eye(4)
    
    # We need to compute fidelity between full Eye and partial Eye.
    # get_subspace_unitary of Eye(9) is Eye(4).
    # Fidelity(Eye, Eye) should be 1.
    
    # Mock system target gate
    # Create a 4x4 eye target
    f = sys.fidelity(eye, target_eye)
    assert np.isclose(f, 1.0), f"Fidelity of Identity failed: {f}"
    print("[PASS] Fidelity Identity")

def test_dynamics_no_controls():
    dt = 1.0
    sys = QuantumSystem(dt=dt)
    U = sys.get_initial_unitary()
    controls = np.zeros(7)
    
    # Evolve 1 step
    U_next = sys.evolve_step(U, controls)
    
    # With 0 controls, only drift Hamiltonian is active (Anharmonicity).
    # H_drift is diagonal. So U_next should be diagonal (phases).
    U_next_np = U_next.full()
    off_diag = U_next_np - np.diag(np.diag(U_next_np))
    assert np.allclose(off_diag, 0), "Drift evolution introduced off-diagonal terms!"
    print("[PASS] Dynamics No Controls")

def test_leakage_proxy():
    sys = QuantumSystem()
    controls = np.zeros(7)
    # No controls -> H is diagonal -> Leakage proxy should be 0
    l = sys.h_off_diag_norm(controls)
    assert np.isclose(l, 0), f"Leakage with 0 controls should be 0, got {l}"
    
    # Add controls that induce leakage?
    # g coupling couples 01 <-> 10 (allowed) and 12 <-> 21?
    # Our system is 3 levels.
    # a1 destroys level 1->0, 2->1.
    # adag1 creates 0->1, 1->2.
    # Coupling H_g = a1 a2dag + a1dag a2.
    # |11> -> (a1|1>)(a2dag|1>) + (a1dag|1>)(a2|1>)
    #      = |0>|2> + |2>|0>
    # |11> (level 4 if 00,01,02...)
    # 00,01,02,10,11,12,20,21,22
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8.
    # |11> is index 4.
    # |02> is index 2. (Non-comp)
    # |20> is index 6. (Non-comp)
    # So coupling term connects Comp (4) to Non-Comp (2, 6).
    # This should show up in h_off_diag_norm.
    
    controls[6] = 100.0 # g = 100 MHz
    l_g = sys.h_off_diag_norm(controls)
    assert l_g > 0, "Leakage proxy should be > 0 with coupling enabled"
    print(f"[PASS] Leakage Proxy (Value: {l_g:.4f})")

if __name__ == "__main__":
    test_initial_unitary()
    test_fidelity()
    test_dynamics_no_controls()
    test_leakage_proxy()
