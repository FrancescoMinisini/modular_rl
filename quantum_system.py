import numpy as np
import qutip as qt

class QuantumSystem:
    """
    Simulates a two-qubit superconducting system (gmon) using RWA Hamiltonian.
    Follows Eq. (1) from Niu et al. (2018).
    """
    def __init__(self, n_levels=3, dt=1.0):
        """
        Args:
            n_levels (int): Number of levels per qubit (default 3 for leakage calculation).
            dt (float): Time step in ns.
        """
        self.dims = n_levels
        self.dt = dt
        
        # Parameters (freq in GHz = 1000 MHz since time is ns? No, usually MHz and us. 
        # Paper says: eta = -200 MHz. Energy gap Delta = 200 MHz.
        # If we use MHz for frequency, time should be in microseconds (us).
        # OR: w in rad/ns, then f in GHz.
        # Let's check common conventions. 
        # If eta = -200e6 Hz. t = 1e-9 s.
        # w*t = 2pi*f*t. 
        # Easier: Scale everything to angular freq in units of 1/ns.
        # 1 MHz = 1e6 Hz = 1e-3 GHz = 1e-3 (1/ns).
        # omega = 2 * pi * f.
        # So 200 MHz = 0.2 GHz. 2pi * 0.2 rad/ns.
        
        # Let's stick to using MHz and microseconds? No, typical gate time 20-50 ns.
        # Let's use internal units: Energies in (2*pi*GHz) i.e. rad/ns.
        # 1 MHz = 0.001 GHz.
        # 200 MHz -> f = 0.2 GHz. w = 2*pi*0.2 = 0.4*pi rad/ns.
        # This keeps numbers around unity ~ 1.25.
        
        self.two_pi = 2 * np.pi
        self.eta_val = -0.200 * self.two_pi # -200 MHz converted to rad/ns
        
        # Operators
        self.a1 = qt.tensor(qt.destroy(n_levels), qt.qeye(n_levels))
        self.a2 = qt.tensor(qt.qeye(n_levels), qt.destroy(n_levels))
        
        self.n1 = self.a1.dag() * self.a1
        self.n2 = self.a2.dag() * self.a2
        
        # Drift Hamiltonian terms (Anharmonicity)
        # term: (eta/2) * sum( nj(nj-1) )
        self.H_drift = (self.eta_val / 2.0) * (
            self.n1 * (self.n1 - 1) + self.n2 * (self.n2 - 1)
        )
        
        # Coupling operators (g term)
        self.H_g = self.a2.dag() * self.a1 + self.a1.dag() * self.a2
        
        # Drive operators
        # term: i f_j ( a_j e^-iphi - a_j^dag e^iphi )
        # We can decompose this:
        # i f (a cos - ia sin - (adag cos + i adag sin))
        # = i f ( (a - adag) cos - i(a + adag) sin )
        # = f ( i(a - adag) cos + (a + adag) sin )
        # = f ( y_op cos + x_op sin )
        # where x_op = a + adag, y_op = i(a - adag) -- Wait, y = -i(a-adag).
        # Let's just construct it numerically at each step to avoid expansion errors.
        
        self.eye = qt.tensor(qt.qeye(n_levels), qt.qeye(n_levels))

    def get_hamiltonian(self, controls):
        """
        Constructs the Hamiltonian for a given set of control values.
        
        Args:
            controls (np.array): Shape (7,). 
                [delta1, delta2, f1, phi1, f2, phi2, g_coupling]
                Units: delta, f in MHz (converted to rad/ns inside), 
                       g in MHz (converted), phi in radians.
        
        Returns:
            qt.Qobj: Hamiltonian at this instant.
        """
        # Unpack and convert units to rad/ns
        d1, d2, f1, phi1, f2, phi2, g_val = controls
        
        d1 = d1 * 1e-3 * self.two_pi # MHz -> rad/ns
        d2 = d2 * 1e-3 * self.two_pi
        f1 = f1 * 1e-3 * self.two_pi
        f2 = f2 * 1e-3 * self.two_pi
        g_val = g_val * 1e-3 * self.two_pi
        
        # H_detuning = sum( delta_j * n_j )
        H_det = d1 * self.n1 + d2 * self.n2
        
        # H_coupling = g(t) * (a2+a1 + a1+a2)
        H_c = g_val * self.H_g
        
        # H_drive = sum( i f_j (a_j e^-iphi - a_j^dag e^iphi) )
        if f1 != 0:
            term1 = 1j * f1 * (self.a1 * np.exp(-1j * phi1) - self.a1.dag() * np.exp(1j * phi1))
        else:
            term1 = 0
            
        if f2 != 0:
            term2 = 1j * f2 * (self.a2 * np.exp(-1j * phi2) - self.a2.dag() * np.exp(1j * phi2))
        else:
            term2 = 0
            
        H = self.H_drift + H_det + H_c + term1 + term2
        return H

    def evolve_step(self, U, controls):
        """
        Evolves unitary U by time dt under constant controls.
        U_new = exp(-i * H * dt) * U
        """
        H = self.get_hamiltonian(controls)
        # Propagator for this step
        U_step = (-1j * H * self.dt).expm()
        return U_step * U

    def get_initial_unitary(self):
        """Returns Identity unitary for the full system."""
        return self.eye

    @staticmethod
    def target_gate_cz():
        """Returns Target Unitary for CZ gate in computational subspace."""
        # CZ = diag(1, 1, 1, -1)
        # But we need to represent it in the full Hilbert space or project U down.
        # Full dims = 3x3 = 9. Comp subspace = 2x2 = 4.
        # Indices: 00->0, 01->1, 10->3, 11->4 (if standard ordering).
        # Wrapper will handle subspace projection.
        # Let's define the 4x4 target matrix.
        return np.diag([1, 1, 1, -1])

    def get_subspace_unitary(self, U):
        """
        Projects full Unitary (dims x dims) to computational subspace (2x2).
        Computational basis indices:
        0: |00>, 1: |01>, 
        n_levels: |10> (index = n_levels*1 + 0 = 3 for dim=3)
        n_levels+1: |11> (index = 4 for dim=3)
        """
        dim = self.dims
        # Indices for 0 and 1 states of each qubit
        basis_indices = [
            0 * dim + 0, # 00
            0 * dim + 1, # 01
            1 * dim + 0, # 10
            1 * dim + 1  # 11
        ]
        
        # Extract 4x4 submatrix
        # U is a Qobj, U.full() is numpy array
        U_np = U.full()
        U_sub = U_np[np.ix_(basis_indices, basis_indices)]
        return U_sub

    def fidelity(self, U, U_target):
        """
        Computes gate fidelity F = (1/16) * |Tr(U_sub^dag * U_target)|^2
        """
        U_sub = self.get_subspace_unitary(U)
        # Fidelity formula for unitary gates
        # Arg U_target should be 4x4 numpy array
        tr = np.trace(U_sub.conj().T @ U_target)
        f = (1.0 / 16.0) * np.abs(tr)**2
        return f

    def leakage_cost(self, H_od_norm, H_od_2nd_deriv_norm, T_total):
        """
        Approximates L_tot from Eq (3).
        L_tot = ||H_od(0)||/Delta + ||H_od(T)||/Delta + integral(...)
        Assuming we just calculate the integral part (Runge-Kutta or sum).
        
        For RL step-by-step, we might compute instantaneous cost.
        But leakage bound is an integral over the whole path.
        """
        # Placeholder for complex TSWT cost.
        # We will simply return the instantaneous off-diagonal norm as proxy
        # for "leakage potential" at this step, if we treat it as a dense reward.
        # Or accumulate it.
        return H_od_norm # Simplified

