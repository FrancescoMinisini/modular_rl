import numpy as np
import qutip as qt

class QuantumSystem:
    """
    Simulates a two-qubit superconducting system (gmon) using RWA Hamiltonian.
    Follows Eq. (1) from Niu et al. (2018).
    
    Units:
    - Time: ns
    - Frequency: rad/ns (internal)
    - Inputs: MHz (controls) -> converted to rad/ns
    """
    def __init__(self, n_levels=3, dt=1.0):
        """
        Args:
            n_levels (int): Number of levels per qubit (default 3 for leakage calculation).
            dt (float): Time step in ns.
        """
        self.dims = n_levels
        self.dt = dt
        self.two_pi = 2 * np.pi
        
        # Base Parameters
        # Anharmonicity eta = -200 MHz
        self.eta_base = -200.0 
        
        # Current Parameters (can be modified by noise)
        self.reset_parameters()
        
        # Operators
        self.a1 = qt.tensor(qt.destroy(n_levels), qt.qeye(n_levels))
        self.a2 = qt.tensor(qt.qeye(n_levels), qt.destroy(n_levels))
        
        self.n1 = self.a1.dag() * self.a1
        self.n2 = self.a2.dag() * self.a2
        
        self.eye = qt.tensor(qt.qeye(n_levels), qt.qeye(n_levels))
        
        # Coupling operator component
        self.op_coupling = self.a2.dag() * self.a1 + self.a1.dag() * self.a2

    def reset_parameters(self):
        """Resets parameters to base values."""
        self.eta = self.eta_base * 1e-3 * self.two_pi # rad/ns
        self.delta_offset = np.zeros(2) # Offsets for detuning noise
    
    def set_parameters(self, eta_val=None, delta_offsets=None):
        """
        Sets system parameters, useful for static noise injection.
        Args:
            eta_val (float): Anharmonicity in MHz.
            delta_offsets (list): [delta1_off, delta2_off] in MHz.
        """
        if eta_val is not None:
            self.eta = eta_val * 1e-3 * self.two_pi
        if delta_offsets is not None:
            self.delta_offset = np.array(delta_offsets) * 1e-3 * self.two_pi

    def get_hamiltonian(self, controls):
        """
        Constructs the Hamiltonian for a given set of control values.
        
        Args:
            controls (np.array): Shape (7,). 
                [delta1, delta2, f1, phi1, f2, phi2, g_coupling]
                Units: delta, f, g in MHz; phi in radians.
        
        Returns:
            qt.Qobj: Hamiltonian at this instant (in rad/ns).
        """
        # Unpack
        d1_in, d2_in, f1_in, phi1, f2_in, phi2, g_in = controls
        
        # Convert to rad/ns
        # Note: 1 MHz = 1e-3 GHz = 1e-3 * 2pi rad/ns
        scale = 1e-3 * self.two_pi
        
        d1 = d1_in * scale + self.delta_offset[0]
        d2 = d2_in * scale + self.delta_offset[1]
        f1 = f1_in * scale
        f2 = f2_in * scale
        g_val = g_in * scale
        
        # H_drift (Anharmonicity) = (eta/2) * sum( nj(nj-1) )
        H_drift = (self.eta / 2.0) * (
            self.n1 * (self.n1 - 1) + self.n2 * (self.n2 - 1)
        )
        
        # H_detuning = sum( delta_j * n_j )
        H_det = d1 * self.n1 + d2 * self.n2
        
        # H_coupling = g(t) * (a2+a1 + a1+a2)
        H_c = g_val * self.op_coupling
        
        # H_drive = sum( i f_j (a_j e^-iphi - a_j^dag e^iphi) )
        # term: i f (a e^-iphi - adag e^iphi)
        H_d1 = 0
        if f1 != 0:
            exp_phi1 = np.exp(1j * phi1)
            # i * f1 * (a1 * conj(exp_phi) - a1^dag * exp_phi)
            # Note: np.exp(-1j * phi1) is conj(exp_phi)
            term = self.a1 * np.conj(exp_phi1) - self.a1.dag() * exp_phi1
            H_d1 = 1j * f1 * term
            
        H_d2 = 0
        if f2 != 0:
            exp_phi2 = np.exp(1j * phi2)
            term = self.a2 * np.conj(exp_phi2) - self.a2.dag() * exp_phi2
            H_d2 = 1j * f2 * term
            
        H = H_drift + H_det + H_c + H_d1 + H_d2
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
        # CZ = diag(1, 1, 1, -1) in basis |00>, |01>, |10>, |11>
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

    def h_off_diag_norm(self, controls):
        """
        Calculates the norm of the off-diagonal terms of the Hamiltonian.
        This roughly corresponds to the leakage-inducing terms.
        
        NOTE: This is a simplified evaluation of leakage potential. 
        It will need to be changed with a higher order evaluation (TSWT) later, 
        as described in Niu et al. (2018).
        """
        H = self.get_hamiltonian(controls)
        
        # We want to identify terms that couple computational subspace to non-computational.
        # Ideally, we'd block-diagonalize or mask.
        # Simple proxy: Just take the norm of elements connecting levels (0,1) to (2).
        
        # Mask for computational subspace
        dim = self.dims
        comp_indices = [0, 1] 
        # For 2 qubits: 00, 01, 10, 11 are comp.
        # But wait, leakage is leaving the logical subspace.
        # Indices in full space:
        # 0:|00>, 1:|01>, 2:|02> (leak), 3:|10>, 4:|11>, 5:|12> (leak), 6:|20> (leak)...
        
        comp_basis = [0, 1, 3, 4]
        
        # Create a mask for off-diagonal (coupling comp to non-comp)
        H_np = H.full()
        
        loss_norm = 0.0
        
        # Iterate over computational rows
        for r in comp_basis:
            # Sum coupling to non-computational cols
            for c in range(dim*dim):
                if c not in comp_basis:
                    loss_norm += np.abs(H_np[r, c])**2
                    
        return np.sqrt(loss_norm)
