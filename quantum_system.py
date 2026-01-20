import numpy as np
import qutip as qt

class QuantumSystem:
    """
    Simulates a two-qubit superconducting system (gmon) using RWA Hamiltonian.
    Follows Niu et al. (2018).
    
    Units:
    - Time: ns
    - Frequencies/Controls: MHz -> converted to rad/ns internally.
      1 MHz = 1e-3 GHz = 1e-3 * 2pi rad/ns.
    """
    def __init__(self, n_levels=3, dt=1.0):
        self.dims = n_levels
        self.dt = dt
        self.two_pi = 2 * np.pi
        
        # Base Parameters (Eta = -200 MHz fixed anharmonicity)
        self.eta_base = -200.0 
        
        # Initial Parameters (can be modified by noise per step)
        self.reset_parameters()
        
        # Operators for 3-level system
        self.a1 = qt.tensor(qt.destroy(n_levels), qt.qeye(n_levels))
        self.a2 = qt.tensor(qt.qeye(n_levels), qt.destroy(n_levels))
        
        self.n1 = self.a1.dag() * self.a1
        self.n2 = self.a2.dag() * self.a2
        
        self.eye = qt.tensor(qt.qeye(n_levels), qt.qeye(n_levels))
        
        # Coupling operator component (a1 a2+ + a1+ a2)
        self.op_coupling = self.a2.dag() * self.a1 + self.a1.dag() * self.a2

        # Computational Subspace Indices (00, 01, 10, 11)
        self.comp_indices = [
            0 * n_levels + 0, 
            0 * n_levels + 1, 
            1 * n_levels + 0,
            1 * n_levels + 1
        ]
        
        # Leakage Subspace Indices (all others)
        self.all_indices = list(range(n_levels**2))
        self.leak_indices = [i for i in self.all_indices if i not in self.comp_indices]

    def reset_parameters(self):
        """Resets parameters to base values (no noise)."""
        self.eta_val = self.eta_base 
        # Noise accumulators (reset at start of episode usually)
        self.eta_noise = 0.0
        
    def set_parameters(self, eta_val):
        """Sets current eta value (MHz)."""
        self.eta_val = eta_val

    def get_hamiltonian(self, controls):
        """
        Constructs the Hamiltonian.
        Args:
            controls (np.array): [delta1, delta2, f1, phi1, f2, phi2, g]
            Units: MHz for delta, f, g; Rad for phi.
        Returns:
            qt.Qobj: Hamiltonian in rad/ns
        """
        # Unpack
        d1, d2, f1, phi1, f2, phi2, g = controls
        
        # Convert MHz to rad/ns
        scale = 1e-3 * self.two_pi
        
        Eta = self.eta_val * scale
        D1 = d1 * scale
        D2 = d2 * scale
        F1 = f1 * scale
        F2 = f2 * scale
        G = g * scale
        
        # H_drift (Anharmonicity)
        H_drift = (Eta / 2.0) * (self.n1 * (self.n1 - 1) + self.n2 * (self.n2 - 1))
        
        # H_detuning
        H_det = D1 * self.n1 + D2 * self.n2
        
        # H_coupling
        H_c = G * self.op_coupling
        
        # H_drive (Microwave)
        # Term: i f (a e^-iphi - a^dag e^iphi)
        # This matches i f(t) [ a exp(-i phi) - a^dag exp(i phi) ] from user req? 
        # Or usually Drive = f(t) * (a + a^dag)?
        # Paper (Eq 1): H_drive_j = f_j(t) [ exp(-i phi_j) a_j + exp(i phi_j) a_j^dag ] ? No that's usually rotating frame.
        # User specified: i f_j(t)( a_j e^(-i phi_j) - a_j^dag e^(i phi_j) )
        # Let's stick to user request strictly.
        
        H_d1 = 0
        if F1 != 0:
            exp_phi1 = np.exp(1j * phi1)
            # i * F1 * (a1 * conj(exp) - a1dag * exp)
            term = self.a1 * np.conj(exp_phi1) - self.a1.dag() * exp_phi1
            H_d1 = 1j * F1 * term
            
        H_d2 = 0
        if F2 != 0:
            exp_phi2 = np.exp(1j * phi2)
            term = self.a2 * np.conj(exp_phi2) - self.a2.dag() * exp_phi2
            H_d2 = 1j * F2 * term
            
        H = H_drift + H_det + H_c + H_d1 + H_d2
        return H

    def get_h_off_diag_norm(self, controls):
        """
        Calculates ||H_OD|| where H_OD contains terms coupling Comp <-> Leakage.
        Used for TSWT Bound.
        """
        H = self.get_hamiltonian(controls)
        H_np = H.full()
        
        # Extract off-diagonal blocks
        # Block 1: Rows in Comp, Cols in Leak
        block_cl = H_np[np.ix_(self.comp_indices, self.leak_indices)]
        
        # Block 2: Rows in Leak, Cols in Comp (Hermitian conjugate usually)
        block_lc = H_np[np.ix_(self.leak_indices, self.comp_indices)]
        
        # Frobenius Norm of these blocks
        # Norm^2 = sum(|elements|^2)
        norm_sq = np.sum(np.abs(block_cl)**2) + np.sum(np.abs(block_lc)**2)
        return np.sqrt(norm_sq)

    def evolve_step(self, U, controls):
        H = self.get_hamiltonian(controls)
        U_step = (-1j * H * self.dt).expm()
        return U_step * U

    def get_initial_unitary(self):
        return self.eye

    def target_gate_niu(self, alpha, gamma=np.pi/2):
        """
        Returns Target Unitary for N(alpha, alpha, gamma).
        Eq: exp( i (alpha XX + alpha YY + gamma ZZ) )
        in the Computational Subspace (4x4).
        """
        # Pauli Matrices
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Tensor Products in Comp Basis (2 qubits)
        XX = np.kron(sx, sx)
        YY = np.kron(sy, sy)
        ZZ = np.kron(sz, sz)
        
        # Exponent
        H_target = alpha * XX + alpha * YY + gamma * ZZ
        U_target = scipy.linalg.expm(1j * H_target)
        
        return U_target

    def get_subspace_unitary(self, U):
        U_np = U.full()
        U_sub = U_np[np.ix_(self.comp_indices, self.comp_indices)]
        return U_sub

    def fidelity(self, U, U_target):
        U_sub = self.get_subspace_unitary(U)
        # Global phase invariant fidelity
        # F = (1/d^2) |Tr(U_dag U_targ)|^2
        # d=4
        tr = np.trace(U_sub.conj().T @ U_target)
        f = (1.0 / 16.0) * np.abs(tr)**2
        return f

# Import scipy here because qutip might not expose expm for numpy arrays easily in all versions
import scipy.linalg
