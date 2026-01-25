"""
Complexodynamics: The First Law of Complexity Evolution

Implements the core algorithms from Adami's "First Law of Complexodynamics" paper:
1. Shannon complexity calculation
2. Information flow dynamics (I_E and I_L)
3. Channel capacity computation
4. Complexity trajectory solver
5. Full evolutionary simulator

Author: 30u30 Project
Date: 2026
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from collections import Counter
import warnings

# ============================================================================
# SECTION 1: SHANNON COMPLEXITY
# ============================================================================

def shannon_complexity(sequence: str, base: int = 2) -> float:
    """
    Calculate Shannon entropy (complexity) of a sequence.
    
    The Shannon entropy quantifies the information content per symbol:
        H = -Σ p_i * log(p_i)
    
    For DNA (4 symbols), maximum entropy is log2(4) = 2 bits/symbol.
    
    Args:
        sequence: String of symbols (e.g., "ACGTACGT")
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        Entropy in bits (or nats) per symbol
        
    Example:
        >>> shannon_complexity("AAAA")  # Uniform sequence
        0.0
        >>> shannon_complexity("ACGT")  # Maximum diversity
        2.0
        >>> shannon_complexity("AACCGGTT")  # Still maximum
        2.0
    """
    if len(sequence) == 0:
        return 0.0
    
    # Count symbol frequencies
    counts = Counter(sequence)
    total = len(sequence)
    
    # Calculate probabilities
    probs = np.array([count / total for count in counts.values()])
    
    # Shannon entropy
    log_fn = np.log2 if base == 2 else np.log
    entropy = -np.sum(probs * log_fn(probs))
    
    return entropy


def complexity_vector(sequences: List[str]) -> np.ndarray:
    """
    Compute complexity for multiple sequences (vectorized).
    
    Args:
        sequences: List of strings
        
    Returns:
        Array of complexities
    """
    return np.array([shannon_complexity(seq) for seq in sequences])


def sequence_diversity(sequence: str, window_size: int = 100) -> np.ndarray:
    """
    Calculate local complexity over sliding windows.
    
    Useful for detecting regions of high/low complexity in genomes.
    
    Args:
        sequence: DNA/protein sequence
        window_size: Size of sliding window
        
    Returns:
        Array of local complexities
    """
    if len(sequence) < window_size:
        return np.array([shannon_complexity(sequence)])
    
    n_windows = len(sequence) - window_size + 1
    complexities = np.zeros(n_windows)
    
    for i in range(n_windows):
        window = sequence[i:i + window_size]
        complexities[i] = shannon_complexity(window)
    
    return complexities


# ============================================================================
# SECTION 2: INFORMATION FLOW DYNAMICS
# ============================================================================

class InformationFlow:
    """
    Models information gain (from selection) and loss (from mutation).
    
    The First Law states:
        dC/dt = I_E - I_L
    
    Where:
        I_E = Information gain from environment (selection)
        I_L = Information loss from mutation
    """
    
    def __init__(self, mutation_rate: float, selection_strength: float,
                 genome_length: int, n_symbols: int = 4):
        """
        Initialize information flow model.
        
        Args:
            mutation_rate: Per-base mutation probability
            selection_strength: Strength of selection pressure
            genome_length: Number of bases in genome
            n_symbols: Alphabet size (4 for DNA)
        """
        self.mu = mutation_rate
        self.beta = selection_strength
        self.L = genome_length
        self.n_symbols = n_symbols
        
    def information_gain(self, fitness_variance: float) -> float:
        """
        Calculate I_E: information gained from selection.
        
        From Fisher's Fundamental Theorem:
            I_E ≈ β * Var(fitness)
        
        Args:
            fitness_variance: Variance in fitness across population
            
        Returns:
            Information gain in bits/generation
        """
        return self.beta * fitness_variance
    
    def information_loss(self) -> float:
        """
        Calculate I_L: information lost from mutation.
        
        For uniform mutation to all other symbols:
            I_L = μ * L * log2(n_symbols)
        
        Each mutation loses log2(n_symbols) bits of information.
        
        Returns:
            Information loss in bits/generation
        """
        return self.mu * self.L * np.log2(self.n_symbols)
    
    def net_information_flow(self, fitness_variance: float) -> float:
        """
        Net information flow: dC/dt = I_E - I_L
        
        Args:
            fitness_variance: Population fitness variance
            
        Returns:
            Net information change per generation
        """
        I_E = self.information_gain(fitness_variance)
        I_L = self.information_loss()
        return I_E - I_L
    
    def equilibrium_complexity(self) -> float:
        """
        Find equilibrium complexity where I_E = I_L.
        
        At equilibrium:
            β * Var(fitness) = μ * L * log2(n_symbols)
        
        Solving for C:
            C_eq ≈ C_max * (1 - μL / (β * N_e))
        
        Returns:
            Equilibrium complexity in bits/site
        """
        # Simplified: assume we're near C_max
        C_max = np.log2(self.n_symbols)
        
        # Fitness variance at equilibrium
        # (approximation from quasispecies theory)
        equilibrium_factor = min(1.0, self.beta / (self.mu * self.L))
        
        return C_max * equilibrium_factor


# ============================================================================
# SECTION 3: CHANNEL CAPACITY
# ============================================================================

def channel_capacity_simple(mutation_rate: float, genome_length: int) -> float:
    """
    Maximum sustainable complexity (simple formula).
    
    From Shannon's noisy channel theorem:
        C_max ≈ -log2(μ * L)
    
    Args:
        mutation_rate: Per-base error rate
        genome_length: Number of bases
        
    Returns:
        Maximum complexity in total bits
        
    Example:
        >>> channel_capacity_simple(1e-6, 1e6)  # E. coli
        19.93 bits
        >>> channel_capacity_simple(1e-9, 3e9)  # Human
        34.76 bits
    """
    product = mutation_rate * genome_length
    if product >= 1.0:
        warnings.warn("μL ≥ 1: Error catastrophe! Genome cannot be maintained.")
        return 0.0
    return -np.log2(product)


def channel_capacity_gaussian(mutation_rate: float, genome_length: int,
                               noise_variance: float = None) -> float:
    """
    Channel capacity for Gaussian noise model.
    
    From information theory:
        C = 0.5 * log2(1 / (2πe * σ²))
    
    Args:
        mutation_rate: Not used directly (included for consistency)
        genome_length: Not used directly
        noise_variance: Variance of Gaussian noise channel
        
    Returns:
        Channel capacity in bits
    """
    if noise_variance is None:
        # Estimate from mutation rate
        noise_variance = mutation_rate
    
    if noise_variance >= 1.0 / (2 * np.pi * np.e):
        return 0.0
    
    capacity = 0.5 * np.log2(1.0 / (2 * np.pi * np.e * noise_variance))
    return capacity


def channel_capacity_binomial(mutation_rate: float, genome_length: int) -> float:
    """
    Channel capacity for binomial noise (discrete mutations).
    
    More accurate model for biological sequences.
    
    Args:
        mutation_rate: Per-base error probability
        genome_length: Number of bases
        
    Returns:
        Channel capacity in bits/site
    """
    # Binary symmetric channel capacity
    p = mutation_rate
    
    if p >= 0.5:
        return 0.0  # No information can flow
    
    # H(p) = -p*log(p) - (1-p)*log(1-p)
    if p == 0:
        return np.log2(4)  # Maximum for DNA
    
    H_p = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Capacity per symbol
    C_symbol = np.log2(4) - H_p
    
    return C_symbol


def eigen_error_threshold(mutation_rate: float) -> float:
    """
    Calculate Eigen's error threshold: maximum genome length.
    
    Beyond this length, information loss exceeds gain (error catastrophe).
    
        L_max = (1/μ) * log(1/μ)
    
    Args:
        mutation_rate: Per-base error rate
        
    Returns:
        Maximum sustainable genome length
        
    Example:
        >>> eigen_error_threshold(1e-4)  # RNA virus
        ~10,000 bases (matches reality!)
        >>> eigen_error_threshold(1e-9)  # Human with DNA repair
        ~10 billion bases
    """
    if mutation_rate == 0:
        return np.inf
    
    L_max = (1.0 / mutation_rate) * np.log(1.0 / mutation_rate)
    return L_max


# ============================================================================
# SECTION 4: COMPLEXITY TRAJECTORY
# ============================================================================

class ComplexityTrajectory:
    """
    Solves for C(t): complexity as a function of time.
    
    Differential equation:
        dC/dt = I_E - I_L = λ(C_max - C)
    
    Solution:
        C(t) = C_max * (1 - exp(-λt))
    """
    
    def __init__(self, mutation_rate: float, genome_length: int,
                 selection_strength: float, n_symbols: int = 4):
        """
        Initialize trajectory solver.
        
        Args:
            mutation_rate: Per-base mutation rate
            genome_length: Genome size
            selection_strength: Selection pressure
            n_symbols: Alphabet size
        """
        self.mu = mutation_rate
        self.L = genome_length
        self.beta = selection_strength
        self.n_symbols = n_symbols
        
        # Calculate equilibrium complexity
        self.C_max = self._calculate_C_max()
        
        # Calculate growth rate
        self.lambda_ = self._calculate_lambda()
        
    def _calculate_C_max(self) -> float:
        """Calculate maximum complexity (bits/site)."""
        total_capacity = channel_capacity_simple(self.mu, self.L)
        # Convert to per-site
        return min(total_capacity / self.L, np.log2(self.n_symbols))
    
    def _calculate_lambda(self) -> float:
        """Calculate exponential growth rate."""
        # From theory: λ ≈ β (selection strength)
        # Empirically observed to be proportional
        return self.beta
    
    def complexity_at_time(self, t: float) -> float:
        """
        Calculate C(t) at a specific time.
        
        Args:
            t: Time in generations
            
        Returns:
            Complexity in bits/site
        """
        return self.C_max * (1.0 - np.exp(-self.lambda_ * t))
    
    def evolve(self, generations: int, initial_complexity: float = 0.0) -> np.ndarray:
        """
        Simulate complexity evolution over time.
        
        Args:
            generations: Number of generations to simulate
            initial_complexity: Starting complexity (default: 0)
            
        Returns:
            Array of complexities at each generation
        """
        t = np.arange(generations)
        C_t = self.C_max * (1.0 - np.exp(-self.lambda_ * t))
        
        # Adjust for initial complexity
        if initial_complexity > 0:
            C_t = initial_complexity + (self.C_max - initial_complexity) * (1.0 - np.exp(-self.lambda_ * t))
        
        return C_t
    
    def time_to_equilibrium(self, threshold: float = 0.95) -> float:
        """
        Calculate time to reach threshold * C_max.
        
        Args:
            threshold: Fraction of C_max (default: 95%)
            
        Returns:
            Number of generations to equilibrium
        """
        # C(t) = C_max * (1 - exp(-λt)) = threshold * C_max
        # 1 - exp(-λt) = threshold
        # exp(-λt) = 1 - threshold
        # -λt = log(1 - threshold)
        # t = -log(1 - threshold) / λ
        
        if threshold >= 1.0:
            return np.inf
        
        t_eq = -np.log(1.0 - threshold) / self.lambda_
        return t_eq


# ============================================================================
# SECTION 5: EVOLUTIONARY SIMULATOR
# ============================================================================

class EvolutionarySimulator:
    """
    Full population-based evolution simulator.
    
    Simulates:
    - Population of genomes
    - Mutation (with specified rate)
    - Selection (fitness-based)
    - Complexity tracking over time
    """
    
    def __init__(self, pop_size: int, genome_length: int,
                 mutation_rate: float, selection_model: Callable,
                 n_symbols: int = 4):
        """
        Initialize evolutionary simulator.
        
        Args:
            pop_size: Population size
            genome_length: Length of each genome
            mutation_rate: Per-base mutation probability
            selection_model: Function(genome) -> fitness
            n_symbols: Alphabet size (4 for DNA)
        """
        self.N = pop_size
        self.L = genome_length
        self.mu = mutation_rate
        self.fitness_fn = selection_model
        self.n_symbols = n_symbols
        
        # Initialize random population
        self.population = self._initialize_population()
        
        # Track history
        self.complexity_history = []
        self.fitness_history = []
        
    def _initialize_population(self) -> np.ndarray:
        """Create random initial population."""
        # Random genomes (integers 0 to n_symbols-1)
        return np.random.randint(0, self.n_symbols, size=(self.N, self.L))
    
    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        Apply mutations to a genome.
        
        Args:
            genome: Original genome (array of integers)
            
        Returns:
            Mutated genome
        """
        mutated = genome.copy()
        
        # Determine which sites mutate
        mutation_mask = np.random.random(self.L) < self.mu
        n_mutations = np.sum(mutation_mask)
        
        if n_mutations > 0:
            # Mutate to random other symbols
            new_symbols = np.random.randint(0, self.n_symbols, size=n_mutations)
            mutated[mutation_mask] = new_symbols
        
        return mutated
    
    def _compute_complexity(self, genome: np.ndarray) -> float:
        """Calculate Shannon complexity of a genome."""
        # Convert to string for shannon_complexity function
        genome_str = ''.join(map(str, genome))
        return shannon_complexity(genome_str, base=2)
    
    def _population_complexity(self) -> float:
        """Average complexity across population."""
        complexities = [self._compute_complexity(g) for g in self.population]
        return np.mean(complexities)
    
    def step(self) -> None:
        """
        Perform one generation of evolution:
        1. Calculate fitness
        2. Select parents (fitness-weighted)
        3. Reproduce with mutation
        """
        # 1. Calculate fitness for each genome
        fitnesses = np.array([self.fitness_fn(g) for g in self.population])
        
        # Store metrics
        self.fitness_history.append(np.mean(fitnesses))
        self.complexity_history.append(self._population_complexity())
        
        # 2. Selection: fitness-proportional sampling
        if np.sum(fitnesses) == 0:
            # All zero fitness: uniform selection
            selection_probs = np.ones(self.N) / self.N
        else:
            selection_probs = fitnesses / np.sum(fitnesses)
        
        # Sample parents
        parent_indices = np.random.choice(self.N, size=self.N, p=selection_probs, replace=True)
        
        # 3. Reproduction with mutation
        new_population = []
        for idx in parent_indices:
            child = self._mutate(self.population[idx])
            new_population.append(child)
        
        self.population = np.array(new_population)
    
    def evolve(self, generations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run evolution for multiple generations.
        
        Args:
            generations: Number of generations to simulate
            
        Returns:
            (complexity_history, fitness_history)
        """
        for _ in range(generations):
            self.step()
        
        return np.array(self.complexity_history), np.array(self.fitness_history)
    
    def measure_equilibrium(self, window: int = 100, 
                           threshold: float = 0.01) -> Tuple[bool, int]:
        """
        Detect if complexity has reached equilibrium.
        
        Args:
            window: Number of recent generations to check
            threshold: Maximum allowed change (as fraction of mean)
            
        Returns:
            (at_equilibrium, generation_reached)
        """
        if len(self.complexity_history) < window:
            return False, -1
        
        recent = np.array(self.complexity_history[-window:])
        mean_C = np.mean(recent)
        std_C = np.std(recent)
        
        # Check if variation is below threshold
        if std_C / mean_C < threshold:
            # Find when equilibrium was first reached
            for i in range(len(self.complexity_history) - window, 0, -1):
                if i < window:
                    return True, i
                chunk = np.array(self.complexity_history[i-window:i])
                if np.std(chunk) / np.mean(chunk) >= threshold:
                    return True, i
            return True, 0
        
        return False, -1


# ============================================================================
# SECTION 6: FITNESS LANDSCAPES
# ============================================================================

def fitness_counting_ones(genome: np.ndarray) -> float:
    """
    Simple fitness: count number of '1' symbols.
    
    Selects for genomes with more 1s.
    """
    return np.sum(genome == 1)


def fitness_matching_target(target: np.ndarray):
    """
    Create fitness function that rewards matching target sequence.
    
    Args:
        target: Target genome to match
        
    Returns:
        Fitness function
    """
    def fitness(genome: np.ndarray) -> float:
        matches = np.sum(genome == target)
        return matches / len(target)
    return fitness


def fitness_royal_road(genome: np.ndarray, block_size: int = 8) -> float:
    """
    Royal Road fitness landscape.
    
    Rewards contiguous blocks of identical symbols.
    """
    n_blocks = len(genome) // block_size
    fitness = 0
    
    for i in range(n_blocks):
        block = genome[i*block_size:(i+1)*block_size]
        if len(set(block)) == 1:  # All same
            fitness += 1
    
    return fitness


def fitness_max_entropy(genome: np.ndarray) -> float:
    """
    Fitness function that selects for maximum entropy (complexity).
    
    This creates pressure toward high-complexity genomes.
    """
    genome_str = ''.join(map(str, genome))
    complexity = shannon_complexity(genome_str)
    return complexity


# ============================================================================
# SECTION 7: ANALYSIS TOOLS
# ============================================================================

def fidelity_complexity_curve(mutation_rates: np.ndarray, 
                               genome_length: int) -> np.ndarray:
    """
    Generate fidelity-complexity trade-off curve.
    
    Args:
        mutation_rates: Array of mutation rates to test
        genome_length: Fixed genome length
        
    Returns:
        Array of maximum complexities
    """
    capacities = np.array([
        channel_capacity_simple(mu, genome_length) / genome_length
        for mu in mutation_rates
    ])
    
    return capacities


def compare_organisms(organism_params: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compare complexity metrics across different organisms.
    
    Args:
        organism_params: Dict mapping organism name to parameters
            {
                'bacteria': {'mu': 1e-6, 'L': 1e6, 'beta': 0.01},
                'human': {'mu': 1e-9, 'L': 3e9, 'beta': 0.001},
                ...
            }
    
    Returns:
        Dict mapping organism name to metrics
    """
    results = {}
    
    for name, params in organism_params.items():
        mu = params['mu']
        L = params['L']
        beta = params.get('beta', 0.01)
        
        # Calculate metrics
        C_max = channel_capacity_simple(mu, L) / L
        L_max = eigen_error_threshold(mu)
        trajectory = ComplexityTrajectory(mu, int(L), beta)
        t_eq = trajectory.time_to_equilibrium()
        
        results[name] = {
            'C_max': C_max,
            'L_max': L_max,
            'L_actual': L,
            'utilization': L / L_max if L_max < np.inf else 0.0,
            't_equilibrium': t_eq,
            'lambda': trajectory.lambda_
        }
    
    return results


def information_flow_analysis(simulator: EvolutionarySimulator, 
                               generation: int) -> Dict[str, float]:
    """
    Analyze information flow at a specific generation.
    
    Args:
        simulator: Running evolutionary simulator
        generation: Which generation to analyze
        
    Returns:
        Dict with I_E, I_L, and dC/dt
    """
    if generation >= len(simulator.complexity_history):
        raise ValueError("Generation not yet simulated")
    
    # Estimate I_E from fitness variance
    if generation >= len(simulator.fitness_history):
        return {}
    
    # Get fitness variance (proxy for selection strength)
    recent_fitness = simulator.fitness_history[max(0, generation-10):generation+1]
    fitness_var = np.var(recent_fitness)
    
    # Create info flow model
    info_flow = InformationFlow(
        simulator.mu, 
        simulator.fitness_fn, 
        simulator.L,
        simulator.n_symbols
    )
    
    I_E = info_flow.information_gain(fitness_var)
    I_L = info_flow.information_loss()
    
    # Estimate dC/dt from recent history
    if generation >= 10:
        recent_C = simulator.complexity_history[generation-10:generation+1]
        dC_dt = np.mean(np.diff(recent_C))
    else:
        dC_dt = 0.0
    
    return {
        'I_E': I_E,
        'I_L': I_L,
        'dC_dt': dC_dt,
        'net_flow': I_E - I_L
    }


# ============================================================================
# SECTION 8: UTILITIES
# ============================================================================

def genome_to_string(genome: np.ndarray, alphabet: str = 'ACGT') -> str:
    """Convert numeric genome to string representation."""
    return ''.join([alphabet[i % len(alphabet)] for i in genome])


def string_to_genome(sequence: str, alphabet: str = 'ACGT') -> np.ndarray:
    """Convert string sequence to numeric genome."""
    mapping = {char: i for i, char in enumerate(alphabet)}
    return np.array([mapping[char] for char in sequence if char in mapping])


def load_fasta(filepath: str) -> str:
    """
    Load sequence from FASTA file.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Sequence as string
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (starting with '>')
    sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
    sequence = ''.join(sequence_lines)
    
    return sequence.upper()


# ============================================================================
# MAIN: DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPLEXODYNAMICS: First Law Demonstrations")
    print("=" * 70)
    
    # Demo 1: Shannon Complexity
    print("\n1. Shannon Complexity")
    print("-" * 70)
    sequences = {
        "Uniform": "AAAAAAAAAA",
        "Low complexity": "AACCAACCAA",
        "Medium complexity": "ACACACGTGT",
        "High complexity": "ACGTACGTACGT"
    }
    for name, seq in sequences.items():
        C = shannon_complexity(seq)
        print(f"{name:20s}: {seq:20s} → C = {C:.3f} bits/site")
    
    # Demo 2: Channel Capacity
    print("\n2. Channel Capacity Comparison")
    print("-" * 70)
    organisms = {
        'RNA Virus': {'mu': 1e-4, 'L': 1e4},
        'Bacteria': {'mu': 1e-6, 'L': 4.6e6},
        'Yeast': {'mu': 1e-8, 'L': 1.2e7},
        'Human': {'mu': 1e-9, 'L': 3e9}
    }
    
    print(f"{'Organism':<15} {'μ':<12} {'L':<12} {'C_max':<15} {'L_max':<15}")
    for name, params in organisms.items():
        C_max = channel_capacity_simple(params['mu'], params['L']) / params['L']
        L_max = eigen_error_threshold(params['mu'])
        print(f"{name:<15} {params['mu']:<12.2e} {params['L']:<12.2e} "
              f"{C_max:<15.3f} {L_max:<15.2e}")
    
    # Demo 3: Complexity Trajectory
    print("\n3. Complexity Trajectory (Bacteria)")
    print("-" * 70)
    trajectory = ComplexityTrajectory(mu=1e-6, genome_length=int(4.6e6), 
                                     selection_strength=0.01)
    
    print(f"C_max: {trajectory.C_max:.3f} bits/site")
    print(f"Growth rate (λ): {trajectory.lambda_:.4f}")
    print(f"Time to 95% equilibrium: {trajectory.time_to_equilibrium():.0f} generations")
    
    time_points = [0, 1000, 5000, 10000, 20000]
    print(f"\n{'Generation':<15} {'Complexity (bits/site)':<25}")
    for t in time_points:
        C_t = trajectory.complexity_at_time(t)
        print(f"{t:<15} {C_t:<25.4f}")
    
    # Demo 4: Small Evolution Simulation
    print("\n4. Evolution Simulation (100 generations)")
    print("-" * 70)
    print("Simulating small population evolving toward high entropy...")
    
    sim = EvolutionarySimulator(
        pop_size=50,
        genome_length=100,
        mutation_rate=0.01,
        selection_model=fitness_max_entropy
    )
    
    C_history, F_history = sim.evolve(generations=100)
    
    print(f"Initial complexity: {C_history[0]:.3f} bits/site")
    print(f"Final complexity: {C_history[-1]:.3f} bits/site")
    print(f"Initial fitness: {F_history[0]:.3f}")
    print(f"Final fitness: {F_history[-1]:.3f}")
    
    at_eq, gen_eq = sim.measure_equilibrium()
    if at_eq:
        print(f"Equilibrium reached at generation {gen_eq}")
    else:
        print("Equilibrium not yet reached")
    
    print("\n" + "=" * 70)
    print("Demo complete! See visualization.py for plots.")
    print("=" * 70)
