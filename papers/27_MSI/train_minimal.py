"""
train_minimal.py - MSI Universal Intelligence Benchmark CLI

Run the MSI benchmark from Legg's "Machine Super Intelligence" thesis 
and visualize the agent hierarchy.

Usage:
    # Run a full benchmark of all agents
    python train_minimal.py

Reference: http://www.vetta.org/documents/Machine_Super_Intelligence.pdf
"""

from implementation import (
    UniversalIntelligenceMeasure, 
    GridWorld, 
    PatternSequence, 
    RandomAgent, 
    SimpleRLAgent,
    PredictiveAgent
)


def run_benchmark():
    """
    Runs a formal benchmark comparing baseline, RL, and predictive agents.
    
    Mathematical Process:
    ---------------------
    1. Select a diverse set of environments (proxy for E)
    2. Evaluate agents across these environments
    3. Calculate the Universal Intelligence Score (Upsilon)
    """
    print("\n" + "=" * 60)
    print("MSI UNIVERSAL INTELLIGENCE BENCHMARK")
    print("=" * 60 + "\n")
    
    # Select a diverse set of environments (proxy for E)
    envs = [
        GridWorld(size=3),
        GridWorld(size=7),
        PatternSequence([1, 0, 1]),
        PatternSequence([1, 2, 3, 4, 5])
    ]
    
    measure = UniversalIntelligenceMeasure(envs)
    
    # 1. Evaluate Baseline (Random)
    print("Phase 1: Evaluating Baseline (Random Agent)...")
    base_results = measure.evaluate(RandomAgent(), episodes=10)
    print(f"[OK] Random Upsilon (Normalized): {base_results['upsilon_normalized']:.4f}")
    
    # 2. Evaluate Learning Agent (Simple RL)
    print("\nPhase 2: Evaluating Learning Agent (Simple RL)...")
    rl_agent = SimpleRLAgent()
    learning_results = measure.evaluate(rl_agent, episodes=100)
    print(f"[OK] RL Upsilon (Normalized): {learning_results['upsilon_normalized']:.4f}")

    # 3. Evaluate Theory-Grounded Agent (Predictive)
    print("\nPhase 3: Evaluating Predictive Agent (Solomonoff Proxy)...")
    pred_agent = PredictiveAgent()
    pred_results = measure.evaluate(pred_agent, episodes=100)
    print(f"[OK] Predictive Upsilon (Normalized): {pred_results['upsilon_normalized']:.4f}")
    
    # Final Comparison
    print("\n" + "=" * 60)
    print(f"{'AGENT':<20} | {'UPSILON SCORE':<15}")
    print("-" * 60)
    print(f"{'Random (Baseline)':<20} | {base_results['upsilon_normalized']:>15.4f}")
    print(f"{'Simple RL':<20} | {learning_results['upsilon_normalized']:>15.4f}")
    print(f"{'Predictive (Proxy)':<20} | {pred_results['upsilon_normalized']:>15.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
