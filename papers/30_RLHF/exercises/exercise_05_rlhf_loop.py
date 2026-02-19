"""
exercise_05_rlhf_loop.py - Day 30: RLHF
Goal: Orchestrate the full RLHF loop (Collect -> Label -> Train RM -> Train Policy).
This is the high-level workflow that aligns the agent to preferences.
"""

import numpy as np

def run_rlhf_step(reward_model, policy, oracle, env):
    """
    Runs one step of the RLHF loop:
    1. Collect trajectory
    2. Get feedback from Oracle (on segments)
    3. Update Reward Model
    4. Update Policy
    
    This exercise asks you to implement the high-level logic.
    Assume helper functions exist (or just write pseudocode/comments).
    """
    
    # 1. Collect Data
    # TODO: obs, _ = env.reset() ...
    
    # 2. Get Labels
    # TODO: label = oracle.query(seg1, seg2)
    
    # 3. Train Reward Model
    # TODO: loss = compute_loss(reward_model, seg1, seg2, label)
    # TODO: optimizer.step()
    
    # 4. Train Policy
    # TODO: Use PPO to optimize policy against reward_model
    
    print("RLHF Step Complete (Implemented by You!)")
    
if __name__ == "__main__":
    # Mock objects for testing
    class Mock: pass
    rm = Mock()
    policy = Mock()
    oracle = Mock()
    env = Mock()
    
    try:
        run_rlhf_step(rm, policy, oracle, env)
    except Exception as e:
        print(f"Error (expected if empty): {e}")
