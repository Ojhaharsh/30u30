"""
Scaling Law Diagnostic Simulation
=================================

Train multiple models of incrementally larger sizes to verify 
Kaplan et al.'s results empirically.

Usage:
    python train_minimal.py --d_models 64 128 256
    python train_minimal.py --steps 100 --batch-size 32

Traceability:
- Equation 2.1 (Compute C)
- Section 3 (N-scaling)
- Section 4 (Parameter efficiency)

Author: 30u30 Project
License: CC BY-NC-ND 4.0
"""

import torch
import torch.optim as optim
from implementation import KaplanTransformer, ComputeEconomy, ScalingDataset
import json
import time
import math

# --- MASTER CONFIGURATION ---
VOCAB_SIZE = 128
SEQ_LEN = 64
SWEEP_D_MODELS = [64, 128, 256, 384] # Sweeping N
N_SAMPLES = 2048
BATCH_SIZE = 32
TRAIN_STEPS = 50 # Small number for demonstration; in paper it's 10^5+

def run_scaling_sweep():
    print("="*60)
    print("SCALING LAW DIAGNOSTIC SIMULATION")
    print(f"Goal: Verify L(N) power-law over {len(SWEEP_D_MODELS)} model sizes.")
    print("="*60)

    dataset = ScalingDataset(VOCAB_SIZE, SEQ_LEN, N_SAMPLES)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for d_model in SWEEP_D_MODELS:
        # 1. Initialize Kaplan Model (Exhaustive Traceability)
        model = KaplanTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            max_seq_len=SEQ_LEN
        ).to(device)
        
        n_params = model.count_parameters("Kaplan")
        n_total = model.count_parameters("Total")
        print(f"\n[RUNNING] N_Kaplan: {n_params:,} (Total: {n_total:,})")
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 2. Training Loop with Diagnostic Logging
        model.train()
        start_time = time.time()
        step_losses = []
        
        for step, batch in enumerate(dataloader):
            if step >= TRAIN_STEPS: break
            
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            step_losses.append(loss.item())
            if step % 10 == 0:
                print(f"  Step {step}/{TRAIN_STEPS} | Loss: {loss.item():.4f}")

        # 3. Analytics & Bottleneck Detection
        final_loss = sum(step_losses[-5:]) / 5 # Average last 5 steps
        elapsed = time.time() - start_time
        tokens_seen = TRAIN_STEPS * BATCH_SIZE * SEQ_LEN
        c_pfdays = ComputeEconomy.calculate_c_pfdays(n_params, tokens_seen)
        
        # Simple Bottleneck Detection: 
        # If loss barely moved in the last half, it might be data-bottlenecked 
        # (Since we are using a very small synthetic dataset)
        improvement = step_losses[0] - final_loss
        bottleneck = "None (Learning)" if improvement > 0.5 else "Potential Data Bottleneck"
        
        print(f"[RESULT] Final Loss: {final_loss:.4f} | Compute: {c_pfdays:.2e} PF-days")
        print(f"[DIAGNOSTIC] {bottleneck}")

        results.append({
            "N": n_params,
            "L": final_loss,
            "D": tokens_seen,
            "C_pfdays": c_pfdays,
            "elapsed": elapsed
        })

    # 4. Save for Dashboard
    with open("scaling_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[OK] Simulation complete. Scaling results saved to scaling_results.json")

if __name__ == "__main__":
    run_scaling_sweep()
