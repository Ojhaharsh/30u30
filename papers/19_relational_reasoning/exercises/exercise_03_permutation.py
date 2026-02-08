import torch
from implementation import RelationNetwork

def verify_invariance():
    """
    TODO: Programmatically prove that the RN is permutation invariant.
    Section 2.1 states: "RNs are invariant to the order of objects in the set."
    
    1. Create a set of random objects (B=1, N=5, D=8).
    2. Create a RelationNetwork.
    3. Forward pass objects.
    4. Shuffle the objects along the N dimension.
    5. Forward pass shuffled objects.
    6. Assert that the outputs are identical.
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    print("-" * 50)
    print("Exercise 3: Order Invariance Proof")
    print("-" * 50)
    verify_invariance()
    print("Exercise 3: [COMPLETE]")
