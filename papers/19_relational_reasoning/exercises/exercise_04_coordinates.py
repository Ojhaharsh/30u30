import torch
from implementation import add_coordinates

def test_spatial_logic():
    """
    TODO: Verify the coordinate injection utility.
    Section 3.1: "For the CLEVR task, we found it useful [...] to append 
    the absolute (x, y) coordinates of each object to its feature vector."
    
    1. Create random objects (B=1, N=4, D=8).
    2. Add coordinates using 'add_coordinates'.
    3. Verify the new dimension is D+2.
    4. Verify that the first object's x-coordinate is -1.0.
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    print("-" * 50)
    print("Exercise 4: Spatial/Coordinate Logic")
    print("-" * 50)
    test_spatial_logic()
    print("Exercise 4: [COMPLETE]")
