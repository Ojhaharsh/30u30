import torch

def prove_counting_bias(num_objects: int):
    """
    Solution 7: Why sum() is better for counting than mean().
    """
    # Create N objects each with the 'property' of intensity 1.
    objects = torch.ones(num_objects)
    
    sum_res = objects.sum().item()
    mean_res = objects.mean().item()
    
    # Sum returns the count (N). Mean returns the average intensity (1).
    # If we need to count objects, only sum_res changes as N changes.
    return sum_res == num_objects and mean_res == 1.0
