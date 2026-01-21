"""
Exercise 4: Long-Range Dependencies
====================================

Goal: Test how far back LSTMs can remember information.

Your Task:
- Create a "remember first character" task
- Test at different distances (5, 10, 20, 50, 100 steps)
- Measure accuracy vs distance
- Compare with vanilla RNN (optional)

Learning Objectives:
1. Understand LSTM's memory capacity
2. See where vanishing gradients still occur
3. Learn practical limits of sequence modeling
4. Design tasks that test memory

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


def generate_memory_task(distance, num_samples=1000, vocab_size=10):
    """
    Generate "remember first character" task.
    
    Format: [A, noise, noise, ..., noise, ?] -> A
    
    Args:
        distance: Number of noise characters between start and end
        num_samples: Number of training examples
        vocab_size: Size of vocabulary
    
    Returns:
        X: Input sequences (num_samples, distance+2)
        y: Target labels (num_samples,)
    
    TODO 1: Implement data generation
    - First character: random from 0 to vocab_size-1
    - Middle characters: random noise
    - Last character: special "query" token (vocab_size)
    - Target: first character
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        # TODO: Generate one example
        # first_char = np.random.randint(0, vocab_size)
        # noise = [np.random.randint(0, vocab_size) for _ in range(distance)]
        # query = vocab_size  # Special token
        # sequence = [first_char] + noise + [query]
        pass
    
    return np.array(X), np.array(y)


def train_on_distance(distance, vocab_size=10, hidden_size=32, num_iterations=500):
    """
    Train LSTM on memory task at specific distance.
    
    Args:
        distance: Distance to test
        vocab_size: Vocabulary size
        hidden_size: Hidden layer size
        num_iterations: Training iterations
    
    Returns:
        accuracy: Final test accuracy
    
    TODO 2: Implement training
    - Generate training data
    - Create LSTM
    - Train for num_iterations
    - Test on held-out set
    - Return accuracy
    """
    print(f"\nTraining at distance {distance}...")
    
    # TODO 3: Generate data
    # X_train, y_train = generate_memory_task(distance, num_samples=1000)
    # X_test, y_test = generate_memory_task(distance, num_samples=200)
    
    # TODO 4: Create LSTM
    # Note: input_size = vocab_size + 1 (includes query token)
    # lstm = LSTM(...)
    
    # TODO 5: Training loop
    learning_rate = 0.01
    
    for iteration in range(num_iterations):
        # TODO: Sample random batch
        # idx = np.random.randint(0, len(X_train))
        # inputs = X_train[idx]
        # target = y_train[idx]
        
        # TODO: Forward, backward, update
        pass
    
    # TODO 6: Evaluate on test set
    correct = 0
    # for inputs, target in zip(X_test, y_test):
    #     prediction = predict(lstm, inputs)
    #     if prediction == target:
    #         correct += 1
    
    accuracy = 0.0  # Replace with: correct / len(X_test) * 100
    print(f"Distance {distance}: {accuracy:.1f}% accuracy")
    
    return accuracy


def test_all_distances(distances=[5, 10, 20, 50, 100]):
    """
    Test LSTM at multiple distances.
    
    TODO 7: Test each distance and return results
    """
    accuracies = []
    
    for distance in distances:
        # TODO 8: Train and evaluate at this distance
        acc = train_on_distance(distance)
        accuracies.append(acc)
    
    return distances, accuracies


def plot_results(distances, accuracies):
    """
    Plot accuracy vs distance.
    
    TODO 9: Create visualization
    - Plot distances on x-axis
    - Plot accuracies on y-axis
    - Add markers for each point
    - Add horizontal line at random guess level (1/vocab_size)
    """
    plt.figure(figsize=(10, 6))
    
    # TODO: Your plotting code
    # plt.plot(distances, accuracies, marker='o')
    # plt.axhline(y=10, color='r', linestyle='--', label='Random guess')
    # plt.xlabel('Distance (steps)')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Long-Range Dependency Test')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    pass


def compare_with_rnn(distances):
    """
    Optional: Compare LSTM with vanilla RNN.
    
    TODO 10 (BONUS): Implement vanilla RNN and compare
    - RNN should fail beyond ~10 steps
    - LSTM should succeed up to 50+ steps
    - Plot both on same graph
    """
    print("\nBonus: Comparing with vanilla RNN...")
    print("(Not implemented - try it yourself!)")


def analyze_findings(distances, accuracies):
    """
    Analyze results and draw conclusions.
    
    TODO 11: Answer these questions:
    1. At what distance does performance start to degrade?
    2. What's the maximum distance the LSTM can handle?
    3. How does this compare to vanilla RNN (if you implemented it)?
    4. What factors limit the maximum distance?
    """
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print("\nYour analysis here...")
    
    # TODO: Print statistics
    # - Best distance
    # - Worst distance
    # - Average accuracy
    # - Where accuracy drops below threshold


if __name__ == "__main__":
    print(__doc__)
    
    print("Testing LSTM memory at different distances...")
    print("=" * 60)
    
    # TODO 12: Run the experiment
    distances = [5, 10, 20, 50, 100]
    
    # Test all distances
    # distances, accuracies = test_all_distances(distances)
    
    # Plot results
    # plot_results(distances, accuracies)
    
    # Analyze
    # analyze_findings(distances, accuracies)
    
    # Optional: Compare with RNN
    # compare_with_rnn(distances)
    
    print("\n✅ Exercise 4 complete!")
    print("Compare with solutions/solution_04_long_range_deps.py")
