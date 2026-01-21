"""
Exercise 5 (Project): Shakespeare vs Hemingway Classifier
==========================================================

Goal: Build a classifier that distinguishes between two authors.

Your Task:
- Train two RNNs (one per author)
- Generate samples from each
- Build a classifier
- Analyze stylistic differences

Learning Objectives:
1. Style transfer and analysis
2. Feature extraction from RNNs
3. Classification using generative models
4. Understanding what makes writing styles unique

Time: 1-3 hours
Difficulty: Project ⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import CharRNN


def download_author_data(author_name):
    """
    Download or load author's works.
    
    TODO 1: Get text data for an author
    
    Options:
    - Project Gutenberg (free books)
    - Local files
    - Pre-prepared datasets
    
    Authors to try:
    - Shakespeare
    - Hemingway
    - Jane Austen
    - Mark Twain
    """
    print(f"Loading {author_name}'s works...")
    
    # TODO: Implement data loading
    # You can use:
    # - Project Gutenberg API
    # - Local text files
    # - Pre-downloaded datasets
    
    return None  # Return text string


def train_author_model(text, author_name, hidden_size=128, epochs=50):
    """
    Train RNN on specific author's style.
    
    TODO 2: Train a model for this author
    """
    print(f"\nTraining {author_name} model...")
    
    # TODO: Create vocabulary
    # TODO: Create and train RNN
    # TODO: Save model
    
    return None  # Return trained model


def generate_author_samples(model, idx_to_char, num_samples=100, length=200):
    """
    Generate samples in author's style.
    
    TODO 3: Generate multiple samples
    """
    samples = []
    
    for i in range(num_samples):
        # TODO: Generate one sample
        # sample = model.sample(...)
        # samples.append(sample)
        pass
    
    return samples


def extract_features(text, model=None):
    """
    Extract features from text for classification.
    
    TODO 4: Extract these features:
    
    1. Simple features (no model needed):
       - Average word length
       - Average sentence length
       - Vocabulary richness (unique words / total words)
       - Punctuation frequency
       - Most common words
    
    2. RNN features (if model provided):
       - Hidden state statistics (mean, std)
       - Final hidden state
       - Hidden state trajectory
    
    Returns:
        features: numpy array of feature values
    """
    features = []
    
    # TODO: Extract simple features
    # words = text.split()
    # avg_word_length = np.mean([len(w) for w in words])
    # features.append(avg_word_length)
    
    # TODO: More features...
    
    # TODO: If model provided, extract RNN features
    # if model:
    #     h_states = model.get_hidden_states(text)
    #     features.extend([h_states.mean(), h_states.std()])
    
    return np.array(features)


def build_classifier(features_A, features_B):
    """
    Build classifier to distinguish authors.
    
    TODO 5: Train a simple classifier
    
    You can use:
    - Logistic Regression (sklearn)
    - SVM (sklearn)
    - Simple neural network
    - Even decision trees
    """
    print("\nTraining classifier...")
    
    # TODO: Combine features and labels
    # X = np.vstack([features_A, features_B])
    # y = np.array([0]*len(features_A) + [1]*len(features_B))
    
    # TODO: Split train/test
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # TODO: Train classifier
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    
    # TODO: Evaluate
    # accuracy = clf.score(X_test, y_test)
    # print(f"Accuracy: {accuracy:.2%}")
    
    return None  # Return trained classifier


def analyze_stylistic_differences(samples_A, samples_B):
    """
    Analyze what makes each author's style unique.
    
    TODO 6: Compare these aspects:
    
    1. Vocabulary:
       - Unique words used
       - Most frequent words
       - Rare words
    
    2. Syntax:
       - Average sentence length
       - Sentence complexity
       - Punctuation patterns
    
    3. Themes:
       - Common topics
       - Tone (formal vs casual)
       - Word choice
    
    4. Style:
       - Active vs passive voice
       - Simple vs complex sentences
       - Descriptive vs action-oriented
    """
    print("\n" + "=" * 60)
    print("STYLISTIC ANALYSIS")
    print("=" * 60)
    
    # TODO: Analyze Author A
    print("\nAuthor A:")
    print("  TODO: Add analysis")
    
    # TODO: Analyze Author B
    print("\nAuthor B:")
    print("  TODO: Add analysis")
    
    # TODO: Compare
    print("\nKey differences:")
    print("  TODO: Add comparison")


def visualize_style_space(features_A, features_B, labels_A, labels_B):
    """
    Visualize authors in feature space.
    
    TODO 7: Create 2D visualization
    
    Use PCA or t-SNE to reduce features to 2D
    Plot both authors with different colors
    """
    from sklearn.decomposition import PCA
    
    # TODO: Combine features
    # all_features = np.vstack([features_A, features_B])
    # all_labels = labels_A + labels_B
    
    # TODO: Reduce to 2D
    # pca = PCA(n_components=2)
    # features_2d = pca.fit_transform(all_features)
    
    # TODO: Plot
    # plt.figure(figsize=(10, 8))
    # plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1], label='Author A')
    # plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1], label='Author B')
    # plt.legend()
    # plt.show()
    
    pass


def run_full_pipeline():
    """
    Run the complete Shakespeare vs Hemingway project.
    
    TODO 8: Complete pipeline
    """
    print("=" * 60)
    print("SHAKESPEARE vs HEMINGWAY CLASSIFIER")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    # shakespeare_text = download_author_data('Shakespeare')
    # hemingway_text = download_author_data('Hemingway')
    
    # Step 2: Train models
    print("\n[2/7] Training author models...")
    # model_shakespeare = train_author_model(shakespeare_text, 'Shakespeare')
    # model_hemingway = train_author_model(hemingway_text, 'Hemingway')
    
    # Step 3: Generate samples
    print("\n[3/7] Generating samples...")
    # samples_shakespeare = generate_author_samples(model_shakespeare, ...)
    # samples_hemingway = generate_author_samples(model_hemingway, ...)
    
    # Step 4: Extract features
    print("\n[4/7] Extracting features...")
    # features_shakespeare = [extract_features(s) for s in samples_shakespeare]
    # features_hemingway = [extract_features(s) for s in samples_hemingway]
    
    # Step 5: Train classifier
    print("\n[5/7] Training classifier...")
    # clf = build_classifier(features_shakespeare, features_hemingway)
    
    # Step 6: Analyze styles
    print("\n[6/7] Analyzing styles...")
    # analyze_stylistic_differences(samples_shakespeare, samples_hemingway)
    
    # Step 7: Visualize
    print("\n[7/7] Visualizing...")
    # visualize_style_space(features_shakespeare, features_hemingway, 
    #                       ['Shakespeare']*len(samples_shakespeare),
    #                       ['Hemingway']*len(samples_hemingway))
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)


def bonus_challenges():
    """
    Bonus challenges to extend the project.
    
    TODO 9 (BONUS): Try these extensions:
    
    1. Add more authors (3-way or 4-way classification)
    2. Build a "style transfer" system (convert Shakespeare → Hemingway)
    3. Use only vocabulary features (no RNN)
    4. Build an "authenticity detector" (real vs generated text)
    5. Create a style "spectrum" (from formal to casual)
    """
    print("\n" + "=" * 60)
    print("BONUS CHALLENGES")
    print("=" * 60)
    
    print("\n1. Multi-author classification")
    print("   TODO: Implement")
    
    print("\n2. Style transfer")
    print("   TODO: Implement")


if __name__ == "__main__":
    print(__doc__)
    
    print("\nThis is a full project! Budget 1-3 hours.")
    print("\nData sources:")
    print("  - Project Gutenberg: https://www.gutenberg.org/")
    print("  - Local text files")
    print("  - Pre-downloaded datasets")
    
    # Run the full pipeline
    # run_full_pipeline()
    
    # Bonus challenges
    # bonus_challenges()
    
    print("\n✅ Exercise 5 (Project) complete!")
    print("Check solutions/project_shakespeare_hemingway/ for reference")
