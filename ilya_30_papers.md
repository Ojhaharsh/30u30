# Ilya Sutskever's 30 Papers Reading List

*"If you really learn all of these, you'll know 90% of what matters today"*

---

## Unit 1: The Foundations (Sequence & Recurrence)

**1. The Unreasonable Effectiveness of Recurrent Neural Networks**  
*Andrej Karpathy (2015)*  
The best intuition pump for why "predicting the next token" = intelligence.  
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

**2. Understanding LSTM Networks**  
*Christopher Olah (2015)*  
The mechanics of memory.  
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**3. Recurrent Neural Network Regularization**  
*Zaremba, Sutskever et al. (2014)*  
How to make RNNs actually work without overfitting.  
https://arxiv.org/abs/1409.2329

**4. Keeping Neural Networks Simple by Minimizing the Description Length of the Weights**  
*Hinton, Van Camp (1993)*  
The "Compression = Intelligence" theory origin.  
https://www.cs.toronto.edu/~hinton/absps/colt93.pdf

**5. A Tutorial Introduction to the Minimum Description Length Principle**  
*Grünwald (2004)*  
The mathematical formalism of compression theory.  
https://arxiv.org/abs/math/0406077

**6. The First Law of Complexodynamics**  
*Scott Aaronson (2011)*  
Theoretical physics of complexity.  
https://scottaaronson.blog/?p=762

**7. Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton**  
*Aaronson et al. (2014)*  
Models how complexity grows and then fades. Explains why life/intelligence exists in the "middle state" of the universe.  
https://arxiv.org/abs/1405.6903

---

## Unit 2: The Deep Learning Explosion (Vision & Architectures)

**8. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)**  
*Krizhevsky, Sutskever, Hinton (2012)*  
The Big Bang of Deep Learning.  
https://arxiv.org/abs/1211.5063

**9. Deep Residual Learning for Image Recognition (ResNet)**  
*He et al. (2015)*  
How to train deep networks (scaling).  
https://arxiv.org/abs/1512.03385

**10. Identity Mappings in Deep Residual Networks (ResNet V2)**  
*He et al. (2016)*  
Refining the signal propagation.  
https://arxiv.org/abs/1603.05027

**11. Multi-Scale Context Aggregation by Dilated Convolutions**  
*Yu, Koltun (2015)*  
How to see "global" context without losing resolution.  
https://arxiv.org/abs/1511.07122

**12. Dropout: A Simple Way to Prevent Neural Networks from Overfitting**  
*Srivastava et al. (2014)*  
The standard regularization technique.  
https://jmlr.org/papers/v15/srivastava14a.html

---

## Unit 3: The Transformer & Reasoning

**13. Attention Is All You Need**  
*Vaswani et al. (2017)*  
The architecture that ate the world.  
https://arxiv.org/abs/1706.03762

**14. The Annotated Transformer**  
*Alexander Rush (2018)*  
Code-level understanding of the Transformer.  
https://nlp.seas.harvard.edu/annotated-transformer/

**14. Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)**  
*Bahdanau, Cho, Bengio (2014)*  The precursor to the Transformer (Attention v1).  https://arxiv.org/abs/1409.0473

**15. Order Matters: Sequence to Sequence for Sets**  
*Vinyals et al. (2015)*  Teaching networks that the order of inputs sometimes doesn't matter (Sets vs. Sequences).  https://arxiv.org/abs/1511.06391

---

## Unit 4: Specialized Architectures (Memory & Graphs)

**16. Neural Turing Machines**  
*Graves, Wayne, Danihelka (2014)*  Giving neural networks external RAM (writing/reading memory).  https://arxiv.org/abs/1410.5401

**18. Pointer Networks**  
*Vinyals, Fortunato, Jaitly (2015)*  
Networks that can "point" to their input (essential for combinatorial problems).  
https://arxiv.org/abs/1506.03134

**19. A Simple Neural Network Module for Relational Reasoning**  
*Santoro et al. (2017)*  
https://arxiv.org/abs/1706.01427

**20. Relational Recurrent Neural Networks**  
*Santoro et al. (2018)*  
https://arxiv.org/abs/1806.01822

**21. Neural Message Passing for Quantum Chemistry**  
*Gilmer et al. (2017)*  Proves that "Intelligence" (Graph Networks) is universal—works for biology and chemistry just as well as for language.  https://arxiv.org/abs/1704.01212

**21. Deep Speech 2: End-to-End Speech Recognition**  
*Amodei et al. (2015)*  
End-to-end deep learning applied to audio.  
https://arxiv.org/abs/1512.02595

---

## Unit 5: Generative Models & Scaling

**22. Variational Lossy Autoencoder**  
*Chen et al. (2016)*  
Information theory applied to generation.  
https://arxiv.org/abs/1611.02731

**23. GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism**  
*Huang et al. (2019)*  
How to train really big models across many GPUs.  
https://arxiv.org/abs/1811.06965

**24. Scaling Laws for Neural Language Models**  
*Kaplan et al., OpenAI (2020)*  
The "Physics" of AI. Predicting performance before you train.  
https://arxiv.org/abs/2001.08361

**25. Kolmogorov Complexity and Algorithmic Randomness**  
*Shen, Uspensky, Vereshchagin*  
The pure math of compression.  
https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf

**26. Machine Super Intelligence**  
*Shane Legg (2008)*  
The alignment and safety perspective from a DeepMind founder.  
https://www.vetta.org/documents/Machine_Super_Intelligence.pdf

**27. CS231n: Convolutional Neural Networks for Visual Recognition**  
*Stanford Course (Karpathy, Fei-Fei Li)*  
The Stanford course that solidifies everything.  
https://cs231n.github.io/

---

## The Missing 3 (Modern Extensions)

**28. Proximal Policy Optimization (PPO)**  
*Schulman et al., OpenAI (2017)*  
The algorithm that trained ChatGPT (RLHF).  
https://arxiv.org/abs/1707.06347

**29. Deep Reinforcement Learning from Human Feedback**  
*Christiano et al., OpenAI (2017)*  
The birth of "Human Feedback" (RLHF).  
https://arxiv.org/abs/1706.03741

**30. Hindsight Experience Replay**  
*Andrychowicz et al., OpenAI (2017)*  
How agents learn from failure.  
https://arxiv.org/abs/1707.01495

---

## Additional Resources

**Neural Ordinary Differential Equations**  
https://arxiv.org/abs/1806.07366

**WaveNet: A Generative Model for Raw Audio**  
https://arxiv.org/abs/1609.03499

**World Models**  
https://arxiv.org/abs/1803.10122