# Quick Start Guide

Welcome to **30u30** - 30 Papers in 30 Days! 🚀

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/30u30.git
cd 30u30

# Install dependencies
pip install -r requirements.txt
```

That's it! No GPU required. Everything runs on CPU.

---

## Day 1: Get Started

Three ways to learn:

### 1️⃣ **Interactive Notebook** (Recommended for beginners)

```bash
cd papers/01_Unreasonable_Effectiveness
jupyter notebook notebook.ipynb
```

Run cells step-by-step. Everything is explained!

### 2️⃣ **Train from Command Line**

```bash
cd papers/01_Unreasonable_Effectiveness
python train_minimal.py --data data/tiny_shakespeare.txt --epochs 200
```

### 3️⃣ **Build from Scratch**

Complete the exercises:

```bash
cd papers/01_Unreasonable_Effectiveness/exercises
python exercise_01_build_rnn.py
```

---

## Folder Structure

Each paper has:

```
papers/XX_Paper_Name/
├── README.md              # Complete guide with theory + examples
├── paper_notes.md         # ELI5 summary
├── implementation.py      # Full implementation (heavily commented)
├── train_minimal.py       # Quick training script
├── visualization.py       # Plotting functions
├── notebook.ipynb         # Interactive Jupyter notebook
├── exercises/             # Practice problems
│   ├── README.md
│   ├── exercise_01_*.py
│   └── solutions/
└── data/                  # Sample datasets
```

---

## Learning Path

### If you're new to deep learning:

1. Read `README.md` for the big picture
2. Read `paper_notes.md` for ELI5 explanation
3. Open `notebook.ipynb` and run cells
4. Try `exercises/` when ready

### If you have ML experience:

1. Skim `README.md` for key ideas
2. Study `implementation.py`
3. Run `train_minimal.py` on your own data
4. Jump to exercises

### If you're advanced:

1. Read the original paper (link in README)
2. Review `implementation.py` for details
3. Complete bonus exercises
4. Move to next day

---

## Tips for Success

✅ **Don't rush** - One day at a time  
✅ **Code along** - Type, don't just read  
✅ **Experiment** - Change hyperparameters, try new data  
✅ **Share** - Post your results with #30u30  
✅ **Ask questions** - Use GitHub discussions

❌ **Don't skip fundamentals** - Later papers build on earlier ones  
❌ **Don't just copy-paste** - Understand each line  
❌ **Don't get discouraged** - Some papers are harder than others

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Training is slow

**Solution:** This is normal! We're using CPU for educational purposes.
- Reduce hidden size
- Reduce sequence length
- Use smaller dataset

### Issue: Loss is not decreasing

**Solution:**
- Check learning rate (try 0.01, 0.1, 0.5)
- Ensure gradient clipping is enabled
- Verify data is loaded correctly

### Issue: Generated text is gibberish

**Solution:**
- Train longer (more iterations)
- Increase hidden size
- Use more training data
- Try different temperature (0.5-1.0)

---

## Daily Routine

Recommended schedule:

- **30-45 min:** Read README + paper_notes
- **45-60 min:** Run notebook / train model
- **30-45 min:** Complete one exercise
- **15 min:** Share learnings, plan next day

**Total: ~2-3 hours per day**

Too much? That's okay:
- Skip exercises initially
- Spend 2 days on harder papers
- Focus on papers most relevant to you

---

## Contributing

Found a bug? Have a better explanation? Want to add exercises?

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Resources

- 📖 [Master reading list](ilya_30_papers.md)
- 💬 [GitHub Discussions](../../discussions) - Ask questions
- 🐛 [Issues](../../issues) - Report bugs
- 📊 [Project board](../../projects) - Track progress

---

## What's Next?

After Day 1, you should understand:
- ✅ How RNNs process sequences
- ✅ Forward and backward propagation
- ✅ Why simple models can be powerful

**Tomorrow:** LSTMs - Solving the vanishing gradient problem!

---

**Ready to start?** Jump into Day 1:

```bash
cd papers/01_Unreasonable_Effectiveness
jupyter notebook notebook.ipynb
```

Let's go! 🔥
