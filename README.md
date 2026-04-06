# SLOTH

**sloth** is a small educational LLM written entirely from scratch in pure C, with no non-unix dependencies.

The goal of the project is not raw speed, scale, or production readiness. The goal is to **understand how a transformer-based language model actually works**, end to end, by implementing the core mechanisms manually and documenting them heavily in the source code (not necessarily in the most efficient way).

If you want to follow a minimal but fairly complete implementation of a GPT-style model (including embeddings, positional encoding,self-attention, feedforward layers, backpropagation) this project is for you.

The code is intentionally verbose and heavily commented so you can follow the logic **function by function**.

## Philosophy

`sloth` is meant to be:

- **educational**
- **readable**
- **hackable**
- **dependency-light**
- **Unix-friendly**

This is not a highly optimized training system, and it is not trying to compete with existing frameworks. It is a project for learning, experimentation, and understanding the internals of LLMs by building them yourself.

That said, if you want to squeeze as much speed as possible out of this implementation, compile it with:

```bash
gcc -O3 -march=native -lm llm.c -o llm
```

This README.md has been written by chatGPT. Apologies.