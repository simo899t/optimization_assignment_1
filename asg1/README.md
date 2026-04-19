# Optimization Assignment 1

## Case 1 — Path planning with gradient-based optimizers

File: asg1/src/case1.py

Optimizes a 2D trajectory from `(1,1)` to `(100,100)` around circular obstacles by minimizing a path-length + smoothness + obstacle-avoidance objective. Implements gradient descent, strong-bracketing line search, momentum, Nesterov momentum, Newton, Nelder-Mead, and **Adam**.

How to run:

```bash
cd asg1/src
python case1.py
```

`main()` is preconfigured to run the **Adam** optimizer and plot trajectory, convergence, and the adaptive learning rate.

## Case 2 — LeNet-5 on Fashion-MNIST

File: multiple files. (asg1/src/case2.py) + others for testeing

Trains a LeNet-5 classifier on Fashion-MNIST with the tuned **fast_AdamW** configuration found during the assignment:

Run:

```bash
cd asg1/src
python Testing_Fast_AdamW.py
```

## Report

The report with figures: asg1/tex/report.pdf.
