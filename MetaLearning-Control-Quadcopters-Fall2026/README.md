# Meta-Learning for Quadcopter Control

This repository provides the code and experimental framework for learning neural network–based
quadcopter controllers via meta-learning. The goal is to enable rapid adaptation to previously unseen
wind disturbances in hover and regulation tasks using only a small number of samples. The project applies **Model-Agnostic Meta-Learning** ([MAML.pdf](./MAML.pdf)) to learn a neural
feedback policy initialization that can be efficiently fine-tuned to new disturbance conditions via a
small number of gradient updates.

**Project report:** [DECODE_Report_Goffin.pdf](./DECODE_Report_Goffin.pdf)

---

## Key Results 
- The neural network controller matches a classical LQR-like behavior on basic regulation tasks.
- Standard neural network training can struggle when data are unbalanced across wind conditions.
- MAML yields **faster and more reliable few-shot adaptation** to unseen wind conditions, on both **linearized** and **nonlinear** dynamics.

---

## Future Work
- Extend the framework to higher-fidelity nonlinear models to further reduce the sim-to-real gap.
- Incorporate formal stability guarantees into the learning-based controller.
- Validate the approach on real-world quadcopter hardware.

---

## Installation
```bash
git clone https://github.com/DecodEPFL/MetaLearning-Control-Quadcopters.git
cd MetaLearning-Control-Quadcopters

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Acknowledgements
Semester Project, EPFL DECODE Lab, Fall 2026  
Author: Cyril Goffin  
Supervision: Daniele Martinelli & Nicolas Kirsch  
Professor: G. Ferrari Trecate

---
