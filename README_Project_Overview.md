# WhiteBit – Smart Mixed-Precision Inference with Reinforcement Learning

## 🚀 Project Overview

**WhiteBit** is a research-driven hackathon project that explores how **reinforcement learning** can be used to optimize the precision used in **neural network inference** — dynamically deciding when to use **FP8** or **FP16** to maximize **efficiency** without sacrificing **accuracy**.

### 🧠 Motivation

Modern neural networks can run faster and more energy-efficiently when using lower-precision operations like FP8. However, some layers are **more sensitive** than others, and uniform precision can hurt accuracy. WhiteBit uses **RL to learn which layers or operations** can be safely run in FP8, and which require higher precision.

The result: an **adaptive inference strategy** that finds the best trade-off between **speed** and **accuracy** — learned automatically.

---

## 🔍 What WhiteBit Does

- Builds a **small, real-world-feeling neural net** for classification (MNIST/FashionMNIST)
- Measures **layer sensitivity** using condition numbers or loss gradients
- Lets an **RL agent decide** precision per layer (FP8 or FP16)
- Visualizes the learned precision policy and its effect on:
  - Model **accuracy**
  - Inference **compute cost**
- Provides a teaching mode to **learn RL from first principles** with visual examples

---

## 🎯 Goals

| Goal                          | Status     |
|-------------------------------|------------|
| ✅ Optimize precision dynamically with RL  | In progress |
| ✅ Visualize layer-wise precision policy   | In progress |
| ✅ Plot accuracy vs. cost trade-off        | In progress |
| 🧪 Add RL teaching mode via Streamlit + chatbot | Planned     |
| 🧠 Integrate Venice AI for interactive tuning | Planned     |

---

## 🗺️ Architecture

whitebit/ ├── core/ # Core RL and model logic │ ├── whitebit_model.py │ └── rl_agent.py │ ├── envs/ # Gym-compatible environments │ └── inference_env.py │ ├── visuals/ # Visualizations and metrics │ └── model_map_plot.py │ ├── streamlit_app/ # UI (teaching/demo mode) ├── docker/ # Docker deployment └── README_Project_Overview.md

---

## 💡 Educational Twist

WhiteBit also includes a **teaching layer**: an interactive interface that helps learners understand RL concepts like:
- Policy, value function, TD(λ)
- Exploration-exploitation trade-offs
- Design choices for state/action/reward spaces

All backed by live visualizations and chatbot interaction via **Venice AI**.

---

## 📈 Optimization Trade-Off

We optimize for the **best position on the Pareto frontier**:

markdown
Copy
Edit
    ▲
Accuracy │ ←←← Ideal │ │ ● FP16 only │ ● WhiteBit (RL-optimized) │ ● FP8 only └────────────────────────▶ Compute Cost

yaml
Copy
Edit

---

## 🧪 Try It Out

Coming soon:
- `toy_mixed_precision.py`: minimal demo with condition-based precision switching
- `train_whitebit.py`: RL loop to learn optimal precision decisions
- `streamlit_app/`: visualize model precision, decisions, and accuracy in real-time

---

## 🙌 Authors

- Gordon Koehn – Streamlit, Docker, Venice AI integration
- Jonas Petersen – RL research, architecture