# 📉 lossscape

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](../../actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-black.svg)](package.json)

> **lossscape** is an interactive gradient descent visualizer that demonstrates optimization failure modes, learning rate sensitivity, and loss landscapes in real time. Perfect for machine learning learners to explore optimization behavior on pathological surfaces.

---

## 🚀 Why lossscape??

- **Learn by visualizing**: Instantly see how gradient descent behaves under different learning rates, saddle points, and exploding/vanishing gradients.
- **Multiple optimizers**: Toggle between SGD, Momentum, and Adam-lite.
- **Animated loss surfaces**: 2D landscapes with live GD trajectories.
- **Recruiter-friendly**: Demonstrates strong ML intuition, coding ability, and visualization skills.

---

## 📦 Folder Structure
```
LossScape/
├── src/
│   └── lossscape/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── optimizers.py
│       ├── surfaces.py
│       ├── visualizers.py
├── tests/
│   ├── test_optimizers.py
│   └── test_surfaces.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Libraries**: NumPy, Matplotlib
- **Testing**: Pytest
- **Visualization**: Animated GD trajectories on 2D loss surfaces

---

## 🏁 Getting Started

Clone and launch the development server:

```bash
git clone https://github.com/0xSris/bug-tracer.git
cd lossscape
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run the App:
```bash
python -m src.lossscape.app
```
Access the app at [http://localhost:5173](http://localhost:5173).

---

## 💡 Usage

Visualizing Loss Landscapes
- Launch the app to open a live interactive window.
- Choose a loss surface from the available options.
- Toggle between optimizers: SGD, Momentum, Adam-lite.
- Adjust learning rates to observe sensitivity and failure modes.
- Watch gradient descent trajectories animate in real time.

Learning Objectives
- Observe pathological loss surfaces.
- Understand saddle points, exploding/vanishing gradients, and learning rate effects.
- Compare optimizer behavior visually.

## 🗂️ Docs & Support

- **Issues:** [issues](../../issues)
- **Documentation:** See the [docs](docs/) folder (if available).
- **Contributing:** See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## 👤 Maintainers & Contributing

Maintained by [@0xSris](https://github.com/0xSris) and contributors.

We welcome PRs and issues! Please review [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the guidelines.


```
## 🤩 Why This Stands Out
- Professional ML visualization: Shows complex optimization concepts clearly.
- Interactive & shareable: Can be used to demonstrate ML intuition in presentations or interviews.
- Portfolio-ready: Demonstrates coding, testing, and visualization skills in one project.
```

_See [LICENSE](LICENSE) for license info. For more help, open an [issue](../../issues)._
