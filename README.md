# 🚀 Intelligent Rover Navigation System using Reflex Agents & Hierarchical A*

**Course:** CSE518 – Artificial Intelligence  
**Language:** Python  

---

## 📌 Project Overview

This project implements an intelligent rover navigation system that integrates **reflex-based real-time control** with **hierarchical, battery-aware A\*** path planning. The rover dynamically adapts to obstacles, hazards, cliffs, traps, terrain costs, and battery limitations. A **utility-based cost model** is used to balance safe navigation with scientifically valuable exploration.

All detailed design explanations, algorithms, flowcharts, and analysis are documented in the **project report**, which is included in this GitHub repository.

---

## 🧠 Key Components

- **Reflex Module:**  
  Handles real-time reactions such as obstacle avoidance, low battery handling, hazard/trap recovery, cliff blocking, and recharge triggering.

- **Hierarchical A\* Planner:**  
  Battery-aware A\* planning with recharge-aware high-level routing to ensure safe and feasible navigation.

- **Utility-Based Cost Model:**  
  Enhances standard A\* by incorporating terrain utility to encourage mission-oriented exploration.

---

## 📐 Heuristics Implemented

- Manhattan Distance  
- Euclidean Distance  
- Adaptive Cost Manhattan (H1)  
- Terrain Aggressive Heuristic  
- Obstacle Aversion Heuristic (H2)  

---

## 📊 Evaluation & Comparisons

- **Heuristic performance comparison:**  
  `Group7_Comparison_file_Part2.ipynb`

- **Standard A* vs Utility-based A* comparison:**  
  `Group7_Standard_vs_Utility_Astar_Comparison.ipynb`

---

## 🎥 Simulations

Rover simulation and execution videos are included in the repository to demonstrate reflex behavior, recharge planning, hazard recovery, and utility-driven navigation.

---

## ▶️ How to Run

```bash
python Group7_AI_Project.py
```

For analysis and comparison results, open the provided Jupyter notebooks.

---

## 🧾 Team Members

- **Nitant Jain**
- **Even Patel**

---

## 📄 Note

This project was developed for academic purposes as part of **CSE518 – Artificial Intelligence**.
