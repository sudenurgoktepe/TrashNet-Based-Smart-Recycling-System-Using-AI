<h1 align="center">♻️ Smart Recycling with AI 🤖</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Recycling-brightgreen?style=for-the-badge&logo=python" alt="Badge">
  <img src="https://img.shields.io/badge/Built%20with-TensorFlow-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge">
</p>

<p align="center">
  An intelligent and eco-friendly waste classification system powered by deep learning.<br>
  Recycle smarter, greener, and faster — all with the help of AI.
</p>

---

## 🌍 Why This Project?

In Turkey, recycling rates are as low as **11.5%**, far behind EU countries.  
At the same time, electricity demand is increasing rapidly, with an average daily consumption of **834,713 MWh**.

That’s why we built an **AI-powered smart recycling bin**:  
✅ Automatically classifies waste using real-time image recognition  
✅ Opens the correct bin compartment only  
✅ Runs on solar energy for **eco-friendly operation**  

---

## 📸 Demo Preview

> *(You can add a gif or image here if available)*

<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="600">
</p>

---

## 🧠 How It Works

1. 📦 Loads and processes the **TrashNet** dataset
2. 🧠 Trains a custom **CNN model** to classify: `plastic`, `metal`, `paper`, `cardboard`, `glass`
3. 📷 Uses OpenCV to capture real-time waste images
4. 🗑️ Detects the waste type and opens the appropriate bin compartment

---

## 📦 Dataset

We used the [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet), which contains 2527 labeled images.

To load the dataset:

```python
from datasets import load_dataset

ds = load_dataset("garythung/trashnet")
