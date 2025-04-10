<h1 align="center">♻️ Smart Recycling with AI 🤖</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Artificial%20Intelligence-Recycling-brightgreen?style=for-the-badge&logo=python" alt="Badge">
  <img src="https://img.shields.io/badge/Built%20with-TensorFlow-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge">
</p>

<p align="center">
  An AI-powered, eco-friendly waste classification system.<br>
  Designed for a smarter, greener, and more sustainable future.
</p>

---

## 🌍 Why This Project?

Recycling rates in Turkey are currently around **11.5%**, far below the European average.  
Meanwhile, the daily electricity consumption exceeds **834,713 MWh**.

That’s why we developed a **smart recycling bin powered by artificial intelligence**:  
✅ Automatically classifies waste using image recognition  
✅ Opens only the appropriate bin compartment  
✅ Designed to run on **solar energy** for eco-friendly operation  

---
## 🧠 How It Works

1. 📦 Loads and processes the **TrashNet** dataset  
2. 🧠 Trains a CNN model for 5 categories: `plastic`, `metal`, `paper`, `cardboard`, `glass`  
3. 📷 Captures real-time video using OpenCV  
4. 🗑️ Classifies waste and opens the correct bin slot

⚠️ **Note:** This project currently runs in a **computer-based environment**. Waste images are captured via a webcam and classified locally on the computer.

In future development, the system will be upgraded into a **fully autonomous smart bin** with:

- 📷 An embedded camera system  
- ☀️ Solar-powered operation  

making the project **sustainable, mobile, and hardware-integrated**.

---

## 📦 Dataset

We used the [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet), developed by Stanford University. It contains 2527 labeled images.

### To Load the Dataset:
```python
from datasets import load_dataset

ds = load_dataset("garythung/trashnet")
