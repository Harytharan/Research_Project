# AI-Driven Decision Support System for Sustainable Paddy Farming Optimization

## 📌 Project Overview
Paddy farming in Sri Lanka faces multiple challenges such as pest attacks, improper fertilizer application, high cultivation costs, and unstable market price fluctuations. Farmers and agricultural officers often make decisions without accurate data-driven insights, resulting in yield loss, increased cost, and environmental damage due to chemical misuse.

This project proposes an **AI-Driven Decision Support System** as a **web-based application** that integrates multiple intelligent prediction and recommendation modules to optimize sustainable paddy farming. The system provides actionable insights using machine learning and visual dashboards to support decision-making for farmers, Agricultural Department officers, and Agri Business Centers.

---

## 🎯 Key Objectives
- Predict pest attack risks using IoT + weather data.
- Provide intelligent fertilizer recommendations and expected yield forecasting.
- Estimate cultivation cost for improved budgeting.
- Forecast paddy price and demand trends for market planning.
- Provide a unified dashboard for decision support with visual insights.

---

## 🧩 System Modules
| Module | Description |
|--------|-------------|
| **Pest Attack Prediction** | Predicts pest risks using environmental conditions and sensor data. |
| **Intelligent Fertilizer Recommendation & Yield Prediction** | Recommends top fertilizers and forecasts expected yield using ML models. |
| **Cultivation Cost Estimation** | Estimates total cost based on inputs, labor, transport, and operational factors. |
| **Paddy Price & Demand Forecasting** | Predicts price and demand trends using time-series forecasting and sentiment analysis. |

---

## 👥 Target Users / Beneficiaries
- **Paddy Farmers** – Better planning, improved yield and fertilizer usage.
- **Agricultural Officers** – Decision support tool for advisory services.
- **Agriculture Business Centers** – Improved regional planning and recommendations.

---

## 🏗️ System Architecture
The system is designed as a modular web application with integrated AI pipelines.

### ✅ Architecture Diagram
> 📌 *Insert your architecture diagram image here*  
Add an image named `architecture.png` inside a folder called `docs/` and reference it below:

![System Architecture](docs/architecture.png)

---

## 🔄 High-Level Workflow
1. User logs into the web application.
2. Selects a module (Pest / Fertilizer & Yield / Cost / Price & Demand).
3. Inputs data via manual input form or IoT sensor mode (where available).
4. Data is preprocessed and passed through trained machine learning models.
5. Output predictions and recommendations are visualized in the dashboard.
6. Reports and predictions can be saved for tracking and future analysis.

---

## 🧠 Technologies & Tools Used
### ✅ Frontend
- React / Angular / Vue *(update based on your team choice)*
- HTML, CSS, JavaScript
- Chart library (Chart.js / Recharts / etc.)

### ✅ Backend
- Node.js + Express OR Django OR Flask *(update based on your team choice)*
- REST API architecture

### ✅ Machine Learning / AI
- Python
- Scikit-learn
- Pandas, NumPy
- Random Forest
- Gradient Boosting
- Model evaluation metrics:
  - Accuracy (classification)
  - R² Score, MSE (regression)

###  Database
- MongoDB / MySQL / PostgreSQL *(update based on your project)*
- Firebase *(optional if used)*

###  IoT / Sensors (Optional Mode)
- Soil moisture sensor
- Soil temperature sensor
- Air temperature sensor
- Humidity sensor
- Arduino / ESP32 / Raspberry Pi *(update based on your group)*

---

##  Project Dependencies
### Python Dependencies
Install required packages:
```bash
pip install -r requirements.txt
