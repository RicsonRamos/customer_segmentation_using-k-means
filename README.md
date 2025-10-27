# 📊 Customer Segmentation using K-Means Clustering

## 📝 Project Overview

This project focuses on applying the K-Means clustering algorithm to segment a supermarket chain's customer base. By analyzing purchasing behavior and satisfaction data, the goal is to group customers into distinct, actionable segments. This segmentation allows the marketing and CRM teams to develop personalized strategies, optimize promotional campaigns, and enhance customer retention efforts.

---

## 🎯 Objectives

1.  **Data Transformation:** Aggregate transactional data to the customer level to create key segmentation metrics (e.g., Average Purchase Value, Frequency, and Average Rating).
2.  **Optimal K Determination:** Employ the **Elbow Method** and **Silhouette Coefficient** to accurately determine the optimal number of clusters (*K*).
3.  **Clustering:** Apply the K-Means algorithm to group similar customers.
4.  **Cluster Profiling:** Analyze the characteristics of each resulting segment (value, frequency, satisfaction) and provide specific business recommendations.

---

## ⚙️ Technologies and Libraries

| Category | Library/Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python | Primary programming language used. |
| **Data Handling** | Pandas, NumPy | Data cleaning, manipulation, and transformation. |
| **Clustering** | Scikit-learn (KMeans) | Implementation of the clustering algorithm. |
| **Optimization** | Kneed | Used to programmatically identify the optimal "elbow" in the inertia curve. |
| **Visualization** | Matplotlib, Seaborn, Plotly | Used for Exploratory Data Analysis (EDA) and visualizing cluster results. |

---

## 🛠️ Methodology (Pipeline)

The project followed a standard Data Science and Machine Learning workflow:

1.  **Data Loading & EDA:** Initial inspection of the supermarket sales dataset.
2.  **Feature Engineering:**
    * Creation of customer-level metrics (e.g., aggregating individual transactions).
    * Encoding of categorical variables (One-Hot Encoding) for inclusion in the model.
3.  **Preprocessing:**
    * **Scaling:** All numerical features were normalized using `StandardScaler` to ensure features contribute equally to the distance calculation in K-Means.
4.  **Optimal K Definition:** Using the Elbow Method to select the best number of segments.
5.  **K-Means Modeling:** Training the K-Means model with the chosen *K* and assigning cluster labels to the customer records.
6.  **Interpretation & Business Insights:** Analyzing the mean values for each feature within each cluster to define the segment profiles and proposing actionable business strategies.

---

## 💡 Key Results and Insights

The analysis identified **4 optimal clusters**, each representing a unique customer profile.

| Cluster | Profile | Value/Behavior | Suggested Business Action (CRM) |
| :--- | :--- | :--- | :--- |
| **0** | **Premium/High Rating** | High total value purchases, strong satisfaction score. | **Loyalty:** Offer VIP programs, dedicated service, and high-margin product cross-selling. |
| **1** | **High Volume Buyer** | High volume of units purchased, consistently high ticket value. | **Retention:** Negotiate bulk deals, offer corporate agreements, or subscription models to secure recurring revenue. |
| **2** | **Frequent/Satisfied** | High purchase frequency, good satisfaction score, medium value. | **Upsell:** Run "buy more, save more" promotions and product bundle campaigns to increase the average transaction value (AOV). |
| **3** | **Low Value/Low Satisfaction** | Lowest purchase value and lowest average rating. | **Recovery:** Launch targeted feedback surveys (NPS) to understand dissatisfaction; use highly personalized offers to incentivize a second chance purchase. |

---

## 📂 Repository Structure
