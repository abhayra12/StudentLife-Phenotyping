# 🎤 Presentation Guide — StudentLife Stress Prediction

> **Audience:** High school science fair / research presentation  
> **Duration:** 10–15 minutes  
> **Key message:** "Your phone already knows when you're stressed."

---

## 📋 Slide-by-Slide Guide

### Slide 1: Title Slide
**Title:** *Can Your Phone Tell When You're Stressed?*  
**Subtitle:** Predicting Student Stress from Passive Phone Sensor Data  
**Your name, school, date**

> 🎯 **Speaker notes:** "What if your phone could detect you're stressed before you even realize it? In this project, I used real data from 48 college students to find out."

---

### Slide 2: The Problem
**Title:** *Student Stress is a Growing Crisis*

**Bullet points:**
- 70% of college students report significant stress (APA, 2023)
- Stress affects grades, sleep, social life, and health
- Students often don't recognize or report stress until it's severe
- **What if we could detect stress passively — without asking?**

**Visual:** Simple infographic with stress statistics

> 🎯 **Notes:** "Most mental health monitoring relies on asking people how they feel. But what if the data your phone already collects could tell us the answer?"

---

### Slide 3: The Big Idea
**Title:** *Passive Sensing → Stress Prediction*

**Explain the two types of data:**

| Passive Data (Phone Sensors) | Active Data (Self-Reports) |
|---|---|
| Physical activity (accelerometer) | "How stressed are you?" (1-5) |
| Screen time (lock/unlock) | Sleep quality rating |
| Audio environment (voice/noise) | Social contact count |
| WiFi locations visited | — |
| Conversation duration | — |
| Charging patterns | — |

**Key insight:** "We use passive sensor data to *predict* the self-reported stress — the phone tells us what the student tells themselves."

---

### Slide 4: The Dataset
**Title:** *StudentLife — 48 Students, 10 Weeks*

- **Source:** Dartmouth College (peer-reviewed, published 2014)
- **Participants:** 48 students across a full academic term
- **Sensor data:** 10 types (activity, audio, screen, WiFi, GPS, etc.)
- **EMA responses:** 2,289 self-reported stress levels
- **Surveys:** PHQ-9 depression, Perceived Stress Scale, Big Five personality

**Visual:** Timeline showing March 27 → June 5, 2013  
**Include:** `reports/figures/ema/05_missing_data.png` (response rate over time)

> 🎯 **Notes:** "This is real data from a real study. Students carried phones that recorded their behavior 24/7, and they also answered stress surveys multiple times a day."

---

### Slide 5: How Stress Looks in Data
**Title:** *What Students Reported*

**Visual:** `reports/figures/ema/01_stress_distribution.png`

**Key stats:**
- 45% said "A little stressed" (most common)
- 12% said "Stressed out" (highest level)  
- Only 6% said "Feeling great"
- Average stress was 3.74/5 (leaning toward stressed)

> 🎯 **Notes:** "Most students were at least somewhat stressed most of the time. Only 6% ever reported feeling great."

---

### Slide 6: Stress Changes Over the Term
**Title:** *Stress Isn't Constant*

**Visual:** `reports/figures/ema/02_stress_over_time.png`

**Point out:**
- Stress peaks during midterms and finals
- Response rates drop when students are busiest (ironic!)
- Weekend stress is lower than weekday

**Visual:** `reports/figures/ema/03_stress_time_patterns.png`

---

### Slide 7: One Student's Full Story
**Title:** *Deep Dive: Participant u59*

**Visual:** `reports/figures/deep_dive/01_timeline_u59.png`

**Walk through:**
- "This is one student's entire term, tracked through their phone"
- Top panels = what the phone sensed (activity, screen time, audio, etc.)
- Bottom panel = stress self-reports (dots colored by level)
- "Notice how phone behavior changes before high-stress reports"

> 🎯 **Notes:** "This is the most powerful slide. Point to specific dates where you can see the phone data shifting before a stress report."

---

### Slide 8: Does Phone Data Actually Correlate?
**Title:** *The Key Question: Correlation*

**Visual:** `reports/figures/correlation/01_sensor_stress_correlation.png`

**Key findings:**
- **Physical activity** correlates most strongly (r=0.09, p<0.001)
- Stressed students were **24% less active** and visited **9% fewer locations**
- More **voice/conversation** → less stressed
- **Screen time** and **dark time** showed weaker but visible patterns

**Visual:** `reports/figures/correlation/02_high_vs_low_stress.png`

> 🎯 **Notes:** "The correlations are small individually, but when you combine all 62 sensor features together, patterns emerge that a machine learning model can learn."

---

### Slide 9: The Machine Learning Approach
**Title:** *From Sensor Data to Stress Prediction*

**Diagram:**
```
6 hours of phone data → 62 features → ML Model → Predicted stress level
                                         ↕
                              Compared against EMA self-report (ground truth)
```

**Explain the pipeline:**
1. For each stress self-report, grab the 6 hours of sensor data before it
2. Compute 62 features (averages, sums, ratios, etc.)
3. Feed into Random Forest / XGBoost
4. Model outputs predicted stress level (1-5)
5. Compare against what the student actually reported

**Data split:** Train 70% → Validate 15% → Test 15%  
*(Chronological split — no future data leaking into training!)*

---

### Slide 10: Results
**Title:** *How Well Can We Predict Stress?*

| Model | Accuracy | F1 Score |
|---|---|---|
| Random guess | 20% | — |
| Always guess "stressed" | 45% | — |
| **Random Forest** | **40%** | **0.33** |
| **XGBoost** | **39%** | **0.33** |

**Visual:** `reports/figures/modeling/` (confusion matrices)

**Interpretation:**
- 2× better than random on a 5-class problem
- Model correctly identifies stressed vs. not-stressed states
- Hardest to distinguish: "Feeling good" vs "A little stressed"

> 🎯 **Notes:** "40% might not sound impressive, but remember: we're predicting a 5-level emotional state from *only phone sensor data*. No questions asked. Random would be 20%."

---

### Slide 11: What Matters Most
**Title:** *Feature Importance — What the Phone 'Sees'*

**Visual:** Feature importance chart from model results

**Top predictors:**
1. 📱 **Phone charging patterns** — irregular charging = disrupted routines
2. 📍 **WiFi location diversity** — fewer unique locations = withdrawal
3. 🏃 **Physical activity** — less movement during stress
4. 🎤 **Voice detection** — less conversation when stressed
5. 🌙 **Dark time patterns** — disrupted sleep patterns

> 🎯 **Notes:** "The model learned that changes in daily routine — charging, movement, social interaction — are the strongest signals of stress."

---

### Slide 12: The Bigger Picture
**Title:** *Why This Matters*

**Applications:**
- 🏥 **Early warning system** for student mental health
- 📱 **No burden** on the student — phone collects data automatically
- 🔔 **Proactive intervention** — alert counselors before crisis
- 🎓 **University wellness programs** could use this for student support

**Limitations (be honest!):**
- Correlations are small (this is behavior, not physics)
- 5-class prediction is hard — binary (stressed/not) works better
- Privacy concerns with phone monitoring
- Individual differences are large (what works for one person ≠ another)

---

### Slide 13: Future Work
**Title:** *Where This Could Go*

- **Deep learning** (LSTM/Transformer) for temporal patterns
- **Personalized models** trained per-student
- **Multi-modal fusion** (phone + wearable + social media)
- **Real-time app** that warns you when stress is rising
- **Longitudinal** — track stress across semesters, not just one term

---

### Slide 14: Conclusion
**Title:** *Yes, Your Phone Knows When You're Stressed*

**Three takeaway points:**
1. Phone sensors capture meaningful behavioral patterns tied to stress
2. Machine learning can predict stress levels 2× better than random
3. Passive sensing could revolutionize student mental health monitoring

**Closing line:** "The data your phone already collects might be the key to better mental health — no surveys required."

---

## 🗂️ Figures to Include

All figures are pre-generated in the `reports/figures/` directory:

| Figure | Path | Use on Slide |
|---|---|---|
| Stress Distribution | `reports/figures/ema/01_stress_distribution.png` | Slide 5 |
| Stress Over Time | `reports/figures/ema/02_stress_over_time.png` | Slide 6 |
| Time Patterns | `reports/figures/ema/03_stress_time_patterns.png` | Slide 6 |
| Participant Variability | `reports/figures/ema/04_participant_variability.png` | Backup |
| Missing Data | `reports/figures/ema/05_missing_data.png` | Slide 4 |
| Sleep Analysis | `reports/figures/ema/06_sleep_analysis.png` | Backup |
| EMA Correlation | `reports/figures/ema/07_ema_correlation.png` | Backup |
| Normalization | `reports/figures/ema/08_normalization.png` | Backup |
| Sensor-Stress Correlation | `reports/figures/correlation/01_sensor_stress_correlation.png` | Slide 8 |
| High vs Low Stress | `reports/figures/correlation/02_high_vs_low_stress.png` | Slide 8 |
| Per-Participant | `reports/figures/correlation/03_per_participant_correlation.png` | Backup |
| Feature Importance (Corr) | `reports/figures/correlation/04_feature_importance_correlation.png` | Backup |
| Full Timeline | `reports/figures/deep_dive/01_timeline_u59.png` | Slide 7 |
| Daily Patterns | `reports/figures/deep_dive/02_daily_patterns_u59.png` | Backup |
| Prediction Story | `reports/figures/deep_dive/03_prediction_story_u59.png` | Backup |
| Scatter Plots | `reports/figures/deep_dive/04_scatter_u59.png` | Backup |

---

## 💡 Presentation Tips

1. **Start with the question**, not the data: "Can your phone tell when you're stressed?"
2. **Show the one-student timeline** (Slide 7) — it's the most visual and impactful
3. **Explain accuracy honestly**: 40% on 5-class is good, not magical
4. **Emphasize "no surveys required"** — that's the real innovation
5. **End with real-world impact** — student wellness, mental health

## 🔑 Key Vocabulary to Know

| Term | Simple Explanation |
|---|---|
| EMA | Ecological Momentary Assessment — short surveys sent to your phone |
| Passive sensing | Data collected without you doing anything (phone sensors) |
| Ground truth | The "correct answer" we compare our predictions against |
| Random Forest | A model that uses many decision trees voting together |
| XGBoost | An advanced model that learns from its own mistakes |
| Feature engineering | Creating useful numbers from raw sensor data |
| Correlation | When two things tend to increase/decrease together |
| F1 Score | A balanced measure of prediction quality (0-1, higher is better) |
| Chronological split | Training on past data, testing on future data (no cheating!) |
