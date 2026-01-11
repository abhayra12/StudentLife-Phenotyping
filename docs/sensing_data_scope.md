# Sensing Data Scope Documentation

**Date**: 2026-01-11  
**Decision**: Use ONLY sensing data from StudentLife dataset

---

## 🎯 Scope Decision

This project will use **ONLY** the `data/raw/dataset/sensing/` folder from the StudentLife dataset.

**Excluded Data**:
- ❌ EMA responses (self-reported stress, mood, sleep quality, social ratings)
- ❌ Survey data (PHQ-9 depression scores, personality tests, loneliness scales)
- ❌ Education data (GPA, course grades, Piazza Q&A usage)

---

## 📊 Available Sensing Data (10 Types)

### 1. **activity/** - Physical Activity
- Accelerometer-based activity classification
- Categories: Stationary, Walking, Running
- Sampling: 1 min ON / 3 min OFF (duty cycling)

### 2. **audio/** - Audio Environment
- Audio volume features (NOT raw audio - privacy preserved)
- Voice/silence classification
- Ambient noise levels

### 3. **bluetooth/** - Device Co-location
- Nearby Bluetooth device scans
- Device encounter frequency
- Social proximity inference (device names removed for privacy)

### 4. **conversation/** - Conversation Detection
- Binary: Conversation yes/no
- Conversation duration
- Triggered by audio classifier

### 5. **dark/** - Screen Dark Periods
- Dark periods ≥1 hour
- Sleep proxy indicator
- Screen-off time tracking

### 6. **gps/** - GPS Location
- Latitude, longitude, altitude
- Sampling: Every 10 minutes
- Accuracy metrics included

### 7. **phonecharge/** - Charging Events
- Charging periods ≥1 hour
- Home/sleep proxy
- Battery state tracking

### 8. **phonelock/** - Lock/Unlock Events  
- Phone lock/unlock timestamps
- Screen-on duration
- Long lock periods (≥1 hour)

### 9. **wifi/** - WiFi Access Points
- WiFi AP detection scans
- SSID removed for privacy
- Indoor location proxy

### 10. **wifi_location/** - Campus Location
- Building-level location on Dartmouth campus
- Derived from WiFi signatures
- Class/library/dorm/dining detection

---

## ✅ What We CAN Do (Sensing-Only)

### Behavioral Analytics:
- ✅ **Activity Patterns**: Daily activity levels, sedentary time, regularity
- ✅ **Sleep Analysis**: Inferred bedtime/wake time, sleep duration, sleep regularity
- ✅ **Mobility Analysis**: Location entropy, radius of gyration, home stay time
- ✅ **Social Engagement**: Conversation frequency/duration, co-location patterns
- ✅ **Phone Usage**: Screen time, unlock frequency, night disruptions
- ✅ **Circadian Rhythms**: Hour-of-day activity patterns, behavioral regularity

### Machine Learning Tasks:
- ✅ **Unsupervised Learning**:
  - Behavioral clustering (group similar students)
  - Anomaly detection (unusual behavior changes)
  - Pattern mining (recurring behavioral motifs)
  
- ✅ **Self-Supervised Learning**:
  - Next-day behavior prediction (predict tomorrow from today)
  - Time series forecasting
  - Behavior autoencoding

- ✅ **Descriptive Analytics**:
  - Term lifecycle trends (how behavior changes over 10 weeks)
  - Weekday vs weekend patterns
  - Individual vs population baselines

### Feature Engineering:
- ✅ Comprehensive digital phenotyping features
- ✅ Multi-sensor fusion features
- ✅ Temporal aggregation (hourly, daily, weekly)
- ✅ Behavioral regularity metrics

---

## ❌ What We CANNOT Do (Without Survey/EMA Data)

### Direct Supervised Prediction:
- ❌ **Depression prediction** (no PHQ-9 scores)
- ❌ **Stress level prediction** (no self-reported stress)
- ❌ **GPA prediction** (no academic performance data)
- ❌ **Mood forecasting** (no mood ratings)

### Validation Against Ground Truth:
- ❌ Cannot directly validate behavioral patterns against mental health outcomes
- ❌ Cannot compute correlation with depression severity
- ❌ Cannot measure prediction accuracy for clinical outcomes

---

## 🎯 Project Impact

### Advantages of Sensing-Only Approach:
1. **Objective Data**: Avoids self-report bias
2. **Passive Collection**: No user burden
3. **Continuous Monitoring**: 24/7 data vs periodic surveys
4. **Scalable**: No manual annotations needed
5. **Privacy-Preserving**: No sensitive self-disclosures

### Limitations:
1. **No Direct Labels**: Cannot validate behavioral inferences
2. **Interpretation Uncertainty**: Behavior patterns correlate with, but don't prove mental health states
3. **No Clinical Validation**: Cannot claim clinical utility without ground truth

### Use Cases:
1. **Research Foundation**: Build comprehensive behavioral feature library
2. **Pattern Discovery**: Identify behavioral markers worth investigating
3. **Proof of Concept**: Demonstrate value of passive sensing
4. **Future Supervised Work**: Features ready when labels become available

---

## 🔮 Future Directions

If we later gain access to survey/EMA data, we can:
- Use our extracted features for supervised learning
- Validate behavioral clusters against depression scores
- Build clinical prediction models
- Compare passive sensing vs self-report accuracy

For now, we focus on:
- Building robust behavioral analytics
- Extracting interpretable features
- Demonstrating patterns over the academic term
- Creating production-ready sensing pipeline

---

## 📚 References

**Paper**: Wang, R., et al. (2014). "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones"

**Dataset**: https://studentlife.cs.dartmouth.edu/dataset.html

---

**Last Updated**: 2026-01-11 22:20
