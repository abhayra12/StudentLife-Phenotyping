# StudentLife Paper Notes

## Study Overview
- **Participants**: 48 undergraduate and graduate students at Dartmouth College
- **Duration**: 10 weeks (Spring 2013 term - one complete academic term)
- **Location**: Dartmouth College campus
- **Goal**: Assess the day-to-day and week-by-week impact of workload on stress, sleep, activity, mood, sociability, mental well-being, and academic performance using continuous smartphone sensing

**Study Type**: Longitudinal, in-the-wild deployment with Android phones  
**Data Volume**: 53 GB of continuous sensor data + 32,000+ self-reports + pre/post surveys

---

## Sensors Used

| Sensor Type | Purpose | Sampling Rate/Duty Cycle | What It Captures |
|-------------|---------|--------------------------|------------------|
| **Activity** | Physical activity detection | 1 min ON / 3 min OFF (duty cycling) | Stationary, walking, running (inference every 2-3 sec) |
| **Audio** | Sound environment | 1 min ON / 3 min OFF (stays ON during conversation) | Voice/silence, noise ambient (NOT raw audio - privacy) |
| **Conversation** | Social interaction | Triggered by audio classifier | Conversation start/end timestamps, duration |
| **GPS** | Location tracking | Every 10 minutes | Latitude, longitude, altitude, accuracy |
| **Bluetooth** | Device co-location | Every 10 minutes | Nearby device scans (names removed for privacy) |
| **WiFi** | Indoor location | Frequent scans | WiFi APs detected (SSID removed for privacy) |
| **WiFi Location** | Campus location | Derived from WiFi | Building-level location on campus |
| **Light** | Phone environment | Continuous | Dark periods (≥1 hour) - sleep proxy |
| **Phone Lock** | Phone usage | Event-based | Lock/unlock times, long lock periods (≥1 hour) |
| **Phone Charge** | Charging state | Event-based | Charging periods (≥1 hour) - sleep/home proxy |

**Privacy Measures**:
- Audio features only (no raw audio recorded)
- Bluetooth device names removed
- WiFi SSID removed
- Browser logs excluded
- All data anonymized

---

## Target Variables

### Mental Health Surveys (Pre/Post Term):
- **PHQ-9**: Depression scale (9 questions, score 0-27)
  - Clinical cutoffs: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe
- **UCLA Loneliness Scale**: Social isolation measure
- **Perceived Stress Scale (PSS)**: Stress assessment
- **PANAS**: Positive and Negative Affect Schedule
- **Big Five Personality**: Personality traits
- **Flourishing Scale**: Well-being measure
- **Pittsburgh Sleep Quality Index (PSQI)**
- **VR-12**: Veterans RAND 12 Item Health Survey

### EMAs (Ecological Momentary Assessments) - Daily/Event-Based:
- **Stress** (self-reported stress levels)
- **Mood** (affect ratings)
- **PAM** (Photographic Affect Meter - visual mood scale)
- **Sleep** (quality ratings)
- **Social** (social interactions)
- **Activity, Behavior, Exercise** (various behavioral questions)
- **Events**: Boston Bombing, Cancelled Classes, Green Key (social events), etc.

### Academic Performance:
- **GPA**: Term GPA and cumulative GPA
- **Grades**: Course grades
- **Deadlines**: Number of deadlines per day
- **Piazza Usage**: Online Q&A participation for CS65 class

---

## Key Findings

### 1. **"Dartmouth Term Lifecycle"** - Clear Behavioral Pattern:
- **Week 1-3 (Beginning)**: 
  - High positive affect
  - High conversation levels
  - Low stress
  - Healthy sleep patterns
  - Good activity levels

- **Week 4-7 (Midterm)**: 
  - Stress increases
  - Positive affect declines
  - Sleep duration decreases
  - Conversation frequency drops
  - Activity levels decline

- **Week 8-10 (Finals)**: 
  - Peak stress
  - Minimum sleep
  - Lowest social interaction
  - Reduced physical activity

### 2. **Depression Prediction** (UbiComp 2014):
- Smartphone sensor data significantly correlates with mental health outcomes
- Passive sensing can detect depression symptoms
- Week-by-week behavioral changes predict mental well-being

### 3. **Academic Performance Prediction** (UbiComp 2015):
- Predicted GPA within **±0.179** of reported grades
- Behavioral patterns (sleep, activity, sociability) predict academic success
- Deadline pressure impacts behavior and performance

### 4. **Subsequent Study** (UbiComp 2018 - 83 students over 2 terms):
- **81.5% recall** and **69.1% precision** in predicting depression week-by-week
- Continuous sensing outperforms periodic surveys
- Wearables + phones provide richer signal

---

## Feature Engineering Insights

### Activity Features:
- **Total activity time** (walking + running duration per day)
- **Sedentary time** (percentage of day stationary)
- **Activity transitions** (stationary → walk → run)
- **Activity regularity** (consistency across days)
- **Daily activity patterns** (active hours, peaks)

### Location Features (GPS + WiFi):
- **Location entropy**: Shannon entropy of places visited (mobility diversity)
- **Radius of gyration**: How far students move from their typical center
- **Home stay time**: Time spent at residence hall
- **Number of unique locations**: Location variety
- **Indoor vs outdoor time**: Building vs outdoor mobility
- **Movement regularity**: Consistency of location patterns

### Social Features (Conversation + Bluetooth):
- **Total conversation time per day**
- **Conversation frequency** (number of conversations)
- **Conversation duration** (average, max, min)
- **Time since last conversation** (social isolation indicator)
- **Bluetooth co-location**: Unique devices encountered
- **Social diversity**: Entropy of social contacts
- **Regular vs novel contacts**: Familiar vs new interactions

### Phone Usage Features:
- **Screen time**: Total unlock duration
- **Unlock frequency**: Number of unlocks per day
- **Session duration**: Average screen-on time per unlock
- **Night-time usage**: 3am-6am activity (sleep disruption)
- **Phone checking patterns**: Unlock distribution over day

### Sleep Proxy Features (Dark + Lock + Charge):
- **Bedtime**: When phone becomes dark + locked + charging
- **Wake time**: When phone unlocks in morning
- **Sleep duration**: Estimated sleep hours
- **Sleep regularity**: Standard deviation of bedtime
- **Sleep quality proxy**: Night disruptions, wake-ups

### Temporal Features:
- **Hour of day** (circularly encoded: sin/cos)
- **Day of week** (circularly encoded)
- **Weekend indicator**
- **Days to deadline**: Academic calendar integration
- **Exam period indicator**
- **Term progression**: Week 1-10 indicator

---

## What Worked Well

✅ **Continuous Smartphone Sensing**:
- Successfully captured 10 weeks of data from 48 students
- 53 GB of sensor data collected passively
- High participant compliance

✅ **Duty Cycling Strategy**:
- 1 min ON / 3 min OFF preserved battery life
- Students could use phones normally
- Minimal disruption to daily life

✅ **Multi-Modal Sensing**:
- Combining multiple sensors gave richer behavioral picture
- Cross-validation across sensor types
- Redundancy improved reliability

✅ **Academic Context Integration**:
- Deadlines, GPA, class info provided ground truth
- Term-based study captured natural lifecycle
- Real-world setting (not lab)

✅ **Privacy-Preserving Design**:
- Audio features (no raw audio)
- Anonymized all identifiable data
- Ethical approval obtained

---

## What Didn't Work / Challenges

❌ **Missing Data**:
- Not all sensors worked for all participants
- Battery drain → some students turned off app
- GPS failures indoors
- Bluetooth/WiFi patchy coverage

❌ **Data Quality Variations**:
- Compliance varied across participants
- Some sensors more reliable than others
- EMAs had varying response rates

❌ **Sample Size**:
- 48 participants is modest for ML
- Single term, single university
- Homogeneous population (college students)
- Generalization concerns

❌ **Temporal Coverage**:
- 10 weeks is one term (seasonal effects?)
- No long-term longitudinal data
- Can't capture year-round patterns

❌ **Privacy Constraints**:
- Removed potentially valuable data (browser, detailed WiFi)
- Limited granularity in some features

---

## Limitations & Gaps

### From Original Study:
1. **Limited Sample Size**: 48 students - need more for robust ML
2. **Single Institution**: Dartmouth-specific culture
3. **Short Duration**: 10 weeks - no long-term trends
4. **Demographic Homogeneity**: College students only
5. **Android Only**: No iOS data
6. **Self-Selection Bias**: Volunteers may differ from general population

### Opportunities for Our Project:
1. **Better Feature Engineering**: Use domain knowledge we now have
2. **Modern ML Models**: Deep learning wasn't common in 2014
3. **Time Series Models**: LSTM/GRU for sequential dependencies
4. **Transfer Learning**: Start with their findings, improve
5. **Interpretability**: SHAP for clinical relevance
6. **Cross-Validation**: Time-aware splits

---

## Ideas for Our Project

### Novel Features to Try:
1. **Circadian Rhythm Features**: Sleep phase, consistency, disruptions
2. **Behavioral Entropy**: Predictability of daily routines
3. **Stress Trajectory**: Rate of stress change, not just level
4. **Social Withdrawal Rate**: Decrease in social interaction
5. **Academic Deadline Interaction**: Stress × deadlines feature
6. **Multi-Day Patterns**: Rolling 3-day, 7-day features

### Advanced Modeling:
1. **Sequence Models**: LSTM/GRU for time dependencies
2. **Attention Mechanisms**: Which days/hours matter most?
3. **Multi-Task Learning**: Predict stress + depression + GPA jointly
4. **Ensemble Methods**: XGBoost + LightGBM + Neural Nets
5. **Personalized Models**: Per-student fine-tuning

### Clinical Relevance:
1. **Early Warning System**: Detect mental health deterioration early
2. **Intervention Timing**: When to reach out to students
3. **Explainable Predictions**: Why is someone at risk?
4. **Privacy-Preserving Deployment**: Edge computing on phone

---

## Questions to Research Further

1. **How did they handle missing sensor data?** (Imputation strategy?)
2. **What was the train/test split?** (Participant-level? Time-based?)
3. **Which specific features had highest feature importance?**
4. **What ML models did they use exactly?** (Logistic regression? Random forest?)
5. **How did they validate predictions?** (Cross-validation approach?)
6. **What was the class distribution for depression?** (Imbalanced?)
7. **Did they use any dimensionality reduction?** (PCA? Feature selection?)
8. **How did they deal with time series aspects?** (Lag features? Window aggregation?)
9. **What was the performance on different prediction horizons?** (Next-day? Next-week?)
10. **Were there student subgroups with different patterns?** (Undergrad vs grad? Major?)

---

## Next Steps for Our Project

### Immediate (Phase 2 - EDA):
1. ✅ Download dataset (in progress - 53 GB)
2. Explore sensor data distributions
3. Validate data quality per participant
4. Identify best participants for analysis
5. Create comprehensive data dictionary

### Phase 3 (Preprocessing):
1. Handle missing data strategically
2. Align all sensors to common timebase
3. Create unified feature matrix
4. Train/validation/test split (time-aware)

### Phase 4 (Feature Engineering):
1. Implement all features from paper
2. Add novel features (ideas above)
3. Domain expert validation
4. Feature selection

### Phase 5+ (Modeling):
1. Replicate paper results (baseline)
2. Improve with modern techniques
3. Deploy production API
4. Monitor model performance

---

**This is our foundation! We now understand what data we have, what works, and where we can innovate.** 🚀
