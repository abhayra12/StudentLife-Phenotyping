# 📚 StudentLife Dataset Reference Guide

## 1. Dataset Overview
**Source**: Dartmouth College (StudentLife Study)  
**Participants**: 48 students (30 undergraduate, 18 graduate)  
**Duration**: 10 weeks (Spring Term 2013)  
**Objective**: Assess the impact of workload on student stress, sleep, activity, mood, and academic performance.

> **Key Characteristic**: This is a **longitudinal, multi-modal** dataset combining continuous passive smartphone sensing with extensive active self-reporting.

---

## 2. Passive Sensor Data (Automatic)
Collected continuously via the StudentLife Android app.

### 📱 Physical Sensors
| Sensor | Filename | Frequency / Duty Cycle | Description |
|:---|:---|:---|:---|
| **Accelerometer** | `activity.csv` | 1 min ON / 3 min OFF | Inferred activity state: `Stationary`, `Walking`, `Running`, `Unknown`. |
| **Audio** | `audio.csv` | 1 min ON / 3 min OFF | Privacy-preserving features. Inferred state: `Silence`, `Voice`, `Noise`, `Unknown`. |
| **GPS** | `gps.csv` | Every 10 mins | Latitude, Longitude, Altitude, Bearing, Accuracy, Speed. |
| **WiFi** | `wifi.csv` | Frequent scans | Scanned WiFi Access Points (hashed SSIDs). Used for indoor location. |
| **Bluetooth** | `bt.csv` | Every 10 mins | Nearby devices (hashed MAC addresses). Used for social proximity. |
| **Light** | `light.csv` | Continuous | Ambient light levels (lux). Used to infer sleep/environment. |

### 🔋 Phone State & Usage
| Feature | Filename | Description |
|:---|:---|:---|
| **Phone Lock** | `phonelock.csv` | Timestamps of screen lock/unlock events. Proxy for screen time & interaction. |
| **Phone Charge** | `phonecharge.csv` | Timestamps of charging start/end. Proxy for being at home/desk. |
| **Conversation** | `conversation.csv` | Start/End times of detected conversations (inferred from Audio). |
| **Darkness** | `dark.csv` | Intervals where phone was in a dark environment (t < 10 lux). |

---

## 3. Active Self-Report Data (EMAs)
Ecological Momentary Assessments (EMAs) prompted via the app multiple times per day.

### 🧠 Daily/Event-Based EMAs
| Category | Metric | Scale/Format |
|:---|:---|:---|
| **Stress** | Perceived Stress | Likert scale (1-5 or similar) |
| **Mood** | Emotional State | PAM (Photographic Affect Meter) - 4x4 grid of images |
| **Sleep** | Sleep Quality | Rating + Duration |
| **Social** | Interactions | Number of people, duration |
| **Activity** | Behavior | "What are you doing?" (Class, Lab, Party, etc.) |

---

## 4. Pre/Post-Study Surveys
Standardized psychological instruments administered at the start and end of the term.

| Instrument | Measures | Clinical Relevance |
|:---|:---|:---|
| **PHQ-9** | Depression Severity | Score 0-27. (≥10 indicates moderate depression) |
| **PSS** | Perceived Stress Scale | General stress levels. |
| **UCLA Loneliness** | Social Isolation | Subjective feelings of loneliness. |
| **PANAS** | Affect (Positive/Negative) | Emotional well-being. |
| **Big Five** | Personality Traits | Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. |
| **VR-12** | Health Status | Physical and mental health summary. |
| **Flourishing** | Psychological Well-being | Self-perceived success/flourishing. |

---

## 5. Academic & Contextual Data
Data provided by the university registrar or scraped from learning management systems.

| Data Point | Description | Utility |
|:---|:---|:---|
| **GPA** | Term GPA and Cumulative GPA | Primary academic performance target. |
| **Course Grades** | Individual class grades | Granular performance metric. |
| **Deadlines** | Date/Time of assignments/exams | Context for stress spikes. |
| **Class Schedule** | Times/Locations of classes | Ground truth for "in class" behavior. |
| **Piazza Usage** | Online Q&A forum activity | Number of posts, views, answers (Engagement proxy). |

---

## 6. Data Structure Notes
- **Timestamps**: All timestamps are **Unix Epoch** (seconds since Jan 1, 1970).
- **Timezone**: Data is in **Eastern Standard Time (EST)** (Dartmouth College).
- **Anonymization**:
    - Participant IDs are randomized (e.g., `u00`, `u01`).
    - WiFi SSIDs and Bluetooth MACs are hashed.
    - No raw audio or text content is preserved.

## 7. Key Research Findings (Context)
- **Dartmouth Term Lifecycle**: Stress rises, sleep/activity fall as term progresses.
- **Correlation**: Passive sensing (sleep, conversation, mobility) strongly correlates with PHQ-9 and GPA.
- **Prediction**: It is possible to predict GPA and depression from sensor data alone.
