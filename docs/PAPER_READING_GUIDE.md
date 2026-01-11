# 📚 Task 1: Reading the StudentLife Research Paper

## 🎯 Objective
Understand the research foundation of the StudentLife dataset - what they measured, how they measured it, and what they discovered. This will guide your entire ML project.

## 📄 Paper Information

**Title**: "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones"

**Authors**: Rui Wang, Fanglin Chen, Zhenyu Chen, Tianxing Li, Gabriella Harari, Stefanie Tignor, Xia Zhou, Dror Ben-Zeev, Andrew T. Campbell

**Published**: 2014, UbiComp (ACM Conference on Ubiquitous Computing)

**Access**: https://studentlife.cs.dartmouth.edu/

---

## 📖 Reading Guide

### Section 1: Introduction & Motivation (Pages 1-2)

**Key Questions to Answer**:
1. Why is smartphone sensing useful for mental health assessment?
2. What are the limitations of traditional assessment methods (surveys)?
3. What is "digital phenotyping"? (They might not use this exact term)
4. What was the gap in existing research they aimed to fill?

**What to Note**:
- The problem they're solving
- Why this matters for college students specifically
- What makes their approach novel

---

### Section 2: Related Work (Usually Pages 2-3)

**Skim This** - Not critical for implementation, but note:
- What other smartphone sensing studies existed
- What sensor types are commonly used
- Any limitations they reference

---

### Section 3: Study Design (CRITICAL - Read Carefully)

**Key Questions**:
1. How many participants? (Demographics?)
2. How long was the study? (Number of weeks)
3. What sensors were used? List them all:
   - Passive sensors: ___________
   - Active surveys: ___________
   - Assessments: ___________

4. What were they trying to predict?
   - Mental health: ___________
   - Academic: ___________
   - Behavioral: ___________

5. How was data collected? (Always on? Sampling rate? Privacy measures?)

**Create This Table While Reading**:

```
| Sensor Type | Sampling Rate | What It Measures | Privacy Handling |
|-------------|---------------|------------------|------------------|
| Activity    |               |                  |                  |
| Location    |               |                  |                  |
| Audio       |               |                  |                  |
| ...         |               |                  |                  |
```

---

### Section 4: Data Collection & Dataset

**Key Questions**:
1. What is the data format?
2. How much data per student?
3. What was participation/compliance rate?
4. Any data quality issues mentioned?

**Important for Your Project**:
- Missing data patterns
- Sensor failures or battery issues
- Participant dropout

---

### Section 5: Features & Methodology

**VERY IMPORTANT** - This guides your Phase 4 (Feature Engineering)

**For Each Sensor, Note**:
1. **Activity**:
   - What features did they engineer? (e.g., total active time, sedentary time)
   - Aggregation windows? (hourly, daily, weekly)

2. **Location**:
   - How did they quantify mobility? (entropy, radius of gyration)
   - Home detection method?

3. **Social (Conversation, Bluetooth)**:
   - Conversation detection approach
   - Co-location features

4. **Phone Usage**:
   - Screen time calculation
   - Sleep proxy (dark + charging + locked)

5. **Temporal**:
   - Time of day effects
   - Day of week patterns
   - Academic calendar integration

**Create Feature List** as you read - you'll implement many of these!

---

### Section 6: Results & Analysis

**Key Questions**:
1. What were the main findings?
   - Which sensors correlated with depression?
   - Which sensors predicted academic performance?
   - Any surprising results?

2. What was the prediction accuracy?
   - Classification metrics?
   - Which models did they use?

3. Feature importance:
   - Which features were most predictive?
   - Any features that didn't work?

**Your Takeaway**: What worked and what didn't - guides your modeling

---

### Section 7: Discussion & Limitations

**Critical for Your Project**:
1. What limitations did they acknowledge?
   - Sample size issues?
   - Generalization concerns?
   - Technical limitations?

2. What would they do differently?
3. Future work suggestions

**Your Advantage**: You have their dataset and findings - can build on their work!

---

## 📝 While Reading: Create `docs/paper_notes.md`

Use this template:

```markdown
# StudentLife Paper Notes

## Study Overview
- **Participants**: 
- **Duration**: 
- **Location**: 
- **Goal**: 

## Sensors Used
| Sensor | Purpose | Sampling | Key Features |
|--------|---------|----------|--------------|
|        |         |          |              |

## Target Variables
- **Mental Health**: PHQ-9 (depression), ___
- **Academic**: GPA, ___
- **Behavioral**: ___

## Key Findings
1. 
2. 
3. 

## Feature Engineering Insights
- Activity features: 
- Location features: 
- Social features: 
- Phone usage features: 

## What Worked Well
- 
- 

## What Didn't Work
- 
- 

## Limitations & Gaps
- 
- 

## Ideas for Our Project
- 
- 

## Questions to Research Further
- 
- 
```

---

## 🎯 Learning Objectives

After reading, you should be able to answer:

1. **What sensors** does the StudentLife dataset contain?
2. **What are we trying to predict** with this data?
3. **What features** are typically engineered from smartphone sensors?
4. **What worked** in the original study (so you can replicate it)?
5. **What didn't work** (so you can improve upon it)?
6. **What ethical/privacy concerns** exist with this type of data?

---

## ⏱️ Time Estimate

- **Quick read** (skim): 30 minutes
- **Thorough read** (recommended): 60-90 minutes
- **Deep read with notes**: 2 hours

**Recommendation**: Spend 60-90 minutes, take notes, create the feature table.

---

## 🚀 Ready to Read?

1. **Access the paper**: https://studentlife.cs.dartmouth.edu/
2. **Open the notes template**: Create `docs/paper_notes.md`
3. **Read systematically**, section by section
4. **Take notes** as you go
5. **Come back with questions** - I'll help clarify!

**Let me know when you're done or if you have questions while reading!** 📚
