# Model Card: Hospital Readmission Prediction

## Model Details

**Model Name:** Hospital Readmission Risk Predictor  
**Model Version:** 1.0.0  
**Model Type:** Random Forest Classifier  
**Date:** November 2024  
**Developers:** Healthcare ML Team  
**License:** MIT  

### Model Description

A multi-class classification model that predicts hospital readmission risk within 30 days. The model categorizes patients into three risk levels:
- **NO**: No readmission expected
- **<30**: High risk of readmission within 30 days
- **>30**: Medium risk of readmission after 30 days

### Architecture

- **Algorithm:** Random Forest with 200 trees
- **Max Depth:** 15
- **Min Samples Split:** 10
- **Class Weighting:** Balanced
- **Feature Count:** 63 engineered features
- **Training Framework:** scikit-learn 1.3.0

---

## Intended Use

### Primary Use Case

Identify hospital patients at high risk of readmission at the time of discharge to:
1. Enable targeted interventions
2. Optimize resource allocation
3. Improve patient outcomes
4. Reduce healthcare costs

### Target Users

- **Clinical Decision Support Systems**: Real-time risk assessment at discharge
- **Care Coordinators**: Prioritizing follow-up scheduling
- **Quality Improvement Teams**: Tracking readmission trends
- **Hospital Administrators**: Resource planning and cost optimization

### Out-of-Scope Uses

❌ **Not intended for:**
- Final clinical decisions without human oversight
- Diagnosis of medical conditions
- Replacement of clinical judgment
- Use on populations significantly different from training data (non-diabetic patients)
- Real-time emergency triage

---

## Training Data

### Dataset

- **Source:** Diabetes 130-US Hospitals (1999-2008)
- **Population:** Diabetic patients from 130 US hospitals
- **Size:** 100,104 patient encounters (post-cleaning)
- **Time Period:** 10 years (1999-2008)
- **Geographic Coverage:** United States

### Data Splits

- **Training:** 70,072 samples (70%)
- **Validation:** 15,016 samples (15%)
- **Test:** 15,016 samples (15%)
- **Split Method:** Stratified by target class

### Demographics

| Feature | Distribution |
|---------|--------------|
| **Age** | Primarily 50-90 years (elderly population) |
| **Gender** | ~53% Female, ~47% Male |
| **Race** | Caucasian (76%), African American (19%), Others (5%) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| NO | 53,205 | 53.2% |
| <30 | 11,356 | 11.3% |
| >30 | 35,543 | 35.5% |

---

## Performance Metrics

### Overall Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 53.21% |
| **F1-Score (Macro)** | 0.443 |
| **F1-Score (Weighted)** | 0.531 |
| **ROC-AUC (OvR)** | 0.652 |
| **Precision (Macro)** | 0.443 |
| **Recall (Macro)** | 0.444 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NO | 0.57 | 0.67 | 0.61 | 7,981 |
| <30 | 0.19 | 0.05 | 0.08 | 1,703 |
| >30 | 0.57 | 0.63 | 0.60 | 5,332 |

### Performance Notes

- **Strengths:** Good performance on NO and >30 classes
- **Limitations:** Low recall (5%) on <30 class due to class imbalance
- **Trade-offs:** Model prioritizes overall accuracy over minority class prediction

### Performance by Demographic Subgroups

**Age Groups:**
- Young (<40): Limited data, lower accuracy
- Middle (40-65): Moderate performance
- Senior (65+): Best performance (most training data)

**Gender:**
- Male: Accuracy ~52%
- Female: Accuracy ~54%
- No significant gender bias detected

**Race:**
- Caucasian: Accuracy ~54% (majority class)
- African American: Accuracy ~51%
- Other races: Insufficient data for reliable estimates

---

## Limitations

### Known Limitations

1. **Class Imbalance**: <30 class (11.3%) is significantly underrepresented
2. **Temporal Drift**: Model trained on 1999-2008 data; healthcare practices have evolved
3. **Population Specificity**: Optimized for diabetic patients only
4. **Geographic Bias**: US hospitals only; may not generalize internationally
5. **Data Recency**: 15+ year old data may not reflect current medical practices

### Technical Constraints

- **Feature Requirements:** All 63 features must be available
- **Missing Data:** Model imputes missing features to 0 (may reduce accuracy)
- **Inference Time:** ~50ms per prediction (suitable for batch, not real-time critical care)
- **Model Size:** ~200MB (requires adequate server resources)

### Known Failure Modes

- **Overfitting to Training Population:** May perform poorly on non-diabetic patients
- **Low Sensitivity for <30 Class:** Misses ~95% of high-risk readmissions
- **Temporal Changes:** Medical coding, treatment protocols evolved since data collection

---

## Ethical Considerations

### Fairness & Bias

**Potential Biases:**
- **Age Bias:** Optimized for elderly (65+) due to training data distribution
- **Socioeconomic Bias:** Hospital data may not capture social determinants of health
- **Historical Bias:** Reflects healthcare disparities from 1999-2008

**Mitigation Strategies:**
1. Monitor performance across demographic subgroups
2. Regular fairness audits using demographic parity metrics
3. Human oversight required for all high-stakes decisions
4. Provide equal access to interventions regardless of predicted risk

### Privacy & Security

- **Data Privacy:** Model does not store patient identifiers
- **HIPAA Compliance:** Deployment must ensure secure data transmission
- **Federated Learning:** Consider for privacy-preserving updates
- **Anonymization:** All training data was de-identified

### Clinical Impact

**Positive Impacts:**
- Early identification of high-risk patients
- Resource optimization for care coordination
- Potential reduction in preventable readmissions

**Negative Risks:**
- False negatives may delay necessary interventions
- Over-reliance on model predictions without clinical context
- Alert fatigue if too many patients flagged as high-risk

---

## Recommendations

### Deployment Best Practices

1. **Human-in-the-Loop:** Require clinician review for all predictions
2. **Monitoring:** Track model performance and drift continuously
3. **Retraining:** Update model annually with recent data
4. **Validation:** Conduct site-specific validation before deployment
5. **Feedback Loop:** Collect outcome data for model improvement

### Intervention Strategies

**For High-Risk Patients (<30):**
- Schedule follow-up within 7 days
- Medication reconciliation at discharge
- Post-discharge phone calls
- Home health services

**For Medium-Risk Patients (>30):**
- Standard discharge planning
- Follow-up within 30 days
- Patient education on warning signs

**For Low-Risk Patients (NO):**
- Routine discharge protocols
- Standard follow-up as needed

---

## Model Interpretability

### Feature Importance (SHAP Analysis)

Top 10 Most Important Features:

1. **number_inpatient** (0.152): Previous inpatient admissions
2. **time_in_hospital** (0.134): Length of current stay
3. **number_diagnoses** (0.118): Complexity of condition
4. **age_numeric** (0.095): Patient age
5. **num_medications** (0.089): Medication count
6. **number_emergency** (0.076): Emergency visits
7. **diabetes_med_count** (0.071): Diabetes medications
8. **discharged_to_home** (0.064): Discharge destination
9. **num_lab_procedures** (0.058): Lab test count
10. **medication_changed** (0.052): Medication adjustments

### Interpretation Guidelines

- Higher previous admissions → Higher readmission risk
- Longer hospital stays → More complex patients
- More diagnoses → Greater comorbidity burden
- Medication changes at discharge → Important signal

---

## Updates & Maintenance

### Version History

- **v1.0.0** (Nov 2024): Initial production model

### Planned Updates

- **Q1 2025:** Retrain with 2020-2024 data
- **Q2 2025:** Add temporal features (day of week, season)
- **Q3 2025:** Implement SMOTE for class balancing
- **Q4 2025:** Evaluate neural network architectures

### Monitoring Plan

**Metrics to Track:**
- Prediction distribution drift
- Per-class performance metrics
- False negative rate for <30 class
- Demographic performance gaps

**Alert Thresholds:**
- Accuracy drop >5%
- F1-score drop >10%
- Significant demographic disparity (>15% gap)

---

## References

### Data Source

Strack, B., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014.

### Model Development

- Scikit-learn: Pedregosa et al., JMLR 2011
- SHAP: Lundberg & Lee, NIPS 2017
- Imbalanced-Learn: Lemaitre et al., JMLR 2017

### Clinical Guidelines

- Centers for Medicare & Medicaid Services (CMS) - Hospital Readmissions Reduction Program
- Society of Hospital Medicine - Care Transitions

---

## Contact

**Model Maintainers:** Healthcare ML Team  
**Email:** ml-team@hospital.org  
**Documentation:** [docs.hospital.org/ml-models]  
**Issue Tracker:** [github.com/hospital/readmission-model/issues]

---

**Last Updated:** November 2024  
**Review Date:** November 2025

