# Fairness Audit Report
**Dataset/Source**: `german_age_bucket_ab_comparison.csv`
**Sensitive Group**: `age_bucket`
**Method**: `fcar`
**Total Instances Evaluated**: 10
**Feasible Recourse Rate**: 100.00%

---
## MISOB Fairness Audit
- **Audit Status**: Fail ❌
- **Audit Score**: **1.000** (0 = Perfect Equity)
- **Maximum Disparity Gap**: nan
- **Most Burdened Group**: ``

### Social Burden by Group Matrix
| Group | Rejection Rate | Avg Cost | Social Burden Score |
|---|---|---|---|

---
## Example Individual Recourse Explanations (Why & How)
### Applicant ID: 162
- **Original Score**: 0.2452
- **Counterfactual Score**: 0.5000
- **Individual Burden**: 0.2801
- **Narrative**: _To overturn the rejection, the applicant must: Decrease duration by 9.00, Decrease credit_amount by 2685.00._

### Applicant ID: 145
- **Original Score**: 0.1456
- **Counterfactual Score**: 0.5337
- **Individual Burden**: 0.0000
- **Narrative**: _To overturn the rejection, the applicant must: ._

### Applicant ID: 4
- **Original Score**: 0.1764
- **Counterfactual Score**: 0.5000
- **Individual Burden**: 0.0771
- **Narrative**: _To overturn the rejection, the applicant must: Decrease credit_amount by 1402.00._

### Applicant ID: 77
- **Original Score**: 0.1480
- **Counterfactual Score**: 0.5000
- **Individual Burden**: 0.1661
- **Narrative**: _To overturn the rejection, the applicant must: Decrease credit_amount by 3019.00._

### Applicant ID: 183
- **Original Score**: 0.4495
- **Counterfactual Score**: 0.5000
- **Individual Burden**: 0.0861
- **Narrative**: _To overturn the rejection, the applicant must: Decrease credit_amount by 1565.00._
