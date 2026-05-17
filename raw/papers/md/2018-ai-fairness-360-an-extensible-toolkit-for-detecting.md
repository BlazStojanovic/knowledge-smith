---
arxiv: '1810.01943'
authors:
- Rachel K. E. Bellamy
- Kuntal Dey
- Michael Hind
- Samuel C. Hoffman
- Stephanie Houde
- Kalapriya Kannan
- Pranay Lohia
- Jacquelyn Martino
- Sameep Mehta
- Aleksandra Mojsilovic
- Seema Nagar
- Karthikeyan Natesan Ramamurthy
- John Richards
- Diptikalyan Saha
- Prasanna Sattigeri
- Moninder Singh
- Kush R. Varshney
- Yunfeng Zhang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and Mitigating
  Unwanted Algorithmic Bias'
url: http://arxiv.org/abs/1810.01943v1
year: 2018
---

# Load the UCI Adult dataset
from aif360.datasets import AdultDataset
ds_orig = AdultDataset()

# Split into train and test partitions
ds_orig_tr, ds_orig_te = ds_orig.split([0.7], shuffle=True, seed=1)

# Look into the training dataset
print("Training Dataset shape")
print(ds_orig_tr.features.shape)
print("Favorable and unfavorable outcome labels")
print(ds_orig_tr.favorable_label, ds_orig_tr.unfavorable_label)
print("Metadata for labels")
print(ds_orig_tr.metadata["label_maps"])
print("Protected attribute names")
print(ds_orig_tr.protected_attribute_names)
print("Privileged and unprivileged protected attribute values")
print(ds_orig_tr.privileged_protected_attributes,
    ds_orig_tr.unprivileged_protected_attributes)
print("Metadata for protected attributes")
print(ds_orig_tr.metadata["protected_attribute_maps"])
```

##### Expected output:

The attributes of the Adult dataset will be printed. The training partition of the Adult dataset has 31655 instances and 98 features with two protected attributes (race and sex). The labels correspond to high-income (>50absent50>50K) or low-income (<=50absent50<=50K), as shown in the metadata. Similar metadata is also available for protected attributes.

### B.2 Checking for bias in the original data

```
# Load the metric class
from aif360.metrics import BinaryLabelDatasetMetric

# Define privileged and unprivileged groups
priv = [{’sex’: 1}] # Male
unpriv = [{’sex’: 0}] # Female

# Create the metric object
metric_otr = BinaryLabelDatasetMetric( ds_orig_tr,
    unprivileged_groups=unpriv, privileged_groups=priv)

# Load and create explainers
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
text_exp_otr = MetricTextExplainer(metric_otr)
json_exp_otr = MetricJSONExplainer(metric_otr)

# Print statistical parity difference
print(text_exp_otr.statistical_parity_difference())
print(json_exp_otr.statistical_parity_difference())
```

##### Expected output:

The statistical parity difference should be −0.19740.1974-0.1974, which is the difference between probability of favorable outcome (high income) between the unprivileged group (females) and the privileged group (male) in this dataset. The JSON output is more elaborate to facilitate consumption by a downstream algorithm.

### B.3 Pre-process data to mitigate bias

```
# Import the reweighing preprocessing algorithm class
from aif360.algorithms.preprocessing.reweighing import Reweighing

# Create the algorithm object
RW = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)

# Train and predict on the training data
# Uses scikit-learn convention (fit, predict, transform)
RW.fit(ds_orig_tr)
ds_transf_tr = RW.transform(ds_orig_tr)
```

##### Expected output:

There will be no output here, but the reweighing algorithm equalizes the weights across (group, label) combination.

### B.4 Checking for bias in the pre-processed training data

```
# Create the metric object for pre-processed data
metric_ttr = BinaryLabelDatasetMetric(ds_transf_tr,
    unprivileged_groups=unpriv, privileged_groups=priv)

# Create explainer
text_exp_ttr = MetricTextExplainer(metric_ttr)

# Print statistical parity difference
print(text_exp_ttr.statistical_parity_difference())
```

##### Expected output:

Because of the action of the re-weighing pre-processing algorithm, the statistical parity difference for the transformed data (ds\_transf\_tr) must be really close to 0.

### B.5 Pre-process out-of-sample testing data and check for bias

```
# Apply the learned re-weighing pre-processor
ds_transf_te = RW.transform(ds_orig_te)

# Create metric objects for original and
# pre-processed test data
metric_ote = BinaryLabelDatasetMetric(ds_orig_te,
    unprivileged_groups=unpriv, privileged_groups=priv)
metric_tte = BinaryLabelDatasetMetric(ds_transf_te,
    unprivileged_groups=unpriv, privileged_groups=priv)

# Create explainers for both metric objects
text_exp_ote = MetricTextExplainer(metric_ote)
text_exp_tte = MetricTextExplainer(metric_tte)

# Print statistical parity difference
print(text_exp_ote.statistical_parity_difference())
print(text_exp_tte.statistical_parity_difference())
```

##### Expected output:

The trained re-weighing pre-processor can be applied on the out-of-sample test data. The metrics for the original and transformed testing data will show a significant reduction in statistical parity difference (-0.2021 to -0.0119 in this case).

## Appendix C Additional Experimental Details

We provide additional details on the experimental evaluations.

### C.1 Datasets

#### C.1.1 Adult Census Income

For protected attribute sex, Male is privileged, and Female is unprivileged. For protected attribute race, White is privileged, and Non-white is unprivileged. Favorable label is High income (>50absent50>50K) and unfavorable label is Low income (<=50absent50<=50K).

#### C.1.2 German Credit

For protected attribute sex, Male is privileged, and Female is unprivileged. For protected attribute age, Old is privileged, and Young is unprivileged. Favorable label is Good credit and unfavorable label is Bad credit.

#### C.1.3 Probpublica recidivism (COMPAS)

For protected attribute sex, Female is privileged, and Male is unprivileged. For protected attribute race, Caucasian is privileged, and Not Caucasian is unprivileged. Favorable label is Did not recidivate and unfavorable label is Did recidivate.

### C.2 Metrics

#### C.2.1 Statistical Parity Difference

This is the difference in the probability of favorable outcomes between the unprivileged and privileged groups. This can be computed both from the input dataset as well as from the dataset output from a classifier (predicted dataset). A value of 00 implies both groups have equal benefit, a value less than 00 implies higher benefit for the privileged group, and a value greater than 00 implies higher benefit for the unprivileged group.

#### C.2.2 Disparate Impact

This is the ratio in the probability of favorable outcomes between the unprivileged and privileged groups. This can be computed both from the input dataset as well as from the dataset output from a classifier (predicted dataset). A value of 111 implies both groups have equal benefit, a value less than 111 implies higher benefit for the privileged group, and a value greater than 111 implies higher benefit for the unprivileged group.

#### C.2.3 Average odds difference

This is the average of difference in false positive rates and true positive rates between unprivileged and privileged groups. This is a method in the ClassificationMetric class and hence needs to be computed using the input and output datasets to a classifier. A value of 00 implies both groups have equal benefit, a value less than 00 implies higher benefit for the privileged group and a value greater than 00 implies higher benefit for the unprivileged group.

#### C.2.4 Equal opportunity difference

This is the difference in true positive rates between unprivileged and privileged groups. This is a method in the ClassificationMetric class and hence needs to be computed using the input and output datasets to a classifier. A value of 00 implies both groups have equal benefit, a value less than 00 implies higher benefit for the privileged group and a value greater than 00 implies higher benefit for the unprivileged group.

## Appendix D Evaluation on different data sets

We present additional results with bias mitigation obtained for various datasets and protected attributes. These correspond to the setting described in Section [10](#S10 "10 Evaluation of the Algorithms").

!(/html/1810.01943/assets/legend_hor.png)

!(/html/1810.01943/assets/adult_sex_statistical_parity_difference_before.png)

!(/html/1810.01943/assets/adult_sex_statistical_parity_difference_after.png)

(a) Statistical parity difference

!(/html/1810.01943/assets/adult_sex_disparate_impact_before.png)

!(/html/1810.01943/assets/adult_sex_disparate_impact_after.png)

(b) Disparate impact

!(/html/1810.01943/assets/adult_sex_average_odds_difference_before.png)

!(/html/1810.01943/assets/adult_sex_average_odds_difference_after.png)

(c) Average odds difference

!(/html/1810.01943/assets/adult_sex_equal_opportunity_difference_before.png)

!(/html/1810.01943/assets/adult_sex_equal_opportunity_difference_after.png)

(d) Equal opportunity difference

Figure 8: Fairness vs. Balanced Accuracy before (top panel) and after (bottom panel) applying various bias mitigation algorithms. Four different fairness metrics are shown. In most cases two classifiers (Logistic regression - LR or Random forest classifier - RF) were used. The ideal fair value of disparate impact is 1, whereas for all other metrics it is 0. The circles indicate the mean value and bars indicate the extent of ±plus-or-minus\pm1 standard deviation. Data set: Adult, Protected attribute: sex.

!(/html/1810.01943/assets/legend_hor.png)

!(/html/1810.01943/assets/german_sex_statistical_parity_difference_before.png)

!(/html/1810.01943/assets/german_sex_statistical_parity_difference_after.png)

(a) Statistical parity difference

!(/html/1810.01943/assets/german_sex_disparate_impact_before.png)

!(/html/1810.01943/assets/german_sex_disparate_impact_after.png)

(b) Disparate impact

!(/html/1810.01943/assets/german_sex_average_odds_difference_before.png)

!(/html/1810.01943/assets/german_sex_average_odds_difference_after.png)

(c) Average odds difference

!(/html/1810.01943/assets/german_sex_equal_opportunity_difference_before.png)

!(/html/1810.01943/assets/german_sex_equal_opportunity_difference_after.png)

(d) Equal opportunity difference

Figure 9: Fairness vs. Balanced Accuracy before (top panel) and after (bottom panel) applying various bias mitigation algorithms. Four different fairness metrics are shown. In most cases two classifiers (Logistic regression - LR or Random forest classifier - RF) were used. The ideal fair value of disparate impact is 1, whereas for all other metrics it is 0. The circles indicate the mean value and bars indicate the extent of ±plus-or-minus\pm1 standard deviation. Data set: german, Protected attribute: sex.

!(/html/1810.01943/assets/legend_hor.png)

!(/html/1810.01943/assets/german_age_statistical_parity_difference_before.png)

!(/html/1810.01943/assets/german_age_statistical_parity_difference_after.png)

(a) Statistical parity difference

!(/html/1810.01943/assets/german_age_disparate_impact_before.png)

!(/html/1810.01943/assets/german_age_disparate_impact_after.png)

(b) Disparate impact

!(/html/1810.01943/assets/german_age_average_odds_difference_before.png)

!(/html/1810.01943/assets/german_age_average_odds_difference_after.png)

(c) Average odds difference

!(/html/1810.01943/assets/german_age_equal_opportunity_difference_before.png)

!(/html/1810.01943/assets/german_age_equal_opportunity_difference_after.png)

(d) Equal opportunity difference

Figure 10: Fairness vs. Balanced Accuracy before (top panel) and after (bottom panel) applying various bias mitigation algorithms. Four different fairness metrics are shown. In most cases two classifiers (Logistic regression - LR or Random forest classifier - RF) were used. The ideal fair value of disparate impact is 1, whereas for all other metrics it is 0. The circles indicate the mean value and bars indicate the extent of ±plus-or-minus\pm1 standard deviation. Data set: german, Protected attribute: age.

!(/html/1810.01943/assets/legend_hor.png)

!(/html/1810.01943/assets/compas_sex_statistical_parity_difference_before.png)

!(/html/1810.01943/assets/compas_sex_statistical_parity_difference_after.png)

(a) Statistical parity difference

!(/html/1810.01943/assets/compas_sex_disparate_impact_before.png)

!(/html/1810.01943/assets/compas_sex_disparate_impact_after.png)

(b) Disparate impact

!(/html/1810.01943/assets/compas_sex_average_odds_difference_before.png)

!(/html/1810.01943/assets/compas_sex_average_odds_difference_after.png)

(c) Average odds difference

!(/html/1810.01943/assets/compas_sex_equal_opportunity_difference_before.png)

!(/html/1810.01943/assets/compas_sex_equal_opportunity_difference_after.png)

(d) Equal opportunity difference

Figure 11: Fairness vs. Balanced Accuracy before (top panel) and after (bottom panel) applying various bias mitigation algorithms. Four different fairness metrics are shown. In most cases two classifiers (Logistic regression - LR or Random forest classifier - RF) were used. The ideal fair value of disparate impact is 1, whereas for all other metrics it is 0. The circles indicate the mean value and bars indicate the extent of ±plus-or-minus\pm1 standard deviation. Data set: compas, Protected attribute: sex.

!(/html/1810.01943/assets/legend_hor.png)

!(/html/1810.01943/assets/compas_race_statistical_parity_difference_before.png)

!(/html/1810.01943/assets/compas_race_statistical_parity_difference_after.png)

(a) Statistical parity difference

!(/html/1810.01943/assets/compas_race_disparate_impact_before.png)

!(/html/1810.01943/assets/compas_race_disparate_impact_after.png)

(b) Disparate impact

!(/html/1810.01943/assets/compas_race_average_odds_difference_before.png)

!(/html/1810.01943/assets/compas_race_average_odds_difference_after.png)

(c) Average odds difference

!(/html/1810.01943/assets/compas_race_equal_opportunity_difference_before.png)

!(/html/1810.01943/assets/compas_race_equal_opportunity_difference_after.png)

(d) Equal opportunity difference

Figure 12: Fairness vs. Balanced Accuracy before (top panel) and after (bottom panel) applying various bias mitigation algorithms. Four different fairness metrics are shown. In most cases two classifiers (Logistic regression - LR or Random forest classifier - RF) were used. The ideal fair value of disparate impact is 1, whereas for all other metrics it is 0. The circles indicate the mean value and bars indicate the extent of ±plus-or-minus\pm1 standard deviation. Data set: compas, Protected attribute: race.

## Appendix E UI page

!(/html/1810.01943/assets/AIFairness360_PosterPage.png)

Figure 13: A screen shot from the web interactive experience, showing the results of mitigation applied to one of the available datasets.
