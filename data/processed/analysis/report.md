# Analysis Report

## Inputs
- prompt_category source: samples.category

## Model Metrics

| model               |   n_essays |   mean_total |   sd_total |   mean_sd |   mean_range |   mean_cv |   icc_total |   icc_n_essays_total |   icc_n_essays_balanced |   icc_n_raters |   auto-approve |   flagged_for_review |
|:--------------------|-----------:|-------------:|-----------:|----------:|-------------:|----------:|------------:|---------------------:|------------------------:|---------------:|---------------:|---------------------:|
| openai/gpt-oss-120b |        130 |      25.4722 |    7.02433 |   1.00361 |       2.4    | 0.0469317 |    0.942911 |                   97 |                      81 |              5 |        55.3846 |              44.6154 |
| qwen/qwen3-32b      |        128 |      28.3168 |    6.00167 |   2.31376 |       5.5625 | 0.0917359 |    0.843858 |                  128 |                     123 |              5 |         7.8125 |              92.1875 |

## Severity Bias (Paired t-test)
- Model 1: openai/gpt-oss-120b
- Model 2: qwen/qwen3-32b
- t=-7.4253, p=0.000000, n_pairs=95

## Topic Bias (Mean Total by Prompt Category)
| model               | prompt_category       |   topic_mean |
|:--------------------|:----------------------|-------------:|
| openai/gpt-oss-120b | data_report           |      24.196  |
| openai/gpt-oss-120b | social_policy_opinion |      26.2176 |
| openai/gpt-oss-120b | tech_society_opinion  |      27.1767 |
| qwen/qwen3-32b      | data_report           |      27.365  |
| qwen/qwen3-32b      | social_policy_opinion |      28.972  |
| qwen/qwen3-32b      | tech_society_opinion  |      28.8464 |

## Category → Question Map
| prompt_category       | questions                                                                                                                                                                                                                                                                                             | visual                      |
|:----------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|
| data_report           | The charts above show the results of a survey of adult education. The first chart shows the reasons why adults decide to study. The pie chart shows how people think the costs of adult education should be shared. Write a report for a university lecturer, describing the information shown above. | ![task_1](image\task_1.png) |
| social_policy_opinion | Some think that governments should support retired people financially while others believe they should take care of themselves. Discuss both views and give your own opinion.                                                                                                                         |                             |
| tech_society_opinion  | Some people think that for robots are very important to you human future development. other think that they are dangerous and have negative effect on society discuss both view and give your opinion.                                                                                                |                             |

## HITL Distribution
| model               | hitl_status        |   count |
|:--------------------|:-------------------|--------:|
| openai/gpt-oss-120b | Auto-Approve       |      72 |
| openai/gpt-oss-120b | Flagged for Review |      58 |
| qwen/qwen3-32b      | Auto-Approve       |      10 |
| qwen/qwen3-32b      | Flagged for Review |     118 |

## Plots
- boxplot: image\boxplot.png
- heatmap_raw: image\heatmap_raw.png
- heatmap_delta: image\heatmap_delta.png
- bland_altman: image\bland_altman.png
- hitl: image\hitl.png

![boxplot](image\boxplot.png)
![heatmap_raw](image\heatmap_raw.png)
![heatmap_delta](image\heatmap_delta.png)
![bland_altman](image\bland_altman.png)
![hitl](image\hitl.png)
