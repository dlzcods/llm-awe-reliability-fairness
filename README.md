# Reliability and Algorithmic Bias Analysis of LLMs in Automated Essay Scoring

This repository contains the research framework for evaluating the reliability and algorithmic bias of Large Language Models (LLMs) within Automated Writing Evaluation (AWE) systems. The study compares two distinct architectures: Qwen3-32B (Dense) and GPT OSS 120B (Mixture-of-Experts) using a multiple-run scoring methodology.

## Research Objectives
- Analyze intrarater reliability of IELTS essay scoring using ICC (Intraclass Correlation Coefficient) and CV (Coefficient of Variation) metrics.
- Detect potential algorithmic bias across 150 essay responses collected from 50 respondents across 3 standardized IELTS prompts.
- Simulate a Human-in-the-Loop (HITL) framework to establish confidence-based routing for automated assessments.

## Key Methodology
1. Rule-Based Topic Categorization: Deterministic classification of essays into categories (Data Report, Social Policy, and Tech & Society) based on IELTS standards and expert judgment.
2. Multiple-Run Scoring: Each essay is evaluated 5 times by both Qwen3-32B and GPT OSS 120B to measure scoring consistency.
3. Statistical Evaluation: Use of Paired T-Tests to validate inter-model severity bias and analysis of variance across topic categories.
4. HITL Simulation: Implementation of decision logic to flag low-confidence or high-variance scores for human review.