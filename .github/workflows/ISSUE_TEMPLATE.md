# Issue Template — MedVision-AI

Thank you for contributing to **MedVision-AI**! Please fill out the relevant section below to help us triage your issue efficiently.

---

## Bug Report

### Description
A clear and concise description of what the bug is.

### Expected Behavior
What you expected to happen.

### Actual Behavior
What actually happened.

### Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Run command '...'
4. See error

### Environment Information
- **OS**: [e.g., Ubuntu 22.04, macOS 14.2]
- **Python version**: [e.g., 3.11.5]
- **MedVision-AI version**: [e.g., v1.2.0, commit SHA]
- **CUDA/cuDNN version**: [e.g., CUDA 12.1, cuDNN 8.9] (if GPU-related)
- **Model**: [e.g., ResNet50, ViT-B/16, BioClinicalBERT]
- **Dataset**: [e.g., CheXpert, MIMIC-CXR, custom]

### Screenshots / Logs
```
Paste relevant logs, error traces, or screenshots here.
```

### Additional Context
Any other context about the problem (e.g., only occurs with specific image formats, certain patient demographics, edge cases in DICOM processing).

---

## Feature Request

### Problem Statement
A clear description of the problem or limitation you are facing. What is the unmet need?

### Proposed Solution
A clear description of the feature or enhancement you'd like to see implemented.

### Alternatives Considered
A description of any alternative solutions or features you've considered.

### Additional Context
- **Use case**: Describe a specific clinical or research scenario where this feature would be valuable.
- **Impact**: How would this feature improve patient outcomes, diagnostic accuracy, or workflow efficiency?
- **References**: Any relevant papers, standards (DICOM, HL7 FHIR), or existing implementations.

---

## Label Guidance

When opening an issue, please apply appropriate labels:

| Label | Description |
|---|---|
| `bug` | Something isn't working as expected |
| `feature` | New functionality or enhancement request |
| `documentation` | Improvements or additions to documentation |
| `model` | Issues related to AI/ML model architecture, training, or inference |
| `data` | Issues related to data loading, preprocessing, or validation |
| `api` | Issues related to the REST API, endpoints, or authentication |
| `security` | Security vulnerabilities or HIPAA compliance concerns |
| `performance` | Performance degradation, memory leaks, or optimization needs |
| `dicom` | DICOM image format handling issues |
| `nlp` | Natural language processing / symptom analysis issues |
| `risk-scoring` | Risk assessment and scoring issues |
| `good first issue` | Welcoming community contributions, beginner-friendly |
| `help wanted` | Extra attention or expertise needed |
| `priority: critical` | Patient safety or data integrity at risk |
| `priority: high` | Significant functionality impaired |
| `priority: medium` | Standard feature request or non-critical bug |
| `priority: low` | Minor improvement or cosmetic issue |

> **⚠️ Note**: Issues involving potential patient data exposure or HIPAA violations should be labeled `security` and `priority: critical`. Do NOT include any Protected Health Information (PHI) in the issue description.
