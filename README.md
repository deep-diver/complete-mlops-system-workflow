# Complete MLOps System Workflow with TFX

This repository consists of two main MLOps workflows in the context of image classification task. 
1. Adapting to changes in codebase
2. Adapting to changes in data

```mermaid
  graph LR;
      A[Codebase Changes]-->B[Unit Test];
      B--Pass-->C[Image Build&Push];
      C-->D[Trigger TFX Pipeline];
      
      E[Data Collection]--Enough Data-->F[Batch Inference];
      F-->G[Performance Evaluation];
      G--Under Threshold-->D;
```
