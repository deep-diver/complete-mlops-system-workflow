# Complete MLOps System Workflow with TFX

This repository consists of two main MLOps workflows in the context of image classification task. 
1. Adapting to changes in codebase
2. Adapting to changes in data

```mermaid
  flowchart LR;
      subgraph codebase
      direction LR
      A[Codebase Changes]-->B[Unit Test];
      B--Pass-->C[Image Build&Push];
      end
      
      subgraph data
      direction LR
      E[Data Collection]--Enough Data-->F[Batch Inference];
      F-->G[Performance Evaluation];
      end
      
      subgraph pipeline
      direction TB
      D[Pipeline Trigger]-->H[Data Injection];
      H-->I[Data Validation];
      I-->J[Model Training];
      J-->K[Model Evaluation];
      K-->L[Deployment];
      end
      
      codebase-->pipeline;
      data--Under Threshold-->pipeline;
```
