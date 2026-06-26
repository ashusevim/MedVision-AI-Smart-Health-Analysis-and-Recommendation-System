# System Overview

## What is the Artificial Cognitive Brain?

The Artificial Cognitive Brain (ACB) is a modular, biologically-inspired AI framework designed to emulate the core cognitive capabilities of the human mind. Rather than functioning as a single monolithic model, ACB decomposes intelligence into five cooperating subsystems — **perception**, **memory**, **reasoning**, **learning**, and **planning** — each implemented as an independent microservice that communicates through a shared global workspace. The system is purpose-built for agentic applications requiring long-horizon reasoning, declarative and episodic recall, and adaptive planning under uncertainty.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Artificial Cognitive Brain                 │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Perception│  │  Memory  │  │Reasoning │  │ Planning │  │
│  │  Module  │  │  Module  │  │  Module  │  │  Module  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │             │          │
│       └─────────────┴──────┬──────┴─────────────┘          │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │Global Workspace │  ◄── Broadcast Bus    │
│                   └────────┬────────┘                       │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │  Learning Hub   │                       │
│                   └─────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        │                                    │
   ┌────▼────┐                          ┌────▼────┐
   │ Sensors │                          │ Actuators│
   │ / Input │                          │ / Output │
   └─────────┘                          └──────────┘
```

## Core Modules

| Module        | Responsibility                                                        |
|---------------|-----------------------------------------------------------------------|
| **Perception**| Ingests raw multi-modal input (text, image, audio, structured data), normalises encodings, and produces compact semantic embeddings. |
| **Memory**    | Manages short-term working memory, long-term declarative (fact) stores, episodic (experience) stores, and procedural (skill) memory. Implements fast retrieval via approximate nearest-neighbour indexes. |
| **Reasoning**  | Executes chains of logical inference, abductive hypothesis generation, causal analysis, and self-consistency verification over the contents of the global workspace. |
| **Learning**  | Consolidates experiences into memory, performs continual / online learning with replay buffers, and updates internal world models through prediction-error feedback. |
| **Planning**  | Constructs multi-step action plans using Monte Carlo Tree Search or hierarchical task networks, evaluates expected outcomes, and manages execution rollback on failure. |

## Design Principles

1. **Modularity & Loose Coupling** — Each cognitive module owns its data model, lifecycle, and scaling policy. Inter-module communication is message-driven through the Global Workspace bus.
2. **Biological Plausibility** — Design choices are informed by cognitive science (Global Workspace Theory, ACT-R, complementary learning systems) to ensure emergent capabilities map to interpretable mental processes.
3. **Continual Learning** — The system learns incrementally from experience without catastrophic forgetting, leveraging experience replay and synaptic consolidation strategies.
4. **Explainability** — Every reasoning trace, memory retrieval, and planning decision is logged as a structured provenance chain, enabling full auditability.
5. **Scalability** — Individual modules can be horizontally scaled and deployed on heterogeneous hardware (CPU for memory, GPU for perception and reasoning, TPU for learning).

## Technology Stack

| Layer              | Technologies                                          |
|--------------------|-------------------------------------------------------|
| Core Language      | Python 3.11+                                          |
| ML Framework       | PyTorch 2.x / JAX                                     |
| Message Bus        | Apache Kafka / Redis Streams                          |
| Vector Store       | Qdrant / Milvus                                       |
| Orchestration      | Kubernetes, Helm Charts                                |
| Serving            | Triton Inference Server, vLLM                         |
| Monitoring         | Prometheus, Grafana, OpenTelemetry                   |
| CI/CD              | GitHub Actions, Dagger                                 |

## Requirements

- **Hardware**: Minimum 8-core CPU, 64 GB RAM, 1× NVIDIA A100 (40 GB) for development. Production clusters should provision per-module GPU quotas.
- **Software**: Docker 24+, Kubernetes 1.28+, Python 3.11+, CUDA 12.x.
- **Network**: Low-latency intra-cluster mesh (< 1 ms P99) for Global Workspace broadcast.
- **Data**: Initial knowledge base in a supported embedding format; optional pre-trained perception models.
