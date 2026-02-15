# AAIPL Hackathon - Dual Agent Q&A System

[![IIT Delhi](https://img.shields.io/badge/IIT-Delhi-blue)](https://home.iitd.ac.in/)
[![Qwen 2.5](https://img.shields.io/badge/Model-Qwen2.5--14B-green)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![Unsloth](https://img.shields.io/badge/Framework-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![ROCm](https://img.shields.io/badge/GPU-AMD%20MI300X-red)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

A dual-agent system for generating and answering difficult MCQ questions in **Quantitative Aptitude** and **Analytical Reasoning** domains. Built for the AAIPL (Artificial AI Programming League) Hackathon at IIT Delhi.

## Topics Covered

| Domain | Topics |
|--------|--------|
| **Logical Reasoning** | Syllogisms |
| **Puzzles** | Seating Arrangements (Linear, Circular) |
| **Blood Relations** | Family Tree Logic |
| **Series & Patterns** | Mixed Series (Alphanumeric) |


## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen2.5-14B-Instruct |
| **LoRA Rank (r)** | 64 |
| **LoRA Alpha** | 64 |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 4 |
| **Gradient Accumulation** | 4 steps |
| **Epochs** | 3 |
| **Precision** | BFloat16 |


## Data Sources

Training data curated from 14 PDF sources across 4 providers:

| Topic | Sources |
|-------|---------|
| **Syllogisms** | Byjus, BankExamsToday, CareerMate |
| **Seating Arrangement** | CareerMate, SPLessons, ReferenceGlobe, PracticeMock, SmartKeeda |
| **Blood Relations** | Byjus, SmartKeeda, TopCoaching |
| **Alphanumeric Series** | PracticeMock, ExamPundit, Byjus |

## Output Format

### Question Agent Output

```json
{
  "topic": "Syllogisms",
  "question": "All roses are flowers. Some flowers are red. Which conclusion follows?",
  "choices": [
    "A) All roses are red",
    "B) Some roses may be red",
    "C) No roses are red",
    "D) All red things are roses"
  ],
  "answer": "B",
  "explanation": "From the premises, we can only conclude possibility, not certainty."
}
```

### Answer Agent Output

```json
{
  "answer": "B",
  "reasoning": "Using Venn diagrams: roses are a subset of flowers, and red overlaps with flowers. This means some roses MAY be in the red region, but it's not guaranteed."
}
```

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **GPU** | AMD MI300X (192GB HBM3) |
| **Framework** | ROCm 6.x |
| **VRAM Usage** | ~50GB (14B model in BF16) |
| **Training Time** | ~30 min per agent (3 epochs, 900 samples) |

## Team Algebra

Built for the **AAIPL Hackathon** at **IIT Delhi** (February 2026)
