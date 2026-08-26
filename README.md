
# Product Pricing LLM Fine-Tuning Pipeline

This repository demonstrates how to fine-tune a Large Language Model (LLM) to accurately predict and estimate product prices based on text summaries. It serves as a portfolio project showcasing data preprocessing, Hugging Face Hub integration, and the OpenAI fine-tuning API lifecycle.

Fine Tuning a frontier model: GPT-4.1-nano

## 🔎 Overview
The script automates the complete downstream adaptation workflow:Data Ingestion: Loads a specialized e-commerce dataset (items_lite) directly from the Hugging Face Hub.Data Engineering: Transforms raw Python objects into formatted JSONL (JSON Lines) conversation structures required by OpenAI.File Management: Validates and uploads training and validation files to OpenAI cloud storage.Job Orchestration: Initiates and monitors a supervised fine-tuning (SFT) job using gpt-4.1-nano.Inference & Evaluation: Deploys the newly minted custom weights to run real-time inference on unseen test items.


UPDATE: The code does not work for new accounts, because of this update from OpenAI:

In May 2026, OpenAI announced the deprecation and wind-down of self-serve fine-tuning.The Block: New accounts or organizations that did not have an active history of fine-tuning prior to the announcement are strictly blocked from starting new training jobs (403 Permission Denied).The Final Date: Even for older, active accounts, all training functionality will completely shut off by January 6, 2027

## ✨ Key Features Demonstrated

- Environment Separation: Uses .env management to protect sensitive API keys (HF_TOKEN, OPENAI_API_KEY).
- Hugging Face Authentication: Authenticates programmatically and saves credentials to local Git contexts.
- ChatML Structuring: Builds deterministic user/assistant conversational roles to force numerical-only outputs from the model.
- API Lifecycle Control: Monitors tracking metrics, retrieves real-time training events, and extracts the deployed fine_tuned_model name dynamically.

## 🛠 Prerequisites & Installation

1. Clone the repository

    git clone https://github.com/mulangph/Product-PricingLLM-Fine-tuning-pipeline.git
    cd Product-PricingLLM-Fine-tuning-pipeline

2. Install dependencies with uv
    uv sync

3.  Set up Environment Variables
    OPENAI_API_KEY=your_openai_api_key_here
    HF_TOKEN=your_huggingface_token_here