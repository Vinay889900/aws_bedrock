# AWS Bedrock Generative AI Models Lab


This project is a practical Generative AI lab built on AWS Bedrock that demonstrates how to work with multiple foundation models for text and image generation using Python. It focuses on prompt-based inference, model comparison, and structured request handling across different large models such as Claude, LLaMA2, and Stable Diffusion.

The repository is designed as an experimentation and learning workspace for understanding how different Bedrock-supported models behave with prompts, parameters, and payload structures.

---

## 🚀 Project Goals

* Learn how to interact with AWS Bedrock foundation models
* Compare outputs from multiple LLMs using the same prompts
* Understand prompt formatting and payload design
* Experiment with parameter tuning (temperature, max tokens, etc.)
* Generate both text and images using GenAI models
* Build reusable Python scripts for Bedrock inference

---

## 🤖 Models Used

### Claude Model

* Used for natural language generation and structured responses
* Suitable for Q&A, summarization, reasoning, and instruction following
* Demonstrates prompt → completion workflow

### LLaMA2 Model

* Open-weight style large language model via Bedrock
* Used for conversational and generative tasks
* Helpful for comparing output style vs Claude

### Stable Diffusion

* Text-to-image generation model
* Generates images from prompt descriptions
* Demonstrates non-text GenAI capability through Bedrock

---

## 📁 Project Structure

claude.py — Claude model request & response handling
llama2.py — LLaMA2 inference script
stablediffusion.py — Image generation script
requirements.txt — Required Python libraries
test.json — Sample structured prompt payload
.gitignore — Git ignored files list
LICENSE — Usage license
README.md — Project documentation

---

## ⚙️ Environment Requirements

* Python 3.9+
* AWS account with Bedrock access enabled
* Bedrock model permissions granted
* Configured AWS credentials
* Internet connectivity for API calls

---

## 🔐 AWS Setup

Configure AWS credentials before running scripts:

```bash
aws configure
```

Provide:

* Access key ID
* Secret access key
* Default region (Bedrock-supported region)
* Output format (json)

Your IAM user/role must have:

* Bedrock model invoke permissions
* Foundation model access enabled

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/your-username/aws_bedrock.git
cd aws_bedrock
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Model Scripts

Run each model script independently:

```bash
python claude.py
python llama2.py
python stablediffusion.py
```

Each script sends a prompt payload to AWS Bedrock and prints or saves the generated output.

---

## 🧪 test.json Usage

The test.json file contains example prompt payloads used for:

* structured model requests
* parameter tuning experiments
* repeatable test inputs

You can modify prompts and parameters to observe output variation.

---

## 🔬 What This Project Demonstrates

* Bedrock API invocation workflow
* Prompt engineering basics
* Multi-model comparison
* Structured JSON payload design
* Text vs image generation pipelines
* Parameter sensitivity in GenAI outputs

---

## 🛠 Tech Stack

* Python
* AWS Bedrock
* Foundation Models (Claude, LLaMA2, Stable Diffusion)
* Boto3 SDK
* JSON payload handling

---

## 📌 Learning Outcomes

By working with this project, you gain hands-on experience in:

* Calling foundation models through APIs
* Designing prompts for different models
* Comparing generative behavior across models
* Handling GenAI responses programmatically
* Building reusable GenAI experiment scripts

---

