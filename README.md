# 차비스 (Cha-Vis) 🚗💬

**차비스 (Cha-Vis)** is an innovative AI-driven personalized driving partner designed to enhance the driving experience by integrating real-time navigation with emotion control. Cha-Vis not only provides accurate directions but also analyzes user emotions through speech, offering empathetic interactions to promote safer and more comfortable driving.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [1. Basic Conversations and Directions](#1-basic-conversations-and-directions)
  - [2. Real-time Data Collection and Analysis](#2-real-time-data-collection-and-analysis)
  - [3. Personalized Recommendation System](#3-personalized-recommendation-system)
  - [4. Emotion Recognition and Management](#4-emotion-recognition-and-management)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Example API Call](#example-api-call)
- [Code Structure](#code-structure)
- [Performance Optimization](#performance-optimization)
  - [Applying Flash Attention to HuBERT](#applying-flash-attention-to-hubert)
  - [Profiling and Analysis](#profiling-and-analysis)
  - [Academic Significance](#academic-significance)
  - [Future Improvement Directions](#future-improvement-directions)
- [Security Considerations](#security-considerations)
  - [API Key and Credential Management](#api-key-and-credential-management)
  - [Data Privacy and Protection](#data-privacy-and-protection)
  - [Dependency Management](#dependency-management)
  - [Code Security Practices](#code-security-practices)
- [Incident Response](#incident-response)
- [Contributing](#contributing)
  - [How to Contribute](#how-to-contribute)
  - [Code of Conduct](#code-of-conduct)
  - [Reviewing Process](#reviewing-process)
  - [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)
- [Future Enhancements](#future-enhancements)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)
- [Additional Resources](#additional-resources)

---

## Overview

Driving can be a stressful experience, especially in heavy traffic or during long commutes. **Cha-Vis** aims to transform this by acting as an AI-driven companion that not only navigates but also helps manage the driver's emotions in real-time. By analyzing speech patterns and driving behavior, Cha-Vis provides empathetic interactions, calming conversations, and personalized recommendations to ensure a safe and pleasant journey.

**Key Objectives:**

- **Safety:** Reduce the risk of accidents by monitoring and managing driver emotions.
- **Comfort:** Provide a supportive and engaging driving experience through conversational AI.
- **Efficiency:** Optimize routes based on real-time data to save time and fuel.
- **Personalization:** Tailor recommendations and interactions based on individual user preferences and behaviors.

---

## Features

### 1. Basic Conversations and Directions

- **Interactive Navigation:** Provides turn-by-turn directions similar to standard navigation systems.
- **Conversational Interface:** Engage in basic conversations, making the driving experience less monotonous.
- **Voice Commands:** Use voice commands to interact with the navigation system hands-free.
- **Multi-language Support:** Supports multiple languages to cater to a diverse user base.

**Example Interaction:**

> **User:** "How do I get to Gangnam Station?"
>
> **Cha-Vis:** "Sure, I'll guide you to Gangnam Station. In 500 meters, take a right onto Teheran-ro."

### 2. Real-time Data Collection and Analysis

- **SK Network Integration:** Utilizes SK Network to collect and analyze real-time data on traffic, weather, and local events.
- **Event Summarization:** Offers summaries of events, such as festivals or traffic incidents, with positive reviews from social media.

  **Example:**
  
  > "There's a festival ahead on your route with many positive reviews on social media. Would you like to stop by?"
  
- **Traffic Updates:** Provides real-time traffic conditions to suggest the fastest and most efficient routes.
- **Weather Alerts:** Alerts users about upcoming weather conditions that may affect driving.

### 3. Personalized Recommendation System

- **User-specific Customization:** Adapts recommendations based on individual preferences and driving habits.
- **Contextual Information:** Considers user interests, driving patterns, and destinations to provide tailored information.
- **Real-time Information Fetching:** Utilizes Google Search to reflect real-time information when necessary.

**Example:**

> "Based on your interest in coffee shops, I recommend visiting the famous Gangnam BBQ restaurant. Would you like directions?"

### 4. Emotion Recognition and Management

- **Multidimensional Emotion Analysis:** Assesses user emotions through various channels:
  - **Voice Analysis:** Real-time analysis of tone, speed, and volume changes.
  - **Language Analysis:** Examination of word choice and sentence structure.
  - **Driving Pattern Analysis:** Infers emotional state from acceleration and braking data.

- **Emotion-based Care:**
  - **Stress Detection:** Plays calming music or reduces volume, engages in soothing conversations to minimize accident risks.
  - **Emotion Coaching:** Offers simple AI-driven coaching sessions, such as guided breathing exercises or empathetic dialogues.
  
  **Example:**
  
  > "It's natural to feel angry in this situation. Please take a deep breath while I explain what's happening. We'll resolve this together."

---

## Demo

*Coming Soon! Stay tuned for a live demonstration of Cha-Vis in action.*

---

## Technologies Used

- **Frontend:**
  - **Flutter:** For building cross-platform mobile applications with a rich user interface.
  
- **Backend:**
  - **FastAPI:** A high-performance web framework for building APIs with Python.
  
- **AI Models:**
  - **Whisper:** For accurate speech-to-text transcription.
  - **HuBERT:** For emotion recognition and analysis. (https://huggingface.co/HyunaZ/hubert_emotion)
  - **LangChain:** For managing conversational prompts and memory.
  - **VITS:** For Emotional text-to-speech transcription.
  
- **Data Analysis:**
  - **SK Network:** For real-time traffic, weather, and event data.
  
- **Search Integration:**
  - **Google Custom Search API:** For fetching real-time information based on user queries.
  
- **Other Libraries and Tools:**
  - `torch`, `librosa`: For audio processing and machine learning tasks.
  - `pytorch_lightning`, `transformers`: For model training and inference.
  - `flash_attn`: For optimized attention mechanisms in AI models.
  - `dotenv`: For managing environment variables securely.
  - `requests`: For making HTTP requests to external APIs.
  - `logging`: For tracking events that happen when the software runs.

---

## Architecture

![Architecture Diagram](https://github.com/AI-and-Application/AI-Project/blob/main/architecture_design.png?raw=true)

### 1. Frontend (Flutter)

- **User Interface:** Captures user speech, displays navigation directions, and provides emotional feedback.
- **Voice Interaction:** Allows users to interact with Cha-Vis through voice commands.
- **Real-time Updates:** Receives real-time data and updates from the backend to provide dynamic responses.

### 2. Backend (FastAPI)

- **Speech Transcription:** Utilizes Whisper to convert user speech into text.
- **Emotion Analysis:** Employs HuBERT to detect and analyze user emotions from the transcribed text and driving behavior.
- **Data Integration:** Gathers real-time traffic, weather, and local event data via SK Network.
- **Search Functionality:** Integrates Google Custom Search API to fetch relevant information based on user queries.
- **Conversational AI:** Uses LangChain to manage prompts and maintain conversation context, generating appropriate responses.

### 3. AI Models

- **Whisper:** Handles accurate and efficient speech-to-text transcription, enabling seamless voice interactions.
- **HuBERT:** Provides reliable emotion recognition through advanced audio and language analysis.
- **LangChain:** Manages complex conversational flows and memory to sustain meaningful and context-aware interactions.
- **VITS:** For Emotional text-to-speech transcription.

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Flutter SDK**
- **Docker (Optional for containerization)**
- **CUDA (If using GPU for model inference)**
- **Git:** For cloning the repository.

### Backend Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AI-and-Application/AI-Project.git
   cd AI-Project
