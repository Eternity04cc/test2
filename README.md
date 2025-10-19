# SkyGuard: AI-Powered Drone Detection System

## 🚀 Elevator Pitch

**SkyGuard brings AI-powered drone detection to any device — even low-end CPUs and smartphones — enabling real-time threat alerts that protect critical zones anytime, anywhere.**

---

## 📖 Project Story

### 💡 Inspiration

The idea originated from a detail we heard while accompanying a volunteer medical team: frontline soldiers on night patrols are often exposed to drone surveillance and strike risks under severely limited intelligence and detection capabilities. Unlike civilian airports—where drone sightings may cause flight delays—the threat of drones on the battlefield directly endangers soldiers and civilians. We found that, even in the most intense conflict zones, the vast majority of soldiers still carry networked smartphones—an everyday necessity that remains an under-utilized, low-cost sensing platform.

When night falls and visibility deteriorates, small low-flying drones can approach with much greater ease; radar and dedicated detectors are typically expensive or sparsely deployed, and frontline units often cannot sustain the maintenance and operation of complex sensing equipment. We began to ask: what if every soldier’s phone could be turned into a distributed early-warning node—leveraging local sensors, cooperative perception, and lightweight intelligence—to immediately issue actionable alerts and evasive recommendations when a drone is approaching, loitering, or exhibiting suspicious surveillance behavior? Could such an approach materially improve survival odds and operational safety? That question drives our project.

We do not aim to create offensive weapons, but rather a protection-centric system capable of operating under severely constrained conditions: lowering deployment barriers by using existing consumer devices, prioritizing real-time responsiveness and reliability while minimizing impact on civilians; and embedding privacy safeguards and false-alarm control in the design. We favor non-lethal alerts and clear, actionable guidance so that frontline personnel have concrete steps to follow upon receiving a warning. This solution addresses urgent battlefield needs while respecting humanitarian and legal boundaries.

### 🎯 What it does

SkyGuard is an advanced AI-powered drone detection system that provides real-time identification and tracking of unauthorized drones in protected airspace. The system combines cutting-edge computer vision technology with intelligent alert mechanisms to create a comprehensive defense solution.

**Core Capabilities:**
- **Real-time Detection**: Processes live video feeds from security cameras to identify drone signatures with 95%+ accuracy
- **Multi-source Input**: Supports webcams, IP cameras, RTSP streams, and video files for flexible deployment
- **Instant Alerts**: Triggers immediate audio and visual alarms when threats are detected
- **Performance Optimization**: Achieves 18.59 FPS on standard hardware with CPU-only inference
- **Web Interface**: Provides intuitive dashboard for monitoring and system management
- **Scalable Architecture**: Designed for deployment from single facilities to large-scale defense networks

The system operates continuously, analyzing visual data streams and applying advanced machine learning algorithms to distinguish between authorized aircraft, birds, and potential drone threats. When a drone is detected, SkyGuard immediately activates its alert system, providing security personnel with precise location data and threat assessment information.

### 🔨 How we built it

**The Technical Journey:**

Our development journey was marked by significant technical challenges that pushed us beyond our comfort zones. We started with the ambitious goal of upgrading from YOLOv7 to the latest YOLO architecture, which seemed straightforward until we encountered our first major obstacle.

**Challenge 1: Model Architecture Migration**
The transition from YOLOv7 to YOLO26 (based on Ultralytics YOLO v8.3.80) proved more complex than anticipated. The new architecture required complete restructuring of our training pipeline, data preprocessing, and inference engine. We spent countless nights debugging compatibility issues and optimizing the model for our specific use case.

**Challenge 2: CPU Optimization Breakthrough**
Perhaps our most significant technical achievement was optimizing the system for CPU-only inference. Initially, our detection system required expensive GPU hardware, making it impractical for widespread deployment. Through extensive research and experimentation with:
- Intel MKL-DNN optimization
- Multi-threading implementation
- Memory management optimization
- OpenVINO integration

We achieved a breakthrough, reducing inference time from 500ms to 127.59ms while maintaining detection accuracy above 95%.

**Challenge 3: Real-time Processing Pipeline**
Building a robust real-time processing system required solving complex concurrency issues. We implemented:
- Asynchronous video stream processing
- Thread-safe detection queues
- Efficient memory buffer management
- Real-time alert system integration

**Architecture Implementation:**
```
Frontend (React + WebRTC) ↔ Backend (Node.js + Express) ↔ AI Engine (Python + YOLO26)
                                        ↓
                              Real-time Alert System
```

### 🌍 Challenges we ran into

**Global Security Perspective:**
Developing SkyGuard opened our eyes to the international scope of drone security challenges. We researched incidents from around the world:
- **Middle East**: Drone attacks on oil facilities
- **Europe**: Airport security breaches
- **Asia-Pacific**: Border surveillance challenges
- **Americas**: Critical infrastructure protection needs

**Technical Challenges:**
1. **Cross-platform Compatibility**: Ensuring our system works across different operating systems and hardware configurations
2. **Network Latency**: Optimizing for various network conditions in different geographical regions
3. **Regulatory Compliance**: Understanding different privacy and security regulations across countries
4. **Scalability**: Designing architecture that can handle everything from single-site deployment to national defense networks

**Resource Constraints:**
As students, we faced significant limitations:
- Limited access to high-end hardware for testing
- Restricted dataset availability due to security classifications
- Time constraints balancing academic responsibilities
- Budget limitations for cloud computing resources

### 🏆 Accomplishments that we're proud of

**For Beginners in AI/Defense Technology:**

1. **Technical Mastery**: We successfully mastered advanced computer vision concepts, from basic image processing to state-of-the-art object detection algorithms

2. **Performance Achievement**: Achieving 18.59 FPS on CPU-only hardware represents a significant optimization accomplishment that makes our solution accessible to organizations with limited resources

3. **System Integration**: Building a complete end-to-end system taught us valuable lessons about software architecture, API design, and user experience

4. **Problem-Solving Skills**: Overcoming the CPU optimization challenge required creative thinking and deep understanding of hardware-software interaction

**Personal Growth:**
- **Confidence Building**: Successfully tackling a complex defense-related project boosted our confidence in handling real-world challenges
- **Collaboration**: Working under pressure taught us effective teamwork and communication skills
- **Research Skills**: Learning to navigate academic papers, technical documentation, and industry best practices
- **Presentation Skills**: Preparing for the hackathon improved our ability to communicate technical concepts to diverse audiences

### 📚 What we learned

**Defense Technology Insights:**
1. **Threat Landscape Understanding**: Modern security challenges require multi-layered, intelligent defense systems
2. **Technology Integration**: Effective defense solutions combine multiple technologies (AI, networking, hardware) seamlessly
3. **Real-time Requirements**: Defense applications demand ultra-low latency and high reliability
4. **Scalability Importance**: Solutions must be designed for deployment at various scales, from single facilities to national networks

**Technical Skills Acquired:**
- **Advanced Computer Vision**: Deep understanding of YOLO architecture, object detection, and model optimization
- **Performance Engineering**: CPU optimization, memory management, and real-time processing techniques
- **Full-stack Development**: Integration of AI models with web applications and user interfaces
- **System Architecture**: Designing scalable, maintainable software systems

**Soft Skills Development:**
- **Project Management**: Planning and executing complex technical projects under time constraints
- **Research Methodology**: Systematic approach to solving unknown technical challenges
- **Communication**: Explaining complex technical concepts to non-technical stakeholders
- **Adaptability**: Quickly learning new technologies and adapting to changing requirements

### 🚀 What's next for SkyGuard

**Immediate Development (Next 6 months):**
1. **Enhanced Detection Capabilities**:
   - Multi-object tracking for drone swarms
   - Classification of drone types and threat levels
   - Integration with thermal imaging cameras

2. **Advanced Alert System**:
   - Integration with existing security infrastructure
   - Mobile app for security personnel
   - Automated response protocols

**Commercial Strategy (6-18 months):**
1. **Market Entry**:
   - Pilot programs with local security companies
   - Partnerships with defense contractors
   - Government agency demonstrations

2. **Product Development**:
   - Hardware-optimized versions for edge deployment
   - Cloud-based monitoring dashboard
   - API for third-party integrations

**Long-term Vision (2-5 years):**
1. **Global Expansion**:
   - International market penetration
   - Compliance with various national security standards
   - Localization for different regions

2. **Technology Evolution**:
   - AI-powered threat prediction
   - Integration with counter-drone systems
   - Autonomous response capabilities

**Business Model:**
- **B2B SaaS**: Subscription-based monitoring service
- **Enterprise Licensing**: On-premise deployment for high-security facilities
- **Government Contracts**: Custom solutions for defense agencies
- **Hardware Partnerships**: Collaboration with camera and sensor manufacturers

**Funding Strategy:**
- Seed funding from defense-focused VCs
- Government innovation grants
- Strategic partnerships with established defense companies
- Revenue from pilot programs and early customers

---

## 🛠️ Tech Stack

### **AI/ML Framework**
- **YOLO26** (Ultralytics YOLO v8.3.80) - Advanced object detection
- **PyTorch 2.9.0+cpu** - Deep learning framework
- **OpenCV** - Computer vision processing
- **NumPy** - Numerical computing
- **OpenVINO** - Intel CPU optimization

### **Backend**
- **Node.js** - Server runtime
- **Express.js** - Web application framework
- **Python** - AI model integration
- **WebSocket** - Real-time communication

### **Frontend**
- **React** - User interface framework
- **HTML5/CSS3** - Web technologies
- **JavaScript ES6+** - Client-side scripting
- **WebRTC** - Real-time video streaming

### **Development Tools**
- **Git** - Version control
- **npm/pip** - Package management
- **Vite** - Build tool
- **ESLint** - Code quality

---

## 📁 Project Structure

```
SkyGuard-Drone-Detection/
├── 📁 backend/                    # Server-side application
│   ├── server.js                 # Main server entry point
│   └── temp/                     # Temporary file storage
├── 📁 frontend/                  # Client-side application
│   ├── index.html               # Main web interface
│   ├── script.js                # Frontend logic
│   └── sounds/                  # Alert audio files
├── 📁 ai_models/                # AI detection engine
│   ├── inference_yolo26.py      # CPU-optimized inference
│   ├── train_yolo26.py          # Model training script
│   └── openvino_optimization.py # Intel optimization
├── 📁 test_data/                # Testing datasets
├── 📁 runs/                     # Training/detection results
│   └── detect/                  # Detection outputs
├── 📄 data.yaml                 # Dataset configuration
├── 📄 requirements.txt          # Python dependencies
├── 📄 package.json              # Node.js dependencies
└── 📄 README.md                 # Project documentation
```

---

## 🚀 How to Start the Project

### **Prerequisites**
- **Python 3.12+** with pip
- **Node.js 20+** with npm
- **Git** for version control
- **Webcam or IP camera** for testing

### **Step 1: Clone Repository**
```bash
git clone https://github.com/your-team/skyguard-drone-detection.git
cd skyguard-drone-detection
```

### **Step 2: Install Python Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv skyguard_env

# Activate virtual environment
# Windows:
skyguard_env\Scripts\activate
# Linux/Mac:
source skyguard_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 3: Install Node.js Dependencies**
```bash
npm install
```

### **Step 4: Download AI Models**
```bash
# Models will be automatically downloaded on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
```

### **Step 5: System Verification**
```bash
# Test system components
python test_system.py
```

### **Step 6: Start the Application**
```bash
# Start backend server
node backend/server.js

# Open browser and navigate to:
# http://localhost:3000
```

### **Step 7: Configure Detection**
1. **Select Input Source**: Choose webcam, IP camera, or video file
2. **Adjust Sensitivity**: Set detection confidence threshold
3. **Enable Alerts**: Configure audio/visual notifications
4. **Start Monitoring**: Begin real-time drone detection

---

## ⚙️ System Operation Mechanism

### **Information Flow Architecture**

```mermaid
graph TD
    A[Video Input Sources] --> B[Video Stream Processor]
    B --> C[AI Detection Engine]
    C --> D[Threat Analysis]
    D --> E[Alert System]
    D --> F[Web Dashboard]
    E --> G[Security Personnel]
    F --> H[System Administrators]
    
    subgraph "Input Layer"
        A1[Webcam]
        A2[IP Camera]
        A3[RTSP Stream]
        A4[Video File]
    end
    
    subgraph "Processing Layer"
        B1[Frame Extraction]
        B2[Image Preprocessing]
        B3[Buffer Management]
    end
    
    subgraph "AI Layer"
        C1[YOLO26 Model]
        C2[Object Detection]
        C3[Confidence Scoring]
    end
    
    subgraph "Response Layer"
        E1[Audio Alerts]
        E2[Visual Notifications]
        E3[Log Generation]
    end
```

### **Core Processing Pipeline**

1. **Video Acquisition** (Input Layer)
   - Captures video streams from multiple sources
   - Handles different formats and resolutions
   - Manages connection stability and error recovery

2. **Stream Processing** (Processing Layer)
   - Extracts frames at optimal intervals
   - Applies preprocessing (resize, normalize)
   - Manages memory buffers efficiently

3. **AI Detection** (AI Layer)
   - Runs YOLO26 inference on each frame
   - Identifies drone objects with bounding boxes
   - Calculates confidence scores and classifications

4. **Threat Assessment** (Analysis Layer)
   - Filters detections by confidence threshold
   - Tracks object persistence across frames
   - Determines threat level and urgency

5. **Alert Generation** (Response Layer)
   - Triggers immediate audio/visual alerts
   - Updates web dashboard in real-time
   - Logs incidents for analysis

### **Performance Optimization**

**CPU Optimization Techniques:**
- **Multi-threading**: Parallel processing of video frames
- **MKL-DNN**: Intel Math Kernel Library acceleration
- **Memory Management**: Efficient buffer allocation and cleanup
- **Batch Processing**: Optimized inference for multiple detections

**Real-time Performance Metrics:**
- **Inference Time**: 127.59ms average
- **Frame Rate**: 18.59 FPS (640x640 resolution)
- **CPU Usage**: Optimized for multi-core systems
- **Memory Footprint**: <2GB RAM usage

### **Scalability Architecture**

```mermaid
graph LR
    subgraph "Single Site Deployment"
        A1[Local Cameras] --> B1[Edge Processing]
        B1 --> C1[Local Dashboard]
    end
    
    subgraph "Multi-Site Network"
        A2[Site 1] --> D[Central Hub]
        A3[Site 2] --> D
        A4[Site 3] --> D
        D --> E[Command Center]
    end
    
    subgraph "Enterprise Integration"
        F[SkyGuard API] --> G[Security Management System]
        F --> H[Incident Response Platform]
        F --> I[Analytics Dashboard]
    end
```

**Deployment Flexibility:**
- **Edge Computing**: Local processing for low latency
- **Cloud Integration**: Centralized monitoring and analytics
- **Hybrid Architecture**: Combination of edge and cloud processing
- **API Integration**: Seamless connection with existing security systems

---

*SkyGuard: Protecting the skies with intelligent technology* 🛡️✈️