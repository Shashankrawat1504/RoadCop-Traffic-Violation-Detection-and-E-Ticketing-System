# 🚦 RoadCop-Traffic-Violation-Detection-and-E-Ticketing-System

> **AI-powered traffic monitoring and automated e-ticketing system for detecting two-wheeler traffic violations.**

RoadCop is an intelligent computer-vision-based traffic violation detection system designed to automate the identification of traffic violations from road/camera footage and streamline the generation of electronic traffic tickets.

The system combines **deep learning, computer vision, object detection, tracking, OCR, and a web-based application** to provide an automated workflow for traffic monitoring and violation management.

---

## 📌 Overview

Traditional traffic monitoring relies heavily on manual observation, which can be time-consuming and difficult to scale.

**RoadCop** aims to automate this process by analyzing traffic footage, identifying relevant objects and violations, extracting vehicle information, recording violation evidence, and generating an electronic ticket.

### Core Workflow

```text
Camera / Video Input
        │
        ▼
┌──────────────────────┐
│  Object Detection    │
│  & Tracking          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Violation Detection  │
│      & Analysis      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ License Plate / OCR  │
│     Processing       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Violation Record     │
│    & Evidence        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ E-Ticket Generation  │
└──────────────────────┘
```

---

## ✨ Features

* 🚗 **AI-based traffic object detection**
* 🏍️ **Two-wheeler traffic violation detection**
* 🎥 **Video/image-based traffic analysis**
* 🔍 **Object detection and tracking**
* 🔤 **License plate recognition using OCR**
* 📋 **Violation record management**
* 🎫 **Automated e-ticket generation**
* 📄 **PDF ticket generation**
* 🗄️ **MySQL database integration**
* 🌐 **Flask-based backend**
* 🖥️ **Web interface**
* ⚙️ **Configurable detection pipeline**
* 📊 **Evidence-oriented violation processing**

---

## 🧠 Technology Stack

### Artificial Intelligence & Computer Vision

| Technology        | Purpose                          |
| ----------------- | -------------------------------- |
| **PyTorch**       | Deep learning framework          |
| **YOLO / YOLOv5** | Object detection                 |
| **Ultralytics**   | Modern YOLO tooling              |
| **OpenCV**        | Image and video processing       |
| **PaddleOCR**     | License plate / text recognition |
| **PaddlePaddle**  | OCR and deep learning support    |
| **FilterPy**      | Filtering/tracking utilities     |
| **scikit-image**  | Image processing                 |

### Backend

| Technology                 | Purpose                   |
| -------------------------- | ------------------------- |
| **Python**                 | Core development language |
| **Flask**                  | Web application/backend   |
| **MySQL**                  | Persistent data storage   |
| **mysql-connector-python** | Database connectivity     |
| **python-dotenv**          | Environment configuration |
| **FPDF2**                  | PDF/e-ticket generation   |

### Frontend

The project includes a dedicated `web/` application directory for the web interface.

---

## 🏗️ Project Structure

```text
RoadCop-Traffic-Violation-Detection-and-E-Ticketing-System/
│
├── assets/
│   └── Project assets and supporting resources
│
├── config/
│   └── Configuration files
│
├── scripts/
│   └── Utility and supporting scripts
│
├── src/
│   └── roadcop/
│       └── Core RoadCop application
│
├── web/
│   └── Web interface
│
├── yolov5/
│   └── YOLOv5 detection components
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── run.py
```

---

## ⚙️ System Architecture

RoadCop is organized into several logical components:

### 1. Input Layer

The system receives traffic imagery/video that can be processed by the computer-vision pipeline.

```text
Video / Image
      │
      ▼
OpenCV Processing
```

### 2. Detection Layer

Deep-learning-based object detection identifies relevant road users and objects.

```text
Input Frame
     │
     ▼
YOLO Detector
     │
     ├── Vehicle
     ├── Motorcycle
     └── Other Relevant Objects
```

### 3. Tracking Layer

Detected objects can be tracked across consecutive frames to maintain object identity during video processing.

```text
Frame 1 ──┐
Frame 2 ──┤
Frame 3 ──┼──► Object Tracking
Frame 4 ──┘
```

### 4. Violation Analysis

Detection and tracking information is used by the application logic to determine whether a traffic violation has occurred.

### 5. OCR Layer

When required, license-plate information is extracted using **PaddleOCR**.

```text
Vehicle
   │
   ▼
License Plate Region
   │
   ▼
OCR Processing
   │
   ▼
Vehicle Registration Number
```

### 6. Database Layer

Violation information and associated records are stored using MySQL.

### 7. E-Ticket Layer

The detected violation can be converted into an electronic ticket containing relevant violation information and generated evidence/documentation.

---

## 🚀 Getting Started

### Prerequisites

Make sure the following are installed:

* Python 3.x
* Git
* MySQL Server
* pip
* A compatible CPU/GPU environment for the selected deep-learning models

For GPU acceleration, install a PyTorch/PaddlePaddle configuration compatible with your CUDA environment.

---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shashankrawat1504/RoadCop-Traffic-Violation-Detection-and-E-Ticketing-System.git
```

### 2. Navigate into the project

```bash
cd RoadCop-Traffic-Violation-Detection-and-E-Ticketing-System
```

### 3. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The repository's current `requirements.txt` includes packages such as PyTorch, torchvision, OpenCV, PaddleOCR, PaddlePaddle, Flask, MySQL Connector, FPDF2, Ultralytics, NumPy, SciPy, and supporting computer-vision libraries.

---

## 🗄️ Database Configuration

RoadCop uses **MySQL** for application data.

Create a MySQL database and configure the required database credentials through the project's configuration/environment settings.

Example:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=roadcop
DB_USER=root
DB_PASSWORD=your_password
```

> Use the variable names expected by the application's configuration files if they differ from the example above.

**Never commit real database passwords, API keys, or other secrets to GitHub.**

---

## ▶️ Running the Application

The repository provides a root-level `run.py` entry point.

Run:

```bash
python run.py
```

The launcher adds the `src` directory to Python's module path, initializes the database, and starts the Flask application in debug mode.

Once the application starts, open the local URL displayed by Flask in your browser.

---

## 🔄 Application Workflow

```text
                 ┌─────────────────┐
                 │ Traffic Footage │
                 └────────┬────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Frame Extraction  │
                │    / OpenCV       │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ YOLO Object       │
                │ Detection         │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Object Tracking   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Violation Logic   │
                └─────────┬─────────┘
                          │
                    Violation?
                     /       \
                   No         Yes
                   │           │
                   ▼           ▼
                 Continue    Evidence
                              Capture
                                │
                                ▼
                         ┌──────────────┐
                         │ Plate / OCR  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ MySQL Record │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ E-Ticket PDF │
                         └──────────────┘
```

---

## 📊 Detection Pipeline

RoadCop follows a computer-vision pipeline consisting of:

1. **Frame acquisition**
2. **Image preprocessing**
3. **Object detection**
4. **Object tracking**
5. **Violation-rule evaluation**
6. **License plate localization**
7. **OCR processing**
8. **Violation database entry**
9. **Evidence/document generation**
10. **E-ticket generation**

This architecture separates the AI detection components from the application and ticketing workflow, making individual components easier to modify and extend.

---

## 🎫 E-Ticket Generation

When a violation is identified, RoadCop can associate the violation with information such as:

```text
Violation ID
Vehicle Registration Number
Violation Type
Date & Time
Location
Fine Amount
Evidence
Status
```

The system uses **FPDF2** for PDF generation.

---

## 🧪 Development

For development, activate the virtual environment before running the application:

```bash
venv\Scripts\activate
python run.py
```

The application currently starts Flask with:

```python
app.run(debug=True, use_reloader=False)
```

For production deployment, use an appropriate WSGI server and production configuration rather than Flask's development server.

---

## 🔐 Security Considerations

Before deploying RoadCop in a production environment:

* Store credentials in environment variables.
* Do not commit `.env` files containing secrets.
* Disable Flask debug mode.
* Use secure database credentials.
* Validate uploaded images/videos.
* Restrict access to violation records.
* Protect generated evidence and e-ticket documents.
* Implement authentication and authorization for administrative functions.
* Use HTTPS in production.
* Apply appropriate data-retention and privacy policies.

---

## 📈 Future Enhancements

Potential future improvements include:

* [ ] Real-time CCTV/RTSP camera integration
* [ ] Multi-camera monitoring
* [ ] Improved violation classification
* [ ] Advanced vehicle tracking
* [ ] Real-time alert notifications
* [ ] SMS/email notification integration
* [ ] Driver/vehicle history
* [ ] Payment gateway integration
* [ ] Admin analytics dashboard
* [ ] Role-based authentication
* [ ] Cloud deployment
* [ ] Containerized deployment using Docker
* [ ] Model optimization for edge devices
* [ ] Improved low-light and adverse-weather detection
* [ ] Model performance monitoring
* [ ] Automated model retraining pipeline

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Shashank Rawat**

GitHub: [@Shashankrawat1504](https://github.com/Shashankrawat1504)

---

## ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🔧 Contribute enhancements

---
