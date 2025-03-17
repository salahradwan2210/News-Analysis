# Financial News Analysis Pipeline | تحليل الأخبار المالية وترتيب الأسهم للاستثمار

## Overview | نظرة عامة
This project implements a financial news analysis pipeline that processes news articles to predict their sentiment and potential impact on stock prices. It includes both an API server and a web interface for easy interaction.

## Features | المميزات
- Real-time news analysis | تحليل الأخبار في الوقت الفعلي
- Sentiment prediction | تحليل المشاعر للأخبار المالية
- Impact assessment | تقييم تأثير الأخبار على الأسهم
- Stock symbol extraction | استخراج رموز الأسهم
- Stock rankings based on news analysis | ترتيب الأسهم بناءً على تحليل الأخبار
- Interactive web interface | واجهة مستخدم تفاعلية
- RESTful API | واجهة برمجة تطبيقات

## Project Structure | هيكل المشروع
```
.
├── app.py              # Main application file (API + Web UI)
├── pipeline.py         # Pipeline orchestration
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
├── data/              # Data directory
│   ├── raw/           # Raw news data
│   └── processed/     # Processed data
├── models/            # Trained models
├── src/               # Source code
│   ├── api/          # API implementation
│   ├── models/       # Model definitions
│   ├── features/     # Feature engineering
│   └── utils/        # Utility functions
├── tests/            # Test files
└── web/              # Web interface files
    ├── static/       # Static assets
    └── templates/    # HTML templates
```

## Setup | التثبيت
1. Clone the repository | نسخ المشروع
2. Create a virtual environment | إنشاء بيئة افتراضية:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies | تثبيت المتطلبات:
   ```bash
   pip install -r requirements.txt
   ```

## Usage | الاستخدام
1. Start the application | تشغيل التطبيق:
   ```bash
   python pipeline.py
   ```
   This will start both the API server and web interface.

2. Access the web interface | الوصول إلى واجهة المستخدم:
   - Open http://localhost:8003 in your browser

3. API Documentation | توثيق واجهة برمجة التطبيقات:
   - Swagger UI: http://localhost:8002/docs
   - ReDoc: http://localhost:8002/redoc

## API Endpoints | نقاط النهاية للواجهة البرمجية
- `POST /analyze`: Analyze a single news article | تحليل خبر واحد
- `POST /batch-analyze`: Analyze multiple news articles | تحليل مجموعة من الأخبار
- `GET /rankings`: Get stock rankings | الحصول على ترتيب الأسهم
- `GET /rankings/{symbol}`: Get ranking for a specific stock | الحصول على ترتيب سهم محدد

## Web Interface | واجهة المستخدم
The web interface provides | توفر واجهة المستخدم:
- News analysis form | نموذج تحليل الأخبار
- Batch analysis upload | تحميل مجموعة من الأخبار للتحليل
- Stock rankings visualization | عرض مرئي لترتيب الأسهم
- Stock details view | عرض تفاصيل الأسهم
- Real-time updates via WebSocket | تحديثات فورية

## Development | التطوير
1. Run tests | تشغيل الاختبارات:
   ```bash
   pytest
   ```

2. Format code | تنسيق الكود:
   ```bash
   black .
   isort .
   ```

3. Check code quality | فحص جودة الكود:
   ```bash
   flake8
   ```

## License | الترخيص
MIT License

## Authors | المؤلفون
- Qafza Team | فريق قفزة
