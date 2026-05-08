# QGuard

A cutting-edge quantum machine learning API built with FastAPI and PennyLane, combining quantum computing with classical ML for advanced data analysis and predictive modeling.

## Features

- **Quantum Machine Learning** - Leverages PennyLane for quantum circuit-based ML models
- **User Authentication** - Secure JWT-based authentication with bcrypt password hashing
- **Data Upload & Processing** - Support for CSV/data file uploads with preprocessing
- **ML Models** - Scikit-learn and PyTorch integration for classical ML
- **Report Generation** - PDF report generation with ReportLab
- **REST API** - Full-featured API with automatic documentation
- **CORS Enabled** - Cross-origin support for frontend integration
- **Static File Serving** - Built-in frontend template support

## Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLAlchemy
- **Quantum ML**: PennyLane, PennyLane-Lightning
- **ML Libraries**: Scikit-learn, PyTorch, Pandas, NumPy
- **Data Processing**: Imbalanced-learn, Joblib
- **Database**: SQLAlchemy ORM
- **Authentication**: Python-JOSE, Bcrypt
- **Utilities**: Pydantic, Python-dotenv, ReportLab, Matplotlib

## Prerequisites

- Python 3.8+
- pip or conda
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/quantumwatch.git
   cd quantumwatch
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Development Mode
```bash
uvicorn main:app --reload
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will start at `http://localhost:8000`

## API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Frontend**: http://localhost:8000

## Project Structure

```
quantumwatch/
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
├── backend/
│   ├── api/
│   │   ├── routes/         # API endpoint definitions
│   │   │   ├── auth.py     # Authentication endpoints
│   │   │   ├── uploads.py  # File upload handling
│   │   │   ├── ml.py       # ML model endpoints
│   │   │   └── reports.py  # Report generation
│   │   └── schemas/        # Pydantic models/schemas
│   ├── core/
│   │   ├── config.py       # Application configuration
│   │   └── security.py     # Security utilities
│   ├── db/
│   │   └── models.py       # SQLAlchemy models
│   ├── ml/
│   │   ├── preprocessing.py # Data preprocessing
│   │   └── qml_models.py   # Quantum ML models
│   └── __init__.py
├── frontend/
│   ├── templates/
│   │   └── index.html      # Main frontend template
│   └── static/             # CSS, JS, images
├── models/                 # Saved ML models
├── reports/                # Generated reports
└── uploads/                # User uploaded files
```

## Environment Configuration

Create a `.env` file in the project root:

```
DATABASE_URL=sqlite:///./quantumwatch.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `POST /auth/logout` - Logout user

### Uploads
- `POST /uploads/` - Upload data file
- `GET /uploads/{file_id}` - Get upload details

### ML Models
- `POST /ml/train` - Train ML model
- `POST /ml/predict` - Make predictions
- `GET /ml/models` - List available models

### Reports
- `POST /reports/generate` - Generate report
- `GET /reports/{report_id}` - Get report

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8 .
```

## Database

The application uses SQLite by default. To initialize the database:

```python
from backend.db.models import create_tables
create_tables()
```

This is automatically called on application startup.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [PennyLane](https://pennylane.ai/) - Quantum ML framework
- [Xanadu](https://www.xanadu.ai/) - Quantum computing pioneers
