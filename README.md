# Client Records Data Entry System

A comprehensive web application for managing client records data entry with agent management, task assignment, and data export capabilities.

## Features

- 🏛️ **Admin Dashboard**: Agent registration, task assignment, data export
- 👤 **Agent Interface**: Secure login, form submission, progress tracking  
- 📊 **Data Management**: Excel export, filtering, session tracking
- 🖼️ **Image Processing**: ZIP upload, bulk task assignment
- 🔒 **Security**: Password hashing, session management, CORS protection

## Tech Stack

**Backend:**
- FastAPI (Python)
- SQLAlchemy (Database ORM)
- SQLite Database
- Pandas (Excel export)
- Pillow (Image processing)

**Frontend:**
- HTML5, CSS3, JavaScript
- Responsive design
- Modern UI with animations

## Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
