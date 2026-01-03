# UVPAS - University Video Processing & Analysis System

A comprehensive CCTV video analysis system with face recognition capabilities for educational institutions.

## 🚀 Features

### 1. Multi-Side Student Photo Management
- **3-Side Photo Upload**: Upload front, left, and right side photos for each student
- **Organized Storage**: Photos are stored by registration number with side indicators
- **Enhanced Recognition**: Better face recognition accuracy using multiple angles

### 2. Advanced Video Analysis
- **Real-time Face Detection**: Automatically detects faces in uploaded videos
- **Student Identification**: Matches detected faces with registered student photos
- **Green Box Overlays**: Visual indicators around detected faces with hover effects
- **Detailed Information**: Shows student details on hover/click

### 3. Interactive Face Detection
- **Hover Details**: Move mouse over green boxes to see student information
- **Clickable Profiles**: Click on identified faces to view student profiles
- **Responsive Design**: Works on different screen sizes and video resolutions

### 4. Student Profile System
- **Comprehensive Profiles**: View all student information and photos
- **Hyperlink Navigation**: Direct links to student portals and profiles
- **Photo Gallery**: Display all 3 sides of student photos

### 5. User Management
- **Role-based Access**: Separate dashboards for faculty and students
- **Student Portal**: Students can view and update their information
- **Faculty Dashboard**: Upload photos, videos, and manage student data

## 🛠️ Technical Requirements

### System Requirements
- Python 3.8 or higher
- OpenCV with face detection capabilities
- Sufficient RAM for video processing (4GB+ recommended)
- Storage space for photos and videos

### Dependencies
```
flask==2.2.5
opencv-python==4.9.0.80
numpy==1.26.4
face_recognition==1.3.0
werkzeug==2.2.3
pillow==10.0.0
```

## 📁 Project Structure

```
cctv/
├── app.py                 # Main Flask application
├── auth.py               # Authentication and authorization
├── data_manager.py       # Database management
├── recognition.py        # Face recognition engine
├── requirements.txt      # Python dependencies
├── auth.db              # Authentication database
├── static/              # Static files (CSS, JS, uploads)
├── templates/           # HTML templates
├── student_photos/      # Student photos (3 sides)
├── students_data/       # Student database
└── track_cache/         # Recognition cache
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd cctv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize the System
```bash
python app.py
```

### 4. Access the Application
- Open browser and go to `http://localhost:5000`
- Use demo credentials to login

## 👥 User Roles & Access

### Faculty Access
- **Login**: PSBCSE / faculty@123
- **Features**:
  - Upload student photos (3 sides)
  - Upload and analyze CCTV videos
  - View analysis results with green boxes
  - Access student profiles
  - Build gait gallery

### Student Access
- **Login**: 99220041142 / student@123
- **Features**:
  - View personal information
  - Update profile details
  - Access student portal

## 📸 Photo Upload System

### Student Photo Requirements
1. **Front Photo** (Required): Clear front-facing photo
2. **Left Side Photo** (Optional): Left profile view
3. **Right Side Photo** (Optional): Right profile view

### File Naming Convention
- Photos are automatically named: `{regno}_{side}.jpg`
- Example: `99220041142_front.jpg`, `99220041142_left.jpg`

## 🎥 Video Analysis Features

### Detection Capabilities
- **Face Detection**: Automatic detection of all faces in video
- **Student Recognition**: Matching with registered student photos
- **Real-time Processing**: Frame-by-frame analysis
- **Visual Indicators**: Green boxes around detected faces

### Analysis Results
- **Detection Summary**: Total faces, identified students, unknown persons
- **Detailed Log**: Frame-by-frame detection results
- **Interactive Elements**: Hover for details, click for profiles
- **Export Options**: View results in organized tables

## 🔗 Navigation & Links

### Student Profile Links
- Direct links from video analysis to student profiles
- Hyperlinks to student portals (if available)
- Seamless navigation between different views

### Dashboard Navigation
- Faculty dashboard with all upload and analysis tools
- Student dashboard for personal information management
- Easy navigation between different sections

## 🎨 User Interface Features

### Modern Design
- Bootstrap 5 responsive design
- Gradient backgrounds and modern styling
- Interactive elements with hover effects
- Mobile-friendly responsive layout

### Interactive Elements
- Hover effects on face detection boxes
- Clickable student profiles
- Expandable forms and sections
- Real-time feedback and notifications

## 🔒 Security Features

### Authentication
- Session-based authentication
- Role-based access control
- Secure password handling
- Protected routes and endpoints

### Data Protection
- Secure file uploads
- Input validation and sanitization
- SQL injection prevention
- Secure file storage

## 📊 Performance Optimization

### Face Recognition
- Optimized face detection algorithms
- Efficient photo encoding storage
- Frame sampling for video processing
- Caching mechanisms for better performance

### Video Processing
- Configurable frame sampling rates
- Memory-efficient video handling
- Background processing capabilities
- Progress tracking for long videos

## 🚨 Troubleshooting

### Common Issues
1. **Face Recognition Not Working**: Ensure photos are clear and well-lit
2. **Video Upload Fails**: Check file format and size limits
3. **Login Issues**: Verify credentials and database connection
4. **Performance Issues**: Check system resources and video quality

### Support
- Check system requirements
- Verify all dependencies are installed
- Review error logs in console
- Ensure proper file permissions

## 🔮 Future Enhancements

### Planned Features
- Real-time video streaming analysis
- Advanced gait recognition
- Mobile application support
- Cloud storage integration
- Advanced analytics dashboard

### Scalability
- Database optimization for large datasets
- Distributed processing capabilities
- API endpoints for external integration
- Multi-language support

## 📝 License

This project is developed for educational purposes. Please ensure compliance with local privacy and data protection laws when using this system.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows project standards
- Tests are included for new features
- Documentation is updated
- Security considerations are addressed

---

**Note**: This system is designed for educational institutions and should be used in compliance with privacy laws and institutional policies.
