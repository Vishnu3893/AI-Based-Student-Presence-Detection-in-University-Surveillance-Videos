import sqlite3, os
import re
from datetime import datetime

class DataManager:
    def __init__(self, db_path="students_data/students.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def validate_reg_no(self, reg_no):
        """Validate registration number format for university students"""
        if not reg_no:
            return False, "Registration number is required"
        
        # Allow faculty accounts (e.g., PSBCSE)
        if reg_no.upper() == "PSBCSE":
            return True, ""
            
        # Check format: 11 digits format like 99220041142
        # Pattern: 99YYNNNXXXX where:
        # 99 = Fixed prefix
        # YY = enrollment year (e.g., 22)
        # NNN = Department code (e.g., 004)
        # XXXX = Student number (1142)
        pattern = r'^99[0-9]{2}[0-9]{3}[0-9]{4}$'
        if not re.match(pattern, reg_no):
            return False, "Invalid registration number format. Should be 11 digits starting with 99 (e.g., 99220041142)"
        
        # Validate ranges
        try:
            year = int(reg_no[2:4])  # YY from position 2-3
            dept = int(reg_no[4:7])  # NNN from position 4-6
            num = int(reg_no[7:])    # XXXX from position 7-10
            
            current_year = datetime.now().year % 100  # Get last 2 digits of current year
            if year > current_year or year < (current_year - 4):  # Allow up to 4 years old
                return False, f"Invalid year in registration number. Must be between {current_year-4} and {current_year}"
            
            if num < 1 or num > 9999:
                return False, "Student number must be between 0001 and 9999"
                
        except ValueError:
            return False, "Invalid number format in registration number"
            
        return True, ""

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            cur = c.cursor()
            
            # Create users table without is_first_login
            cur.execute("""
              CREATE TABLE IF NOT EXISTS users(
                  reg_no TEXT PRIMARY KEY,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL
              )""")
              
            cur.execute("""
              CREATE TABLE IF NOT EXISTS students(
                  reg_no TEXT PRIMARY KEY,
                  name TEXT,
                  dept TEXT,
                  room_no TEXT,
                  father_name TEXT,
                  father_phone TEXT
              )""")
              
            # seed default accounts and student data
            default_accounts = [
                # Users table data
                ("INSERT OR IGNORE INTO users(reg_no,password,role) VALUES(?,?,?)", [
                    ("99220041142", "student@123", "student"),
                    ("99220040138", "student@123", "student"),
                    ("99220040139", "student@123", "student"),
                    ("PSBCSE", "faculty@123", "faculty")
                ]),
                # Students table data
                ("INSERT OR IGNORE INTO students(reg_no,name,dept,room_no,father_name,father_phone) VALUES(?,?,?,?,?,?)", [
                    ("99220041142", "Demo Student", "CSE", "A-101", "Mr. Kumar", "9876543210"),
                    ("99220040138", "John Doe", "CSE", "B-205", "Mr. Doe", "9876543211"),
                    ("99220040139", "Jane Smith", "CSE", "C-303", "Mr. Smith", "9876543212")
                ])
            ]
            
            # Execute all inserts
            for query, values in default_accounts:
                for value in values:
                    try:
                        cur.execute(query, value)
                    except sqlite3.Error as e:
                        print(f"Error inserting data: {e}")
            
            c.commit()

    def verify_user(self, reg_no, password):
        # First validate the registration number format
        valid, message = self.validate_reg_no(reg_no)
        if not valid:
            return None, message
        
        with self._conn() as c:
            cur = c.cursor()
            # First check if user exists
            cur.execute("SELECT role, password FROM users WHERE reg_no=?", (reg_no,))
            r = cur.fetchone()
            
            if not r:
                # For new valid student registration numbers, create account automatically
                if reg_no.startswith("99") and len(reg_no) == 11:
                    try:
                        # Create new student account with default password
                        cur.execute("INSERT INTO users(reg_no, password, role) VALUES(?,?,?)",
                                  (reg_no, "student@123", "student"))
                        c.commit()
                        # If password matches default, let them in
                        if password == "student@123":
                            return "student", "Login successful"
                    except:
                        pass
                return None, "Invalid credentials. For new students, use registration number and password 'student@123'"
            
            stored_role, stored_password = r
            if password != stored_password:
                if password == "student@123":
                    return None, "Incorrect password. If you've changed your password, please use your new password."
                return None, "Incorrect password"
            
            return stored_role, "Login successful"

    def update_user_password(self, reg_no, old_password, new_password):
        with self._conn() as c:
            cur = c.cursor()
            cur.execute("SELECT password FROM users WHERE reg_no=?", (reg_no,))
            row = cur.fetchone()
            if not row:
                return False, "User not found"
                
            current_password = row[0]
            if current_password != old_password:
                return False, "Old password is incorrect"
                
            if new_password == "students@123":
                return False, "Cannot use the default password. Please choose a different password."
                
            if len(new_password) < 8:
                return False, "Password must be at least 8 characters long"
                
            # Update password and set is_first_login to 0
            cur.execute("""
                UPDATE users 
                SET password=?, is_first_login=0 
                WHERE reg_no=?
            """, (new_password, reg_no))
            c.commit()
            return True, "Password updated successfully"

    def create_new_student(self, reg_no, name, dept, room_no="", father_name="", father_phone=""):
        """Create a new student with default password"""
        valid, message = self.validate_reg_no(reg_no)
        if not valid:
            return False, message
            
        with self._conn() as c:
            cur = c.cursor()
            try:
                # First create the user account with default password "students@123"
                cur.execute("INSERT INTO users(reg_no,password,role,is_first_login) VALUES(?,?,?,?)",
                          (reg_no, "students@123", "student", 1))
                          
                # Then create student details
                cur.execute("""
                    INSERT INTO students(reg_no,name,dept,room_no,father_name,father_phone) 
                    VALUES(?,?,?,?,?,?)
                """, (reg_no, name, dept, room_no, father_name, father_phone))
                c.commit()
                return True, f"Student {reg_no} created successfully. Default password is 'students@123'"
            except sqlite3.IntegrityError:
                return False, "Registration number already exists"
            except Exception as e:
                return False, f"Error creating student: {str(e)}"

    def update_student(self, reg_no, name, dept, room_no, father_name, father_phone):
        # Validate registration number
        valid, message = self.validate_reg_no(reg_no)
        if not valid:
            return False, message

        with self._conn() as c:
            cur = c.cursor()
            try:
                # Make sure user exists in users table with default password
                cur.execute("SELECT 1 FROM users WHERE reg_no=?", (reg_no,))
                if not cur.fetchone() and reg_no != "PSBCSE":
                    # Create user with default password if doesn't exist
                    cur.execute("INSERT INTO users(reg_no,password,role) VALUES(?,?,?)",
                              (reg_no, "student@123", "student"))
                
                # Update student details
                cur.execute("""
                    INSERT INTO students(reg_no,name,dept,room_no,father_name,father_phone) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(reg_no) DO UPDATE SET
                    name=excluded.name, dept=excluded.dept, room_no=excluded.room_no,
                    father_name=excluded.father_name, father_phone=excluded.father_phone
                """, (reg_no, name, dept, room_no, father_name, father_phone))
                c.commit()
                return True, "Student details updated successfully"
            except Exception as e:
                return False, f"Error updating student: {str(e)}"

    def get_student(self, reg_no):
        """Get student details and create if doesn't exist"""
        with self._conn() as c:
            cur = c.cursor()
            # First check if the student exists
            cur.execute("""
                SELECT s.reg_no, s.name, s.dept, s.room_no, s.father_name, s.father_phone, u.role
                FROM students s
                LEFT JOIN users u ON s.reg_no = u.reg_no
                WHERE s.reg_no=?
            """, (reg_no,))
            r = cur.fetchone()
            
            if r:
                # Student exists, return their details
                return {
                    "reg_no": r[0],
                    "name": r[1] or '',
                    "dept": r[2] or '',
                    "room_no": r[3] or '',
                    "father_name": r[4] or '',
                    "father_phone": r[5] or '',
                    "role": r[6] or 'student'
                }
            
            # If student doesn't exist but reg_no is valid, create an empty record
            valid, _ = self.validate_reg_no(reg_no)
            if valid:
                try:
                    # Try to get any existing user record
                    cur.execute("SELECT role FROM users WHERE reg_no=?", (reg_no,))
                    user = cur.fetchone()
                    role = user[0] if user else 'student'
                    
                    # Create empty student record
                    cur.execute("""
                        INSERT OR IGNORE INTO students 
                        (reg_no, name, dept, room_no, father_name, father_phone)
                        VALUES (?, '', '', '', '', '')
                    """, (reg_no,))
                    
                    if not user:
                        # Create user account if doesn't exist
                        cur.execute("""
                            INSERT OR IGNORE INTO users (reg_no, password, role)
                            VALUES (?, 'student@123', 'student')
                        """, (reg_no,))
                    
                    c.commit()
                    return {
                        "reg_no": reg_no,
                        "name": '',
                        "dept": '',
                        "room_no": '',
                        "father_name": '',
                        "father_phone": '',
                        "role": role
                    }
                except sqlite3.Error:
                    pass
            return None

    def get_all_students(self):
        with self._conn() as c:
            cur = c.cursor()
            cur.execute("SELECT reg_no,name,dept,room_no,father_name,father_phone FROM students")
            rows = cur.fetchall()
            keys = ["reg_no","name","dept","room_no","father_name","father_phone"]
            return [dict(zip(keys, r)) for r in rows]

    def get_students_dict(self):
        return {s["reg_no"]: s for s in self.get_all_students()}

    def create_user(self, reg_no, password, role, name="", dept=""):
        """Create a new user account (student or faculty)"""
        with self._conn() as c:
            cur = c.cursor()
            try:
                # Check if user already exists
                cur.execute("SELECT 1 FROM users WHERE reg_no=?", (reg_no,))
                if cur.fetchone():
                    return False, "User already exists. Please login instead."
                
                # Create user account
                cur.execute("INSERT INTO users(reg_no, password, role) VALUES(?,?,?)",
                          (reg_no, password, role))
                
                # If student, also create student record
                if role == "student":
                    cur.execute("""
                        INSERT INTO students(reg_no, name, dept, room_no, father_name, father_phone) 
                        VALUES(?,?,?,?,?,?)
                    """, (reg_no, name, dept, "", "", ""))
                
                c.commit()
                return True, "Account created successfully"
            except sqlite3.IntegrityError:
                return False, "User already exists"
            except Exception as e:
                return False, f"Error creating account: {str(e)}"
