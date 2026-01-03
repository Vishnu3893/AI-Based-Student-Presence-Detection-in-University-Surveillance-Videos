from functools import wraps
from flask import session, redirect, url_for, flash

def login_required(f):
    @wraps(f)
    def inner(*a, **kw):
        if "role" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return inner

def student_required(f):
    @wraps(f)
    def inner(*a, **kw):
        if session.get("role") != "student":
            flash("Student access only", "danger")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return inner

def faculty_required(f):
    @wraps(f)
    def inner(*a, **kw):
        if session.get("role") != "faculty":
            flash("Faculty access only", "danger")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return inner
