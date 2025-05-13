import os
import cv2
import numpy as np
import torch
import csv
import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.conf import settings
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
from django.utils.timezone import localtime
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
import json
from django.db import models  # Added missing models import
from collections import defaultdict
import calendar
from .forms import ComplaintForm,LeaveRequestForm,StudentUpdateForm
from .models import Student, Attendance, CameraConfiguration,Complaint,LeaveRequest
from django.core.mail import send_mail






# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized images
    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        known_image = cv2.imread(image_path)
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        encodings = detect_and_encode(known_image_rgb)
        if encodings:
            known_face_encodings.extend(encodings)
            known_face_names.append(student.name)

    return known_face_encodings, known_face_names

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# View for capturing student information and image
def capture_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        department = request.POST.get('department')
        image_data = request.POST.get('image_data')

        # Decode the base64 image data
        if image_data:
            header, encoded = image_data.split(',', 1)
            image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

            student = Student(
                name=name,
                email=email,
                phone_number=phone_number,
                department=department,
                image=image_file,
                authorized=False  
            )
            student.save()
            send_mail(
                subject='Welcome to Smart Attendance System',
                message=f'Hi {student.name},\n\nYour registration is successful. You will be  authorized by the admin soon! \n\n Have a nice day :)',
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[student.email],
                fail_silently=False,
            )

            return redirect('selfie_success')  
        
    return render(request, 'capture_student.html')


# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')


# This views for capturing studen faces and recognize
def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)  # Function to detect and encode face in frame

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    students = Student.objects.filter(name=name)
                                    if students.exists():
                                        student = students.first()

                                        # Attendance based on check-in and check-out logic
                                        attendance, created = Attendance.objects.get_or_create(student=student, date=datetime.now().date())
                                        if created:
                                            attendance.mark_checked_in(1)
                                            success_sound.play()
                                            cv2.putText(frame, f"{name}, checked in (1).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        else:
                                            now = timezone.now()
                                            time_gap = timedelta(seconds=60)  # adjust this if needed

                                            if not attendance.check_in_1:
                                                attendance.mark_checked_in(1)
                                                success_sound.play()
                                                cv2.putText(frame, f"{name}, checked in (1).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            elif not attendance.check_out_1:
                                                if now >= attendance.check_in_1 + time_gap:
                                                    attendance.mark_checked_out(1)
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out (1).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in (1).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                                            elif not attendance.check_in_2:
                                                attendance.mark_checked_in(2)
                                                success_sound.play()
                                                cv2.putText(frame, f"{name}, checked in (2).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            elif not attendance.check_out_2:
                                                if now >= attendance.check_in_2 + time_gap:
                                                    attendance.mark_checked_out(2)
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out (2).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in (2).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                                            elif not attendance.check_in_3:
                                                attendance.mark_checked_in(3)
                                                success_sound.play()
                                                cv2.putText(frame, f"{name}, checked in (3).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            elif not attendance.check_out_3:
                                                if now >= attendance.check_in_3 + time_gap:
                                                    attendance.mark_checked_out(3)
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out (3).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in (3).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                                            elif not attendance.check_in_4:
                                                attendance.mark_checked_in(4)
                                                success_sound.play()
                                                cv2.putText(frame, f"{name}, checked in (4).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            elif not attendance.check_out_4:
                                                if now >= attendance.check_in_4 + time_gap:
                                                    attendance.mark_checked_out(4)
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out (4).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in (4).", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                            else:
                                                cv2.putText(frame, f"{name}, attendance complete.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                                        

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  

    return redirect('student_attendance_list')


#this is for showing Attendance list
def student_attendance_list(request):
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    students = Student.objects.all()

    if search_query:
        students = students.filter(name__icontains=search_query)

    # Prepare the attendance data
    student_attendance_data = []

    for student in students:
        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student': student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,  
        'date_filter': date_filter       
    }
    return render(request, 'student_attendance_list.html', context)



#admin dashboard
def main(request):
    return render(request, 'main.html')


#leave request 

def submit_leave_request(request):
    student = Student.objects.get(name=request.user)
    
    if request.method == 'POST':
        form = LeaveRequestForm(request.POST)
        if form.is_valid():
            leave_request = form.save(commit=False)
            leave_request.student = student
            leave_request.save()
            messages.success(request, 'Leave request submitted successfully!')
            return redirect('submit_leave_request')
    else:
        form = LeaveRequestForm()

    return render(request, 'submit_leave_request.html', {'form': form})


def view_leave_requests(request):
    requests = LeaveRequest.objects.all().order_by('-created_at')
    return render(request, 'view_leave_requests.html', {'requests': requests})


def update_leave_status(request, pk, status):
    leave_request = get_object_or_404(LeaveRequest, pk=pk)
    leave_request.status = status
    leave_request.save()
    return redirect('view_leave_requests')


def update_profile(request):
    try:
        
        student = Student.objects.get(name=request.user)
    except Student.DoesNotExist:
        return redirect('employee_dashboard')

    if request.method == 'POST':
        form = StudentUpdateForm(request.POST, instance=student)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('update_profile')
    else:
        form = StudentUpdateForm(instance=student)

    return render(request, 'update_profile.html', {'form': form})


#############

def my_leave_requests(request):
    try:
        student = Student.objects.get(name=request.user)
        leave_requests = LeaveRequest.objects.filter(student=student).order_by('-created_at')
    except Student.DoesNotExist:
        return redirect('employee_dashboard')
    
    return render(request, 'my_leave_requests.html', {'leave_requests': leave_requests})




#admin report download
def download_attendance_report(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Attendance_report.csv"'
    response.write('\ufeff'.encode('utf8'))  # Write BOM for Excel
    writer = csv.writer(response)

    writer.writerow([
        'Student Name', 'Email', 'Phone Number', 'Department', 'Authorized',
        'Date',
        'Check In 1', 'Check Out 1',
        'Check In 2', 'Check Out 2',
        'Check In 3', 'Check Out 3',
        'Check In 4', 'Check Out 4',
        'Daily Duration', 'Total Monthly Duration'
    ])

    # Filter records for the current month
    today = timezone.now().date()
    month_start = today.replace(day=1)
    last_day = calendar.monthrange(today.year, today.month)[1]
    month_end = today.replace(day=last_day)

    # Group records per student
    student_records = defaultdict(list)
    for record in Attendance.objects.select_related('student').filter(date__range=(month_start, month_end)):
        student_records[record.student.id].append(record)

    # Write data
    for student_id, records in student_records.items():
        total_monthly = timedelta()

        for idx, record in enumerate(records):
            daily = timedelta()
            for ci, co in [
                (record.check_in_1, record.check_out_1),
                (record.check_in_2, record.check_out_2),
                (record.check_in_3, record.check_out_3),
                (record.check_in_4, record.check_out_4)
            ]:
                if ci and co:
                    daily += co - ci
                    total_monthly += co - ci

            # Format durations
            dd_hours, dd_remainder = divmod(daily.total_seconds(), 3600)
            dd_minutes, _ = divmod(dd_remainder, 60)
            daily_str = f"{int(dd_hours)}h {int(dd_minutes)}m" if daily else ''

            td_str = ''
            if idx == len(records) - 1:  # Only write total duration in last row
                td_seconds = int(total_monthly.total_seconds())
                td_hours, td_remainder = divmod(td_seconds, 3600)
                td_minutes, td_seconds = divmod(td_remainder, 60)
                td_str = f"{td_hours}h {td_minutes}m {td_seconds}s"

            writer.writerow([
                record.student.name,
                record.student.email,
                record.student.phone_number,
                record.student.department,
                'Yes' if record.student.authorized else 'No',
                record.date.strftime("%d/%m/%Y"),
                localtime(record.check_in_1).strftime("%I:%M %p") if record.check_in_1 else '',
                localtime(record.check_out_1).strftime("%I:%M %p") if record.check_out_1 else '',
                localtime(record.check_in_2).strftime("%I:%M %p") if record.check_in_2 else '',
                localtime(record.check_out_2).strftime("%I:%M %p") if record.check_out_2 else '',
                localtime(record.check_in_3).strftime("%I:%M %p") if record.check_in_3 else '',
                localtime(record.check_out_3).strftime("%I:%M %p") if record.check_out_3 else '',
                localtime(record.check_in_4).strftime("%I:%M %p") if record.check_in_4 else '',
                localtime(record.check_out_4).strftime("%I:%M %p") if record.check_out_4 else '',
                daily_str,
                td_str
            ])

    return response


#################



def submit_complaint(request):
    try:
        student = Student.objects.get(name=request.user)
    except Student.DoesNotExist:
        return redirect('login')

    if request.method == 'POST':
        form = ComplaintForm(request.POST)
        if form.is_valid():
            complaint = form.save(commit=False)
            complaint.student = student
            complaint.save()
            messages.success(request, 'Complaint submitted successfully!')
            return redirect('submit_complaint')
    else:
        form = ComplaintForm()

    return render(request, 'submit_complaint.html', {'form': form})


def view_complaints(request):
    complaints = Complaint.objects.all().order_by('-date')
    return render(request, 'view_complaints.html', {'complaints': complaints})


def complaint_detail(request, pk):
    complaint = get_object_or_404(Complaint, pk=pk)
    return render(request, 'complaint_detail.html', {'complaint': complaint})

#######################################################################



#Employee Attendance Report-Download

def export_attendance(request):
    try:
        student = Student.objects.get(name=request.user)
    except Student.DoesNotExist:
        return HttpResponse("Student not found", status=404)

    # Define date range for current month
    today = timezone.now().date()
    start_date = today.replace(day=1)
    end_date = today

    attendance_records = Attendance.objects.filter(
        student=student,
        date__range=(start_date, end_date)
    ).order_by('date')

    # CSV response
    response = HttpResponse(content_type='text/csv')
    filename = f"attendance_report_{today.strftime('%B_%Y')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)
    
    writer.writerow([
        'Name', 'Email', 'Phone Number', 'Department', 'Date',
        'Check-in 1', 'Check-out 1',
        'Check-in 2', 'Check-out 2',
        'Check-in 3', 'Check-out 3',
        'Check-in 4', 'Check-out 4',
        'Daily Duration'
    ])

    total_duration_month = timedelta()

    for record in attendance_records:
        daily_duration = timedelta()
        time_pairs = [
            (record.check_in_1, record.check_out_1),
            (record.check_in_2, record.check_out_2),
            (record.check_in_3, record.check_out_3),
            (record.check_in_4, record.check_out_4)
        ]

        row = [
            student.name,
            student.email,
            student.phone_number,
            student.department,
            record.date.strftime('%d/%m/%Y'),
        ]

        for check_in, check_out in time_pairs:
            row.append(localtime(check_in).strftime('%I:%M %p') if check_in else '')
            row.append(localtime(check_out).strftime('%I:%M %p') if check_out else '')


            if check_in and check_out:
                daily_duration += (check_out - check_in)

        total_duration_month += daily_duration
        hours, remainder = divmod(daily_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        row.append(f"{int(hours)}h {int(minutes)}m")
        writer.writerow(row)

    # Add a final row with total duration of the month
    writer.writerow([''] * 14 + [''])
    total_sec = int(total_duration_month.total_seconds())
    hours, remainder = divmod(total_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    writer.writerow([''] * 14 + [f"Total Monthly Duration: {hours}h {minutes}m {seconds}s"])
    
    return response



def my_attendance_record(request):
    try:
        student = Student.objects.get(name=request.user)
    except Student.DoesNotExist:
        return redirect('login')

    today = timezone.localtime(timezone.now()).date()
    current_month = today.month
    current_year = today.year

    # Get all attendance records for current month ordered by date
    attendance_records = Attendance.objects.filter(
        student=student,
        date__year=current_year,
        date__month=current_month
    ).order_by('date')  # Changed to ascending order for proper date sequence

    # Calculate statistics
    present_days = attendance_records.exclude(check_in_1__isnull=True).count()
    
    # Calculate total monthly hours using the model method
    total_hours = sum(
        record.calculate_duration_in_hours() 
        for record in attendance_records 
        if record.check_in_1
    )
    
    # Calculate average daily hours
    average_daily = total_hours / present_days if present_days > 0 else 0

    # Makes complete month's data including empty days
    month_start = today.replace(day=1)
    next_month = month_start.replace(day=28) + timedelta(days=4)  # Get first day of next month
    month_end = next_month - timedelta(days=next_month.day)
    
    month_days = []
    current_date = month_start
    while current_date <= month_end:
        record = attendance_records.filter(date=current_date).first()
        month_days.append({
            'date': current_date,
            'record': record
        })
        current_date += timedelta(days=1)

    # Last activity tracking
    last_activity = None
    last_activity_type = None
    all_activities = []
    for record in attendance_records:
        for i in range(1, 5):
            check_in = getattr(record, f'check_in_{i}')
            check_out = getattr(record, f'check_out_{i}')
            if check_in:
                all_activities.append(('check_in', i, check_in))
            if check_out:
                all_activities.append(('check_out', i, check_out))
    
    if all_activities:
        latest_activity = max(all_activities, key=lambda x: x[2])
        last_activity_type = f"{latest_activity[0]} {latest_activity[1]}"
        last_activity = latest_activity[2]

    # Prepare chart data for last 7 days
    seven_days_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        record = attendance_records.filter(date=day).first()
        seven_days_data.append({
            'date': day,
            'record': record
        })

    context = {
        'student': student,
        'month_days': month_days,
        'present_days': present_days,
        'current_month_duration': f"{total_hours:.1f}h",
        'average_daily_hours': f"{average_daily:.1f}h",
        'last_activity_time': timezone.localtime(last_activity).strftime('%I:%M %p') if last_activity else 'No activity',
        'last_activity_type': last_activity_type,
        'weekly_hours': json.dumps(get_weekly_hours(attendance_records, month_start)),
        'date_labels': json.dumps([d['date'].strftime('%d-%m') for d in seven_days_data]),
        'checkin_times': json.dumps([
            d['record'].check_in_1.hour + d['record'].check_in_1.minute/60 
            if d['record'] and d['record'].check_in_1 else None 
            for d in seven_days_data
        ]),
        'checkout_times': json.dumps([
            d['record'].check_out_4.hour + d['record'].check_out_4.minute/60 
            if d['record'] and d['record'].check_out_4 else None 
            for d in seven_days_data
        ]),
    }

    return render(request, 'my_attendance_record.html', context)

def get_weekly_hours(records, month_start):
    weeks = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    for record in records:
        week_num = (record.date.day - 1) // 7 + 1
        if week_num in weeks:
            weeks[week_num] += record.calculate_duration_in_hours()
    return [weeks.get(i, 0) for i in range(1, 6)]

def format_duration(hours):
    """Convert decimal hours to hours:minutes format"""
    hours = int(hours)
    minutes = int((hours * 60) % 60)
    return f"{hours}h {minutes}m"

#########################################################################################


def home(request):          
    total_students = Student.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_1__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_1__isnull=False).count()
    total_complaints = Complaint.objects.count()
    total_cameras = CameraConfiguration.objects.count()
    total_leave_request = LeaveRequest.objects.count()

    context = {
        'total_students': total_students,
        'total_attendance': total_attendance,
        'total_check_ins': total_check_ins,
        'total_check_outs': total_check_outs,
        'total_complaints': total_complaints,
        'total_cameras': total_cameras,
        'total_leave_request': total_leave_request,
    }
    return render(request, 'home.html', context)

def employee_signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        
        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists!")
            else:
                user = User.objects.create_user(username=username,password=password)
                user.save()
                messages.success(request, "Employee registered successfully!  You can now  Login !  ")
                return redirect("capture_student")
        else:
            messages.error(request, "Passwords do not match!")

    return render(request,'employee_signup.html')

def employee_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('employee_dashboard') 
        else:
            messages.error(request, "Invalid username or password")
    
    return render(request, 'employee_login.html')





def employee_dashboard(request):        
    student = Student.objects.get(name=request.user)
    attendance_records = Attendance.objects.filter(student=student).order_by('-date')  

    context = {
        'student': student,
        'attendance_records': attendance_records,
    }
    return render(request, 'employee_dashboard.html', context)






# Custom user pass test for admin access
#def is_admin(user):
#    return user.is_superuser

#@login_required
#@user_passes_test(is_admin)


def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students}) ##################


def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})



def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        messages.success(request, "Authorization has been updated successfully!")
        return redirect('student-detail', pk=pk)    
    
    return render(request, 'student_authorize.html', {'student': student})



# This views is for Deleting student
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list')  
    
    return render(request, 'student_delete_confirm.html', {'student': student})


# View function for user login
def user_login(request):
    
    if request.method == 'POST':
        
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

       
        if user is not None:
            
            login(request, user)
        
            return redirect('home')  
        else:
            
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  





# Function to handle the creation of a new camera configuration

def camera_config_create(request):
    if request.method == "POST":
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            return redirect('camera_config_list')

        except IntegrityError:
            messages.error(request, "A configuration with this name already exists.")
            return render(request, 'camera_config_form.html')

    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations


def camera_config_list(request):
    configs = CameraConfiguration.objects.all()
    return render(request, 'camera_config_list.html', {'configs': configs})




# UPDATE: Function to edit an existing camera configuration

def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    if request.method == "POST":
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        config.save()  

        return redirect('camera_config_list')  
    
    return render(request, 'camera_config_form.html', {'config': config})




# DELETE: Function to delete a camera configuration

def camera_config_delete(request, pk):
    config = get_object_or_404(CameraConfiguration, pk=pk)

    if request.method == "POST":
        config.delete()  
        return redirect('camera_config_list')

    return render(request, 'camera_config_delete.html', {'config': config})




