from django.urls import path
from . import views


urlpatterns = [
    path('', views.main, name='main'),
    path('home/', views.home, name='home'),
    path('employee_dashboard/', views.employee_dashboard, name='employee_dashboard'),
    path('capture_student/', views.capture_student, name='capture_student'),
    path('employee_signup/',views.employee_signup, name='employee_signup'),
    path('employee_login/',views.employee_login, name='employee_login'),
    path('selfie-success/', views.selfie_success, name='selfie_success'),
    path('reports/download/', views.download_attendance_report, name='download_attendance_report'),
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('my_attendance_record/', views.my_attendance_record, name='my_attendance_record'),
    path('students/attendance/', views.student_attendance_list, name='student_attendance_list'),
    path('students/', views.student_list, name='student-list'),
    path('students/<int:pk>/', views.student_detail, name='student-detail'),
    path('students/<int:pk>/authorize/', views.student_authorize, name='student-authorize'),
    path('students/<int:pk>/delete/', views.student_delete, name='student-delete'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),

    path('submit_complaint/', views.submit_complaint, name='submit_complaint'),
    path('view_complaints/', views.view_complaints, name='view_complaints'),
    path('complaint_details/<int:pk>/', views.complaint_detail, name='complaint_detail'),

    path('export-attendance/', views.export_attendance, name='export_attendance'),
    
    
    path('submit-leave-request/', views.submit_leave_request, name='submit_leave_request'),
    path('view-leave-requests/', views.view_leave_requests, name='view_leave_requests'),
    path('leave-requests/<int:pk>/<str:status>/', views.update_leave_status, name='update_leave_status'),
    
    path('my-leave-requests/', views.my_leave_requests, name='my_leave_requests'),
    
    path('update-profile/', views.update_profile, name='update_profile'),

    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
]
    

