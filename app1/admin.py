from django.contrib import admin
from .models import Student, Attendance,CameraConfiguration,Complaint,LeaveRequest

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone_number', 'department', 'authorized']
    list_filter = ['department', 'authorized']
    search_fields = ['name', 'email']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'check_in_1', 'check_out_1', 'check_in_2', 'check_out_2','check_in_3', 'check_out_3','check_in_4', 'check_out_4']
    list_filter = ['date']
    search_fields = ['student__name']

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return ['student', 'date', 'check_in_1', 'check_out_1','check_in_3', 'check_out_3','check_in_4', 'check_out_4',]
        else:  # Adding a new object
            return ['date', 'check_in_1', 'check_out_1']

    def save_model(self, request, obj, form, change):
        if change:  # Editing an existing object
            # Ensure check-in and check-out times cannot be modified via admin
            obj.check_in_1 = Attendance.objects.get(id=obj.id).check_in_1
            obj.check_out_1 = Attendance.objects.get(id=obj.id).check_out_1
        super().save_model(request, obj, form, change)


@admin.register(CameraConfiguration)
class CameraConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_source', 'threshold']
    search_fields = ['name']
    
    
@admin.register(Complaint)
class ComplaintAdmin(admin.ModelAdmin):
    list_display = ['date', 'subject', 'message', ]
    list_filter = ['date']
    search_fields = ['name']    
    
 
    
@admin.register(LeaveRequest)
class LeaveRequestAdmin(admin.ModelAdmin):
    list_display = ['start_date', 'end_date','created_at', 'status']
    list_filter = ['created_at']
    search_fields = ['name']    
    