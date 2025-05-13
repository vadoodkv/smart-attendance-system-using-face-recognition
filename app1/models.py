from django.db import models
from django.utils import timezone
from datetime import timedelta





class Student(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    phone_number = models.CharField(max_length=15)
    department = models.CharField(max_length=100)
    image = models.ImageField(upload_to='students/')
    authorized = models.BooleanField(default=False)
    

    def __str__(self):
        return self.name

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_1 = models.DateTimeField(null=True, blank=True)
    check_out_1 = models.DateTimeField(null=True, blank=True)
    check_in_2 = models.DateTimeField(null=True, blank=True)
    check_out_2 = models.DateTimeField(null=True, blank=True)
    check_in_3 = models.DateTimeField(null=True, blank=True)
    check_out_3 = models.DateTimeField(null=True, blank=True)
    check_in_4 = models.DateTimeField(null=True, blank=True)
    check_out_4 = models.DateTimeField(null=True, blank=True)

    # New field to store monthly total up to that date
    total_monthly_duration = models.DurationField(default=timedelta())
    
    
    #added extra
    def get_duration(self):
        total = timedelta()
        for ci, co in [
            (self.check_in_1, self.check_out_1),
            (self.check_in_2, self.check_out_2),
            (self.check_in_3, self.check_out_3),
            (self.check_in_4, self.check_out_4),
        ]:
            if ci and co:
                total += co - ci
        return total

    
    
    #def mark_checked_in(self, slot):
    #    setattr(self, f'check_in_{slot}', timezone.now())
    #    self.save()
        
    def mark_checked_in(self, slot):
        now = timezone.now()

        # Find the latest check-out before this slot
        previous_slot = slot - 1
        if previous_slot >= 1:
            last_checkout = getattr(self, f'check_out_{previous_slot}', None)
            if last_checkout:
                gap = (now - last_checkout).total_seconds()
                if gap < 60:
                     print(f"Check-in ignored: Must wait {60 - int(gap)} more seconds.")

        setattr(self, f'check_in_{slot}', now)
        self.save()


    def mark_checked_out(self, slot):
        setattr(self, f'check_out_{slot}', timezone.now())
        self.save()
        self.update_monthly_duration()      # Automatically update after check-out
        
    def calculate_duration(self):
        total = timedelta()
        for ci, co in [
            (self.check_in_1, self.check_out_1),
            (self.check_in_2, self.check_out_2),
            (self.check_in_3, self.check_out_3),
            (self.check_in_4, self.check_out_4),
        ]:
            if ci and co:
                total += co - ci
        hours, remainder = divmod(total.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m" if total else None
    
    
    def calculate_duration_in_hours(self):
        duration = timedelta()
        if self.check_in_1 and self.check_out_1:
            duration += self.check_out_1 - self.check_in_1
        if self.check_in_2 and self.check_out_2:
            duration += self.check_out_2 - self.check_in_2
        return duration.total_seconds() / 3600
    
    def update_monthly_duration(self):
        today = timezone.now().date()
        month_start = today.replace(day=1)
        attendances = Attendance.objects.filter(student=self.student, date__gte=month_start, date__lte=today)

        total_duration = timedelta()
        for att in attendances:
            total_duration += att.get_duration()  # This now returns timedelta

        self.total_monthly_duration = total_duration
        self.save(update_fields=['total_monthly_duration'])

        
    def get_readable_monthly_duration(self):
        total_seconds = int(self.total_monthly_duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"


    def save(self, *args, **kwargs):
        if not self.pk:  # Only on creation
            self.date = timezone.now().date()
        super().save(*args, **kwargs)




class Complaint(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()

    def __str__(self):
        return f"{self.student.name} - {self.subject}"
    
    
    
class LeaveRequest(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]
    
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    start_date = models.DateField()
    end_date = models.DateField()
    reason = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.name} - {self.start_date} to {self.end_date}"




class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(max_length=255, help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)")
    threshold = models.FloatField(default=0.6, help_text="Face recognition confidence threshold")

    def __str__(self):
        return self.name
