# forms.py
from django import forms
#from .models import UploadedImage
from .models import Complaint,LeaveRequest,Student


#class UploadImageForm(forms.ModelForm):
#    name = forms.CharField(label='Name', max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
#    image = forms.ImageField(label='Image', widget=forms.FileInput(attrs={'class': 'form-control-file'}))

#    class Meta:
#        model = UploadedImage
#        fields = ['name', 'image']


class ComplaintForm(forms.ModelForm):
    class Meta:
        model = Complaint
        fields = ['subject', 'message']
        widgets = {
            'subject': forms.TextInput(attrs={'placeholder': 'Enter complaint subject'}),
            'message': forms.Textarea(attrs={
                'placeholder': 'Describe your issue in detail', 
                'rows': 5,
                'style': 'resize: vertical;'  # Makes textarea vertically resizable
            }),
        }
        
        


class LeaveRequestForm(forms.ModelForm):
    class Meta:
        model = LeaveRequest
        fields = ['start_date', 'end_date', 'reason']
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
            'reason': forms.Textarea(attrs={'rows': 4}),
        }
        
        
class StudentUpdateForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = {'name','email','phone_number','department'}
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter your full name'}),
            'email' : forms.EmailInput(attrs={'placeholder' : 'Enter new email'}),
            'phone_number' : forms.TextInput(attrs={'placeholder' : 'Enter new phone number'}),
            'department' : forms.TextInput(attrs={'placeholder' : 'Enter new department'}),
        }

