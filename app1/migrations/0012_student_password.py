# Generated by Django 5.1.7 on 2025-04-04 04:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0011_rename_student_class_student_department'),
    ]

    operations = [
        migrations.AddField(
            model_name='student',
            name='password',
            field=models.CharField(blank=True, max_length=128, null=True),
        ),
    ]
