# Generated by Django 3.0.3 on 2020-04-22 07:07

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tdse', '0005_auto_20200422_0705'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tdse',
            name='entropy',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), help_text='von Neumann entropy vs. time', size=None),
        ),
        migrations.AlterField(
            model_name='tdse',
            name='prob',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), help_text='Ising ground state probability vs. time', size=None),
        ),
        migrations.AlterField(
            model_name='tdse',
            name='time',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), help_text='Normalized time array for evolution', size=None),
        ),
    ]
