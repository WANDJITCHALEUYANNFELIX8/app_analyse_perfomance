from models import Student
from database import *


def process_student(form_data):

    student = Student(
        age=int(form_data["age"]),
        sexe=form_data["sexe"],
        etude=float(form_data["etude"]),
        sommeil=float(form_data["sommeil"]),
        distraction=float(form_data["distraction"]),
        env=int(form_data["env"]),
        assiduite=float(form_data["assiduite"]),
        ponctualite=float(form_data["ponctualite"]),
        discipline=float(form_data["discipline"]),
        tache=float(form_data["tache"]),
        niveau=form_data["niveau"],
        moyenne=float(form_data["moyenne"])
    
    )

    insert_student(student.to_tuple())

    return student
    

    
    
