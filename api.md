!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="QjuEPvKDQL5W3V8qKRtu")
project = rf.workspace("kwame-nkrumah-university-of-science-and-technology").project("Illegal mining")
dataset = project.version(1).download("yolov8")