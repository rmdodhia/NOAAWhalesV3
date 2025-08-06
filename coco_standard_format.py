import json
from datetime import datetime


class AnnotationCreator:
    """
    Creates annotations in a standardized format similar to COCO.
    """
    
    def __init__(self):
        self.data = {
            "info": {},
            "categories": [],
            "sounds": [],
            "annotations": []
        }
    
    def add_dataset_info(self, description="", version="1.0", contributor="", url="", year=None):
        """Add dataset metadata."""
        self.data["info"] = {
            "description": description,
            "version": version,
            "contributor": contributor,
            "url": url,
            "year": year or datetime.now().year,
            "date_created": datetime.now().isoformat()
        }
    
    def add_categories(self, categories_df):
        """Add categories from a DataFrame."""
        for _, row in categories_df.iterrows():
            category = {
                "id": int(row["id"]),
                "name": row["name"],
                "supercategory": row.get("supercategory", "")
            }
            self.data["categories"].append(category)
    
    def add_sound(self, id, file_name_path, duration, sample_rate, latitude=None, longitude=None, 
                  date_recorded=None, location=None):
        """Add a sound file."""
        sound = {
            "id": int(id),
            "file_name_path": file_name_path,
            "duration": float(duration),
            "sample_rate": int(sample_rate),
            "latitude": latitude,
            "longitude": longitude,
            "date_recorded": date_recorded,
            "location": location
        }
        self.data["sounds"].append(sound)
    
    def add_annotation(self, anno_id, sound_id, category, t_min, t_max, f_min=None, f_max=None, 
                      source_file="", location=""):
        """Add an annotation."""
        annotation = {
            "anno_id": int(anno_id),
            "sound_id": int(sound_id),
            "category": int(category),
            "t_min": float(t_min),
            "t_max": float(t_max),
            "f_min": float(f_min) if f_min is not None else None,
            "f_max": float(f_max) if f_max is not None else None,
            "source_file": source_file,
            "location": location
        }
        self.data["annotations"].append(annotation)
    
    def save_to_file(self, file_path):
        """Save the annotations to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"Annotations saved to: {file_path}")
