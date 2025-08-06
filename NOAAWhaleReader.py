import os
import csv
import pandas as pd
from BaseReader import BaseReader


class WhaleSpeciesReader(BaseReader):
    def __init__(self, data_path, species):
        self.species = species
        self.species_path = os.path.join(data_path, species)
        super().__init__(self.species_path)
        self.annotation_csv_path = os.path.join(
            self.species_path, "Processed", f"{species}_annotations_processed.csv"
        )

    def add_dataset_info(self):
        self.annotation_creator.add_dataset_info(
            description=f"{self.species} call dataset with annotations",
            version="1.0",
            year=2025
        )

    def add_categories(self):
        df = pd.DataFrame([{"id": 0, "name": self.species}])
        self.annotation_creator.add_categories(df)
        self.category_id = 0

    def add_sounds(self):
        print(f"Reading annotations from {self.annotation_csv_path}")
        if not os.path.exists(self.annotation_csv_path):
            print("Annotation CSV not found!")
            return  # Stop processing if annotations are not found
        self.audio_id_map = {}
        next_id = 0
        with open(self.annotation_csv_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["audiofile_path"]
                if rel_path not in self.audio_id_map:
                    duration = float(row["durationSeconds"])
                    sample_rate = int(row["sampleRate"])
                    self.audio_id_map[rel_path] = next_id
                    print("rel_path:", rel_path)
                    prefix = "DataInput_New" + os.sep
                    if rel_path.startswith(prefix):
                        file_name_path = rel_path[len(prefix):]
                    else:
                        file_name_path = rel_path

                    # Remove the species prefix since we're now working from the species directory
                    species_prefix = self.species + os.sep
                    if file_name_path.startswith(species_prefix):
                        file_name_path = file_name_path[len(species_prefix):]

                    # Now file_name_path will be like: Audio/... for all cases
                    self.annotation_creator.add_sound(
                        id=next_id,
                        file_name_path=file_name_path,
                        duration=duration,
                        sample_rate=sample_rate
                    )
                    next_id += 1

    def add_annotations(self):
        with open(self.annotation_csv_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for anno_id, row in enumerate(reader):
                path = row["audiofile_path"]
                sound_id = self.audio_id_map[path]
                t_min = float(row["startSeconds"])
                t_max = t_min + float(row["durationSeconds"])
                f_min = float(row["lowFreq"])
                f_max = float(row["highFreq"])
                self.annotation_creator.add_annotation(
                    anno_id=anno_id,
                    sound_id=sound_id,
                    category=self.category_id,
                    t_min=t_min,
                    t_max=t_max,
                    f_min=f_min,
                    f_max=f_max,
                    source_file=row.get("source_annotation_file", ""),
                    location=row.get("location", "")
                )
