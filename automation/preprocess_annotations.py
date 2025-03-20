import os

def preprocess_annotations(annotation_folder):
    for filename in os.listdir(annotation_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(annotation_folder, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            class_10_lines = []
            other_class_lines = []
            
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id == 10:
                    class_10_lines.append(line)
                else:
                    other_class_lines.append(line)
            
            # Write class 10 lines first, then other class lines
            with open(filepath, 'w') as file:
                file.writelines(class_10_lines + other_class_lines)

if __name__ == "__main__":
    annotation_folder = r"/d:/Programming/urine_interpret/Datasets/Final Dataset I think/labels"
    preprocess_annotations(annotation_folder)
    print("Annotation preprocessing completed.")
