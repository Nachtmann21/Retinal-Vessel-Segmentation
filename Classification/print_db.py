import os
from collections import defaultdict

# Path to the directory containing images
RIDB_PATH = "./RIDB"

def group_images_by_person(directory):
    person_images = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            # Extract person id and photo count from the filename
            try:
                parts = filename.split("_")
                photo_id = parts[0]  # Example: IM000001
                person_id = parts[1].split(".")[0]  # Example: 1
                person_images[person_id].append(filename)
            except IndexError:
                print(f"Filename format incorrect: {filename}")

    return person_images

def main():
    if not os.path.exists(RIDB_PATH):
        print(f"Path does not exist: {RIDB_PATH}")
        return

    grouped_images = group_images_by_person(RIDB_PATH)

    for person_id in sorted(grouped_images, key=lambda x: int(x)):
        print(f"Person ID {person_id}: {grouped_images[person_id]}")

if __name__ == "__main__":
    main()