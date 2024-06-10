import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
rachel_image = face_recognition.load_image_file("jenniferaniston.webp")
rachel_face_encoding = face_recognition.face_encodings(rachel_image)[0]

# Load a second sample picture and learn how to recognize it.
ross_image = face_recognition.load_image_file("Ross.webp")
ross_face_encoding = face_recognition.face_encodings(ross_image)[0]
#For fun let us have marie biscuit image
chandler_image = face_recognition.load_image_file("chandler.webp")
chandler_face_encoding = face_recognition.face_encodings(chandler_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    rachel_face_encoding,
    ross_face_encoding,
    chandler_face_encoding
]
known_face_names = [
    "Rachel Green",
    "Ross Geller",
    "Chandler Bing"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("friendscast.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
print("1")
# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)
print("before for loop")
# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    print("before if")
    """ if True in matches:        
         first_match_index = matches.index(True)
         name = known_face_names[first_match_index]
         print(name) """

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        print(name)
    # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        font = ImageFont.truetype("arial.ttf", 36)

    # Draw a label with a name below the face
    """ text_width, text_height = draw.textsize(name,font)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
  """

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
pil_image.save("image_with_boxes.jpg")