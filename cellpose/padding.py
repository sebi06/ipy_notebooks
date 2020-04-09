# padding example

# Du hast ein Bild mit Höhe height und Breite width und willst es auf input_height und input_width padden.
# Die paddings in Pixeln werden so berechnet:

import numpy as np

width = 800
height = 900
input_width = 1024
input_height = 1024
image2d = np.zeros((height, width))

rgb = True

if len(image2d.shape) == 2:
    image2d = image2d[..., np.newaxis]
    rgb = False

print(image2d.shape)

padding_height = input_height - height
padding_width = input_width - width
padding_left, padding_right = padding_width // 2, padding_width - padding_width // 2
padding_top, padding_bottom = padding_height // 2, padding_height - padding_height // 2


# Das Padding wird dann folgendermaßen auf das Bild(image) angewendet:

image2d_padded = np.pad(image2d, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), 'reflect')
print(image2d_padded.shape)

# Dann einfach die Prediction anwenden und das Padding vom Ergebnis wieder entfernen:

#prediction = prediction_padded[padding_top:input_height - padding_bottom, padding_left:input_width - padding_right]
image2d = image2d_padded[padding_top:input_height - padding_bottom, padding_left:input_width - padding_right]
print(image2d.shape)

if not rgb:
    image2d = np.squeeze(image2d, axis=2)


print(image2d.shape)
