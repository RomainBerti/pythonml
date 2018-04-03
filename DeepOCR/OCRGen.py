import numpy as np
from numpy.random import randint, choice
import os
import string
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as mpl


def generate_block(min_char, max_char, size=(100, 100), origin=(0, 0)):
    """
    Generates a single image block. This is the image size the CNN uses
    returns image as a numpy array and string used to generate image

    """
    string_to_write = ''.join(choice(CHARSTRING, randint(min_char, max_char)))
    img = Image.new('RGB', size, "black")
    draw = ImageDraw.Draw(img)
    draw.text(origin, string_to_write, (255, 255, 255), font=loaded_font)
    return np.array(img), string_to_write


def generate_image(num_rows, num_cols, image_size, nb_min_char, nb_max_char, nb_images=128, dir_path='Training'):
    """
    num_rows:   Number of rows (blocks)
    num_cols:   Number of columns (blocks)
    image_size:   Image size
    nb_min_char: Minimum number of characters per line
    nb_max_char: Maximum number of characters per line
    nb_images: Number of images to generate
    dir_path:   Directory path to write images
    outputs are: images with the strings and csv file with the strings used to generate images
    """
    string_to_write = []
    nb_blocks = num_rows * num_cols # Number of blocks total = Rows * Cols
    for i in range(nb_images):               # Write images to ./Out/ directory
        image_to_write, string_i = merge_blocks(num_rows, num_cols, [generate_block(nb_min_char, nb_max_char, image_size) for _ in range(nb_blocks)])
        filename_i = os.path.join(dir_path, '{:05d}.png'.format(i))
        mpl.imsave(filename_i, image_to_write)
        string_to_write.append(filename_i + '/t' + string_i)
    with open(dir_path + '.csv', 'w') as csvfile:   # Write CSV file
        csvfile.write('\n'.join(string_to_write))


def get_font_size(nb_max_char):
    """
    Gets the maximum size of an image containing characters in loaded_font
    of maximum length nb_max_char
    return max image size (height , width) in pixels
    """
    img = Image.new('RGB', (1, 1), "black")
    draw = ImageDraw.Draw(img)
    im_height, im_width = 0, 0
    for char in CHARSTRING:  # Get max height and width possible characters
        tsi = draw.textsize(char * nb_max_char, font=loaded_font)
        im_width = max(tsi[1], im_width)
        im_height = max(tsi[0], im_height)
    return im_height, im_width


def merge_blocks(nb_rows, nb_cols, blocks_to_merge):
    """
    Merges blocks into combined images that are nb_rows blocks tall and nb_cols blocks wide
    nb_rows:  Number of rows (blocks)
    nb_cols:  Number of columns (blocks)
    T:   List of outputs from generate_block
    ret: Merged image, Merged string
    """
    B = np.array([t[0] for t in blocks_to_merge])
    Y = np.array([t[1] for t in blocks_to_merge])
    n, r, c, _ = B.shape
    return unblock(B, r * nb_rows, c * nb_cols), '@'.join(''.join(Yi) for Yi in Y.reshape(nb_rows, nb_cols))


def unblock(im_array, new_height, new_width):
    """
    im_array:   Array of shape (n, num_rows, num_cols, c)
    new_height:   Height of new array
    new_width:   Width of new array
    ret: Array of shape (new_height, new_width, c)
    """
    n, num_rows, num_cols, c = im_array.shape
    return im_array.reshape(new_height // num_rows, -1, num_rows, num_cols, c).swapaxes(1, 2).reshape(new_height, new_width, c)


if __name__ == "__main__":
    loaded_font = ImageFont.truetype('Arial.ttf', 18)
    # Possible characters to use
    CHARSTRING = list(string.ascii_letters + string.digits + ' ' + string.punctuation)
    nb_min_char, nb_max_char = 50, 64
    im_max_size = get_font_size(nb_max_char)
    print('CNN Image Size: ' + str(im_max_size))
    generate_image(1, 1, im_max_size, nb_min_char, nb_min_char, nb_images=32768, dir_path='../Training')  # Training data
    generate_image(4, 2, im_max_size, nb_min_char, nb_max_char, nb_images=256, dir_path='../Validation')  # Testing data
