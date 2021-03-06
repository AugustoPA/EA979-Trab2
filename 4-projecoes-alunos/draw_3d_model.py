# Renders a 2D model into a PPM image
import sys
import numpy as np

# ---------- Configuration types and constants ----------

MAX_SIZE = 1024
MAX_VAL = 255
MAX_LINE_LEN = 10240-1 # 10240 characters minus the \0 terminator
DEFAULT_BACKGROUND = 255
CHANNELS_N = 3
COORD_N = 3
DEFAULT_COLOR = (0, 0, 0,)
IMAGE_DTYPE = np.uint8
VIEWPORT_DTYPE = np.int64
MODEL_DTYPE = np.float64
ZBUFFER_DTYPE = np.float64
ZBUFFER_BACKGROUND = -np.inf

# ---------- Output routines ----------

def put_string(output, output_file):
    output = output.encode('ascii') if isinstance(output, str) else output
    written_n = output_file.write(output)
    if written_n != len(output):
        print('error writing to output stream', file=sys.stderr)
        sys.exit(1)


def save_ppm(image, output_file):
    # Defines image header
    magic_number_1 = 'P'
    magic_number_2 = '6'
    width  = image.shape[1]
    height = image.shape[0]
    end_of_header = '\n'

    # Writes header
    put_string(magic_number_1, output_file)
    put_string(magic_number_2, output_file)
    put_string('\n', output_file)
    put_string('%d %d\n' % (width, height), output_file)
    put_string('%d' % MAX_VAL, output_file)
    put_string(end_of_header, output_file)

    # Outputs image
    put_string(image.tobytes(), output_file)

# ---------- Drawing/model routines ----------

def draw_line(image, x0, y0, z0, x1, y1, z1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    if (x1 > x0):
        x_inc = 1
    else:
        x_inc = -1

    if (y1 > y0):
        y_inc = 1
    else:
        y_inc = -1

    if (z1 > z0):
        z_inc = 1
    else:
        z_inc = -1

    # X-axis direction
    if (dx >= dy and dx >= dz):
        p1 = 2*dy - dx
        p2 = 2*dz = dx
        while (x1 != x2):
            x1 = x1 + x_inc
            if (p1 >= 0):
                y1 = y1 + y_inc
                p1 = p1 - 2*dx
            if (p2 >= 0):
                z1 = z1 + z_inc
                p2 = p2 - 2*dx
            p1 = p1 + 2*dy
            p2 = p2 + 2*dz
            image[x1][y1][z1] = color

    # Y-axis direction
    elif (dy >= dx and dy >= dz):
        p1 = 2*dx - dy
        p2 = 2*dz = dy
        while (y1 != y2):
            y1 = y1 + y_inc
            if (p1 >= 0):
                x1 = x1 + x_inc
                p1 = p1 - 2*dy
            if (p2 >= 0):
                z1 = z1 + z_inc
                p2 = p2 - 2*dy
            p1 = p1 + 2*dx
            p2 = p2 + 2*dz
            image[x1][y1][z1] = color

    # Y-axis direction
    else:
        p1 = 2*dy - dz
        p2 = 2*dx = dz
        while (z1 != z2):
            z1 = z1 + z_inc
            if (p1 >= 0):
                y1 = y1 + y_inc
                p1 = p1 - 2*dz
            if (p2 >= 0):
                x1 = x1 + x_inc
                p2 = p2 - 2*dx
            p1 = p1 + 2*dy
            p2 = p2 + 2*dx
            image[x1][y1][z1] = color
           
def draw_circle(r):
    points = []
    x = 0
    y = round(r)
    Delta = (5/4) - r
    Delta = round(Delta)
    dE = 2*x + 3
    dSE = 2*x - (2*y) + 5
    points.append((x, y))
    points.append((-x, y))
    points.append((x, -y))
    points.append((-x, -y))
    points.append((y, x))
    points.append((-y, x))
    points.append((y, -x))
    points.append((-y, -x))
    while (y >= x):
        if(Delta >= 0):
            Delta += dSE
            y -= 1
        else:
            Delta += dE
        dSE = 2*x - (2*y) + 5
        dE = 2*x + 3
        x += 1
        points.append((x, y))
        points.append((-x, y))
        points.append((x, -y))
        points.append((-x, -y))
        points.append((y, x))
        points.append((-y, x))
        points.append((y, -x))
        points.append((-y, -x))
    return points

# ---------- Main routine ----------

# Parses and checks command-line arguments
if len(sys.argv)!=3:
    print("usage: python draw_2d_model.py <input.dat> <output.ppm>\n"
          "       interprets the drawing instructions in the input file and renders\n"
          "       the output in the NETPBM PPM format into output.ppm")
    sys.exit(1)

input_file_name  = sys.argv[1]
output_file_name = sys.argv[2]

# Reads input file and parses its header
with open(input_file_name, 'rt', encoding='utf-8') as input_file:
    input_lines = input_file.readlines()

if input_lines[0] != 'EA979V4\n':
    print(f'input file format not recognized!', file=sys.stderr)
    sys.exit(1)

dimensions = input_lines[1].split()
width = int(dimensions[0])
height = int(dimensions[1])

if width<=0 or width>MAX_SIZE or height<=0 or height>MAX_SIZE:
    print(f'input file has invalid image dimensions: must be >0 and <={MAX_SIZE}!', file=sys.stderr)
    sys.exit(1)

# Creates image
image = np.full((height, width, CHANNELS_N), fill_value=DEFAULT_BACKGROUND, dtype=IMAGE_DTYPE)
zbuffer = np.full((height, width,), fill_value=ZBUFFER_BACKGROUND, dtype=ZBUFFER_DTYPE)

#
# TODO: Inicialize as demais variaveis
#


# Main loop - interprets and renders drawing commands
for line_n,line in enumerate(input_lines[2:], start=3):

    if len(line)>MAX_LINE_LEN:
        print(f'line {line_n}: line too long!', file=sys.stderr)
        sys.exit(1)

    if not line.strip():
        # Blank line - skips
        continue
    if line[0] == '#':
        # Comment line - skips
        continue

    tokens = line.strip().split()
    command = tokens[0]
    parameters = tokens[1:]
    def check_parameters(n):
        if len(parameters) != n:
            print(f'line {line_n}: command {command} expected {n} parameters but got {len(parameters)}!',
                  file=sys.stderr)
            sys.exit(1)

    if(color_set):
        color = current_color
    else:
        color = DEFAULT_COLOR

    if command == 'c':
        # Clears with new background color
        check_parameters(CHANNELS_N)
        background_color = np.array(parameters, dtype=IMAGE_DTYPE)
        image[...] = background_color
        zbuffer[...] = ZBUFFER_BACKGROUND

    elif command == 'C':
        check_parameters(CHANNELS_N)

        current_color = np.array(parameters, dtype=IMAGE_DTYPE)
        if(!color_set):
            color_set = True

    elif command == 'L':
        # Draws given line
        check_parameters(6)
        parameters = list(map(int, parameters))

        # Falta aplicar las matrices cabr??n
        draw_line(image, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], color)

    elif command == 'P':

        parameters = list(map(int, parameters))
        num_param, parameters = (parameters[0] * 3), parameters[1:]
        check_parameters(num_param)

        for ind in range(0, num_param - 3, 3):
            draw_line(image, parameters[ind], parameters[ind + 1], parameters[ind + 2], parameters[ind + 3], parameters[ind + 4], parameters[ind + 5], color)

    elif command == 'R':

        parameters = list(map(int, parameters))
        num_param, parameters = (parameters[0] * 3), parameters[1:]
        check_parameters(num_param)

        for ind in range(0, num_param - 3, 3):
            draw_line(image, parameters[ind], parameters[ind + 1], parameters[ind + 2], parameters[ind + 3], parameters[ind + 4], parameters[ind + 5], color)
        draw_line(image, parameters[0], parameters[1], parameters[2], parameters[-3], parameters[-2], parameters[-1], color)

    elif command == 'm':
        check_parameters(16)

        matrixT = matrixT.dot(np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4))

    elif command == 'M':
        check_parameters(16)

        matrixT = np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4)

    elif command == 'V':
        check_parameters(16)

        matrixP = np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4)

    else:
        print(f'line {line_n}: unrecognized command "{command}"!', file=sys.stderr)
        sys.exit(1)

# If we reached this point, everything went well - outputs rendered image file
with open(output_file_name, 'wb') as output_file:
    save_ppm(image, output_file)
