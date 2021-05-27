# Renders a 2D model into a PPM image
import sys
import numpy as np

class StackMatrix():
	def __init__(self, initial_stack = []):
		self.stack = initial_stack

	def push(self, matrix: np.array):
		'''
		Manda uma matriz para a pilha de matrizes
		'''

		self.stack.append(matrix)

	def pop(self) -> np.array:
		'''
		Retira o ultimo elemento da pilha
		Retorna:
			O ultimo elemento da pilha
		Raises:
			IndexError: Quando a pilha esta vazia
		'''

		try:
			return self.stack.pop(-1)

		except IndexError:
			print("Error: Try to pop from a empty matrix")

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

def apply_matrix(matrix, x, y, z):

	x1 = x*matrix[0][0] + y*matrix[0][1] + z*matrix[0][2] + matrix[0][3]
	y1 = x*matrix[1][0] + y*matrix[1][1] + z*matrix[1][2] + matrix[1][3]
	z1 = x*matrix[2][0] + y*matrix[2][1] + z*matrix[2][2] + matrix[2][3]
	w = x*matrix[3][0] + y*matrix[3][1] + z*matrix[3][2] + matrix[3][3]
	x = x1/w
	y = y1/w
	z = z1/w
	return round(x), round(y), round(z)

def big_brain(p1, p2, aux):
	x = abs((p1[0] - p2[0])/(2*p1[0]))
	y = abs((p1[1] - p2[1])/(2*p1[0]))
	z = abs((p1[2] - p2[2])/(2*p1[0]))
	if((x + y + z) == 1):
		return True
	elif((x + y + z) == 3):
		return False
	elif((x + y + z) == 2):
		return bool(int(aux))

def draw_line(vector, x0, y0, z0, x1, y1, z1, color):
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
		p2 = 2*dz - dx
		while (x0 != x1):
			x0 = x0 + x_inc
			if (p1 >= 0):
				y0 = y0 + y_inc
				p1 = p1 - 2*dx
			if (p2 >= 0):
				z0 = z0 + z_inc
				p2 = p2 - 2*dx
			p1 = p1 + 2*dy
			p2 = p2 + 2*dz
			#image[x1][y1][z1] = color
			vector.append(((x0, y0, z0), color))

	# Y-axis direction
	elif (dy >= dx and dy >= dz):
		p1 = 2*dx - dy
		p2 = 2*dz - dy
		while (y0 != y1):
			y0 = y0 + y_inc
			if (p1 >= 0):
				x0 = x0 + x_inc
				p1 = p1 - 2*dy
			if (p2 >= 0):
				z0 = z0 + z_inc
				p2 = p2 - 2*dy
			p1 = p1 + 2*dx
			p2 = p2 + 2*dz
			#image[x1][y1][z1] = color
			vector.append(((x0, y0, z0), color))

	# Z-axis direction
	else:
		p1 = 2*dy - dz
		p2 = 2*dx - dz
		while (z0 != z1):
			z0 = z0 + z_inc
			if (p1 >= 0):
				y0 = y0 + y_inc
				p1 = p1 - 2*dz
			if (p2 >= 0):
				x0 = x0 + x_inc
				p2 = p2 - 2*dz
			p1 = p1 + 2*dy
			p2 = p2 + 2*dx
			#image[x1][y1][z1] = color
			vector.append(((x0, y0, z0), color))

# ---------- Main routine ----------

# Parses and checks command-line arguments
if len(sys.argv)!=3:
	print("usage: python draw_2d_model.py <input.dat> <output.ppm>\n"
		  "	   interprets the drawing instructions in the input file and renders\n"
		  "	   the output in the NETPBM PPM format into output.ppm")
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

color_set = False
current_color = (0, 0, 0,)
matrixT = [[1, 0, 0, 0], [0, 1, 0, 0,], [0, 0, 1, 0,], [0, 0, 0, 1]]
matrixP = [[1, 0, 0, 0], [0, 1, 0, 0,], [0, 0, 1, 0,], [0, 0, 0, 1]]
vector = []
matrix_stack = StackMatrix()

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

#---------------------------------------------------------------------------------------------------------#
	if(color_set == True):
		color = current_color
	else:
		color = DEFAULT_COLOR
#---------------------------------------------------------------------------------------------------------#
	if command == 'c':
		# Limpa a imagem com uma nova cor de fundo
		check_parameters(CHANNELS_N)
		background_color = np.array(parameters, dtype=IMAGE_DTYPE)
		image[...] = background_color
		zbuffer[...] = ZBUFFER_BACKGROUND
#---------------------------------------------------------------------------------------------------------#
	elif command == 'C':
		# Define uma nova cor de caneta
		check_parameters(CHANNELS_N)
		current_color = np.array(parameters, dtype=IMAGE_DTYPE)
		if(not color_set):
			color_set = True
#---------------------------------------------------------------------------------------------------------#
	elif command == 'L':
		# Desenha uma linha entre dois ponts dados
		check_parameters(6)
		parameters = list(map(int, parameters))
		x_init, y_init, z_init = apply_matrix(matrixT, *parameters[:3])
		x_fin, y_fin, z_fin = apply_matrix(matrixT, *parameters[3:])
		draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
#---------------------------------------------------------------------------------------------------------#
	elif command == 'P':
		# Desenha uma polilinha entre N pontos dados
		parameters = list(map(int, parameters))
		num_param, parameters = (parameters[0] * 3), parameters[1:]
		check_parameters(num_param)
		for ind in range(0, num_param - 3, 3):
			x_init, y_init, z_init = apply_matrix(matrixT, parameters[ind], parameters[ind + 1], parameters[ind + 2])
			x_fin, y_fin, z_fin = apply_matrix(matrixT, parameters[ind + 3], parameters[ind + 4], parameters[ind + 5])
			draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
#---------------------------------------------------------------------------------------------------------#
	elif command == 'R':
		# Desenha uma região delimitada entre N pontos dados
		parameters = list(map(int, parameters))
		num_param, parameters = (parameters[0] * 3), parameters[1:]
		check_parameters(num_param)
		for ind in range(0, num_param - 3, 3):
			x_init, y_init, z_init = apply_matrix(matrixT, parameters[ind], parameters[ind + 1], parameters[ind + 2])
			x_fin, y_fin, z_fin = apply_matrix(matrixT, parameters[ind + 3], parameters[ind + 4], parameters[ind + 5])
			draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
		x_init, y_init, z_init = apply_matrix(matrixT, *parameters[:3])
		x_fin, y_fin, z_fin = apply_matrix(matrixT, *parameters[-3:])
		draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
#---------------------------------------------------------------------------------------------------------#
	elif command == 'm':
		# Multiplica a matriz de transformação atual por uma dada como entrada
		check_parameters(16)
		matrixT = matrixT.dot(np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4))
#---------------------------------------------------------------------------------------------------------#
	elif command == 'M':
		# Substitui a matriz de transformação atual por uma dada de entrada
		check_parameters(16)
		matrixT = np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4)
#---------------------------------------------------------------------------------------------------------#
	elif command == 'V':
		# Substitui a matriz de projeção atual por uma dada de entrada
		check_parameters(16)
		matrixP = np.array(parameters, dtype=MODEL_DTYPE).reshape(4, 4)
#---------------------------------------------------------------------------------------------------------#
	elif command == "PUSH":
		# Guarda a matriz de transformação atual em uma pilha
		matrix_stack.push(matrixT)
#---------------------------------------------------------------------------------------------------------#
	elif command == "POP":
		# Substitui a matriz de transformação atual pela matriz no topo da pilha
		matrixT = matrix_stack.pop()
#---------------------------------------------------------------------------------------------------------#
	elif command == "CUB":
		# Desenha um cubo com ou sem diagonais, com dimensões referentes ao tamanho do lado dado de entrada
		check_parameters(2)
		r = int(parameters[0])//2
		points = [(r, r, r), (-r, r, r), (r, -r, r), (-r, -r, r), (r, r, -r), (-r, r, -r), (r, -r, -r), (-r, -r, -r)]
		for i in range(7):
			for u in range(i+1, 8):
				if(big_brain(points[i], points[u], parameters[1]) == True):
					x_init, y_init, z_init = apply_matrix(matrixT, points[i][0], points[i][1], points[i][2])
					x_fin, y_fin, z_fin = apply_matrix(matrixT, points[u][0], points[u][1], points[u][2])
					draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
#---------------------------------------------------------------------------------------------------------#
	elif command == "SPH":
		# Desenha uma esfera com N meridianos e M paralelos, com dimensões referentes ao raio dado de entrada
		check_parameters(3)
		sphere = []
		teta = (np.pi/int(parameters[1]))
		sphere.append([(0, 0, int(parameters[0]))])
		odd = False
		div = 2
		if(int(parameters[2])%2 == 1):
			odd = True
			div = 1
			parameters[2] = int(parameters[2]) - 1

		z_mov = (int(parameters[0])*2)//(int(parameters[2]) + 1)
		for t in range(1, (int(parameters[2])//2 + 1)):  # <- caso tenha mais pralelos do que deve ter.... remova o "+ 1" do for
			r_novo = np.sqrt((int(parameters[0])**2) - ((z_mov*(((int(parameters[2])//2) - t)) + z_mov//div)**2))
			sphere.append([])
			for k in range(int(parameters[1])*2):
				if(((int(parameters[2])//2 + 1) - t) == 1 and odd == False):
					# sphere.append(x, y, z)
					sphere[t].append((round(r_novo * np.cos(k*teta)), round(r_novo * np.sin(k*teta)), (z_mov//2)))
				else:
					sphere[t].append((round(r_novo * np.cos(k*teta)), round(r_novo * np.sin(k*teta)), (((int(parameters[2])//2 - t)) * z_mov) + (z_mov//div)))

		if(odd):
			sphere.append([])
			for k in range(int(parameters[1])*2):
				# sphere.append(x, y, z) <- caso haja paralelo na origem
				sphere[(int(parameters[2])//2 + 1)].append((round(int(parameters[0]) * np.cos(k*teta)) , round(int(parameters[0]) * np.sin(k*teta)), 0))

		for t in range(1, (int(parameters[2])//2)+1):  # <- caso tenha mais pralelos do que deve ter.... remova o "+ 1" do for
			#r_novo = np.sqrt(abs((int(parameters[0])**2) - (z_mov*t)**2))
			if(t == 1 and odd == False):
				r_novo = np.sqrt((int(parameters[0])**2) - ((z_mov/2)**2))
			else:
				r_novo = np.sqrt((int(parameters[0])**2) - ((z_mov*(t-1) + z_mov//div)**2))
			sphere.append([])
			for k in range(int(parameters[1])*2):
				if(t == 1 and odd == False):
					sphere[t+int(parameters[2])//2+(div%2)].append((round(r_novo * np.cos(k*teta)), round(r_novo * np.sin(k*teta)), -(z_mov//2)))
				else:
					sphere[t+int(parameters[2])//2+(div%2)].append((round(r_novo * np.cos(k*teta)), round(r_novo * np.sin(k*teta)), -((t-1) * z_mov) - (z_mov//div)))
		sphere.append([(0, 0, -int(parameters[0]))])
		# for a in range(len(sphere[1])-1):
		for b in range(len(sphere[1])):
			x_init, y_init, z_init = apply_matrix(matrixT, sphere[0][0][0], sphere[0][0][2], sphere[0][0][1])
			x_fin, y_fin, z_fin = apply_matrix(matrixT, sphere[1][b][0], sphere[1][b][2], sphere[1][b][1])
			draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
		for a in range(1,len(sphere)-2):
			for b in range(len(sphere[1])):
				x_init, y_init, z_init = apply_matrix(matrixT, sphere[a][b][0], sphere[a][b][2], sphere[a][b][1])
				x_fin, y_fin, z_fin = apply_matrix(matrixT, sphere[a+1][b][0], sphere[a+1][b][2], sphere[a+1][b][1])
				draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
		for b in range(len(sphere[1])):
			x_init, y_init, z_init = apply_matrix(matrixT, sphere[-1][0][0], sphere[-1][0][2], sphere[-1][0][1])
			x_fin, y_fin, z_fin = apply_matrix(matrixT, sphere[-2][b][0], sphere[-2][b][2], sphere[-2][b][1])
			draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
		for a in range(1,len(sphere)-1):
			for b in range(len(sphere[1])-1):
				x_init, y_init, z_init = apply_matrix(matrixT, sphere[a][b][0], sphere[a][b][2], sphere[a][b][1])
				x_fin, y_fin, z_fin = apply_matrix(matrixT, sphere[a][b+1][0], sphere[a][b+1][2], sphere[a][b+1][1])
				draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
			x_init, y_init, z_init = apply_matrix(matrixT, sphere[a][-1][0], sphere[a][-1][2], sphere[a][-1][1])
			x_fin, y_fin, z_fin = apply_matrix(matrixT, sphere[a][0][0], sphere[a][0][2], sphere[a][0][1])
			draw_line(vector, x_init, y_init, z_init, x_fin, y_fin, z_fin, color)
		#return
#---------------------------------------------------------------------------------------------------------#
	else:
		print(f'line {line_n}: unrecognized command "{command}"!', file=sys.stderr)
		sys.exit(1)
#---------------------------------------------------------------------------------------------------------#
# Aplica a matriz de projeção em todos os pontos computados pelo programa
for t in range(len(vector)):
	#print(vector[t][0])
	x, y, d = apply_matrix(matrixP, vector[t][0][0], vector[t][0][1], vector[t][0][2])
	y = y + round(height/2)
	x = x + round(width/2)
	if((x >= 0 and y >= 0) and (x < width and y < height)):
		if(zbuffer[height - y - 1][x] <= vector[t][0][2]):
			zbuffer[height - y - 1][x] = vector[t][0][2]
			image[height - y - 1][x] = vector[t][1]
#---------------------------------------------------------------------------------------------------------#
# If we reached this point, everything went well - outputs rendered image file
with open(output_file_name, 'wb') as output_file:
	save_ppm(image, output_file)
