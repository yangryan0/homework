def grid(height, width):
	grid = [[1 for i in range(width)] for _ in range(height)]
	print(grid)
	for i in range(1, height):
		for j in range(1, width):
			grid[i][j] = grid[i - 1][j] + grid[i][j - 1]
	return grid[height - 1][width - 1]

if __name__ == '__main__':
	print(grid(500, 500))
