import sympy

class M(sympy.Matrix):
    def get(self, i, j):
        '''
        获得啊a[i][j]的值
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.get(0,2)
        3
        '''
        return self[i * self.cols + j]

    def change(self, i, j, key):
        '''
        修改啊a[i][j]的值
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.change(0,2,2)
        >>> A
        Matrix([
        [1, 2, 2],
        [0, 1, 0],
        [2, 3, 5]])
        '''
        self[i * self.cols + j] = key

    def adjoint(self):
        '''
        求伴随矩阵
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.adjoint()
        Matrix([
        [-5, -1,  3],
        [ 0,  1,  0],
        [ 2,  1, -1]])
        '''
        ans = M([])
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                t = M(self)
                t.row_del(i)
                t.col_del(j)
                row.append(-1 ** (i + j) * t.det())
            ans = ans.col_join(M([row]))
        return ans.T

    def partition_by_col(self):
        '''
        按列分块
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> (a0,a1,a2) = A.partition_by_col()
        >>> a2
        Matrix([
        [2],
        [0],
        [5]])
        >>> A.partition_by_col()[1]
        Matrix([
        [2],
        [1],
        [3]])
        '''
        return [self.col(i) for i in range(self.cols)]

    def partition_by_row(self):
        '''
        按行分块
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.partition_by_row()[0]
        Matrix([[1, 2, 3]])
        '''
        return [self.row(i) for i in range(self.rows)]

    def rcef(self):
        '''
        获得列简化梯形矩阵,返回一个元组，两个元素分别是列简化梯形矩阵和主1所在行数
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.rcef()
        (Matrix([
        [1,  0, 0],
        [0,  1, 0],
        [2, -1, 0]]), (0, 1))
        '''
        now = self.T
        T_rref = now.rref()
        return (T_rref[0].T, T_rref[1])

    def col_mli(self):
        '''
        获得列向量的一组极大无关组
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.col_mli()
        Matrix([
        [1, 2],
        [0, 1],
        [2, 3]])
        '''
        now_rref = self.rref()
        ans = M([])
        for index in now_rref[1]:
            ans = ans.row_join(self.col(index))
        return ans

    def bss(self):
        '''
        获得基本解组
        >>> A=M([
        ... [1,2,3],
        ... [0,1,0],
        ... [2,3,5]])
        >>> A.rref()
        (Matrix([
        [1, 0, 3],
        [0, 1, 0],
        [0, 0, 0]]), (0, 1))
        >>> A.bss()
        Matrix([
        [-3],
        [ 0],
        [ 1]])
        '''
        ans = M([])
        r = self.rank()
        not_one = [index for index in range(self.cols) if not index in self.rref()[1]]
        one = list(self.rref()[1])
        for i in range(len(not_one)):
            l = [0] * self.cols
            for j in range(r):
                l[one[j]] = -self.col(not_one[i])[j]
            l[not_one[i]] = 1
            ans = ans.row_join(M(l))
        return ans


def eye(n):
    return M(sympy.eye(n))

def zeros(r,c):
    return M(sympy.zeros(r, c))

def diag(*lst):
    return M(sympy.diag(*lst))

def symbol(x):
    return sympy.symbols(x)

def join(matrix_list):
    '''
    合并分块矩阵
    >>> A=diag(2,3,4)
    >>> B=M([
    ... [1,4,5],
    ... [3,6,1],
    ... [2,6,5]])
    >>> C=zeros(3,3)
    >>> D=eye(3)
    >>> E=join([
    ... [A,B],
    ... [C,D]])
    >>> E
    Matrix([
    [2, 0, 0, 1, 4, 5],
    [0, 3, 0, 3, 6, 1],
    [0, 0, 4, 2, 6, 5],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])
    '''
    ans = M([])
    if(type(matrix_list[0]) == M):
        for col in matrix_list:
            ans = ans.row_join(col)
        return ans
    for row in matrix_list:
        now_row = M([])
        for col in row:
            now_row = now_row.row_join(col)
        ans = ans.col_join(now_row)
    return ans
    
def EM(n,*l):
	'''
	产生初等矩阵，完成初等行列变换
	1.产生三阶初等阵E(0,2):交换第0,2行 | 交换第0,2列
	>>> EM(3,0,2)
	Matrix([
	[0, 0, 1],
	[0, 1, 0],
	[1, 0, 0]])
	2.产生三阶初等阵E(1(3)):将第1行乘以3 | 交换第0,2列
	    #注意python中一元元组需要在元素后面加逗号
	>>> EM(3,1,(3,))
	Matrix([
	[1, 0, 0],
	[0, 3, 0],
	[0, 0, 1]])
	3.产生三阶初等阵E(1,0(3)):将第0行乘以2加到第1行 | 将第1列乘以2加到第0行
	>>> EM(3,1,0,(2,))
	Matrix([
	[1, 0, 0],
	[2, 1, 0],
	[0, 0, 1]])
	
	
	下面是初等行列变换的例子
	>>> A
	Matrix([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]])
	1.交换A的第0,1列
	>>> A*EM(3,0,1)
	Matrix([
	[2, 1, 3],
	[5, 4, 6],
	[8, 7, 9]])
	2.将A的第0行乘以(-3)加到第2行上去
	>>> EM(3,2,0,(-3,))*A
	Matrix([
	[1, 2, 3],
	[4, 5, 6],
	[4, 2, 0]])
	3.将A的第1列乘以(-1)
	>>> A*EM(3,1,(-1,))
	Matrix([
	[1, -2, 3],
	[4, -5, 6],
	[7, -8, 9]])
	'''
	res = eye(n)
	if len(l) == 2:
		if type(l[1]) == tuple:
			res.change(l[0],l[0],l[1][0])
		else:
			i, j = l[0], l[1]
			res.change(i,i,0)
			res.change(j,j,0)
			res.change(i,j,1)
			res.change(j,i,1)
	elif type(l[2]) == tuple:
		res.change(l[0],l[1],l[2][0])
	return res


def print_help():
    help_information = '''
    1.定义矩阵: 
    >>> A=M([
    ... [1,2,3],
    ... [2,3,4],
    ... [3,4,5]])   #或者直接A=M([[1,2,3],[2,3,4],[3,4,5]])
    >>> B=M([
    ... [1,2,3],
    ... [-1,-5,4],
    ... [6,4,5]])

    2.加法: A + B
    >>> A + B
    Matrix([
    [2,  4,  6],
    [1, -2,  8],
    [9,  8, 10]])

    3.数乘: 5 * A
    >>> 5 * A
    Matrix([
    [ 5, 10, 15],
    [10, 15, 20],
    [15, 20, 25]])

    4.转置: A.T
    >>> A.T
    Matrix([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]])

    5.矩阵乘法: A * B
    >>> A * B
    Matrix([
    [17, 4, 26],
    [23, 5, 38],
    [29, 6, 50]])

    6.矩阵乘方: A ** 2 或 pow(A, 2)
    Matrix([
    [14, 20, 26],
    [20, 29, 38],
    [26, 38, 50]])

    7.逆矩阵: A ** -1 或 pow(A, -1)
    >>> B ** -1
    Matrix([
    [-41/95,   2/95, 23/95],
    [ 29/95, -13/95, -7/95],
    [ 26/95,   8/95, -3/95]])

    8.单位矩阵: m1 = eye(3)
    >>> eye(3)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    9.零矩阵: m2 = zeros(3, 4)
    >>> zeros(3,4)
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    10.一矩阵: m3 = ones(3, 4)
    >>> ones(3,4)
    Matrix([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]])

    11.对角矩阵: m4 = diag(1, 2, 3)
    >>> diag(1,2,3)
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    12.latex格式化输出latex(A)

    13.求行列式: A.det()
    >>> B.det()
    95

    14.行简化梯形矩阵,主1所在列数: 
    获得行简化梯形矩阵,返回一个元组，两个元素分别是行简化梯形矩阵和主1所在列数
    >>> A=M([
    ... [1,2,3],
    ... [0,1,0],
    ... [2,3,5]])
    >>> A.rref()
    (Matrix([
    [1, 0, 3],
    [0, 1, 0],
    [0, 0, 0]]), (0, 1))

    15.求秩 A.rank()
    >>> A.rank()
    2

    16.特征值,重根数: A.eigenvals()
    >>> A.eigenvals()
    {9/2 - sqrt(105)/2: 1, 9/2 + sqrt(105)/2: 1, 0: 1}

    17.特征根,重根数,特征向量: A.eigenvects()
    >>> A.eigenvects()
    [(0, 1, [Matrix([
    [ 1],
    [-2],
    [ 1]])]), (9/2 - sqrt(105)/2, 1, [Matrix([
    [(-245 + 23*sqrt(105))/(-455 + 45*sqrt(105))],
    [    (-40 + 4*sqrt(105))/(-55 + 5*sqrt(105))],
    [                                          1]])]), (9/2 + sqrt(105)/2, 1, [Matrix([
    [(23*sqrt(105) + 245)/(455 + 45*sqrt(105))],
    [    (40 + 4*sqrt(105))/(5*sqrt(105) + 55)],
    [                                        1]])])]


    18.对角化: A.diagonalize()
    >>> A.diagonalize()
    (Matrix([
    [ 1, -sqrt(105)/10 - 1/2, -1/2 + sqrt(105)/10],
    [-2,  1/4 - sqrt(105)/20,  1/4 + sqrt(105)/20],
    [ 1,                   1,                   1]]), Matrix([
    [0,                 0,                 0],
    [0, 9/2 - sqrt(105)/2,                 0],
    [0,                 0, 9/2 + sqrt(105)/2]]))

    19.定义矩阵多项式函数
    def f(x):
        return x**3 + 4 * x**2 -2 * x + 4 * x**0
    >>> def f(x):
    ...     return x**3 + 4 * x**2 -2 * x + 4 * x**0
    ...
    >>> f(A)
    Matrix([
    [190, 268, 350],
    [268, 393, 510],
    [350, 510, 674]])


    20.获得行数，列数: A.rows A.cols
    21.获得行向量，列向量: A.row(0) A.col(2)
    >>> A.row(0)
    Matrix([[1, 2, 3]])
    >>> A.col(2)
    Matrix([
    [3],
    [4],
    [5]])

    '''
    for lines in help_information.splitlines():
        print(lines)
