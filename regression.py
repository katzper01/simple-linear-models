import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def id(i):
    def f(v):
        return v.item((0, i))
    return f

def mult(i, j):
    def f(v):
        return v.item((0, i)) * v.item((0, j))
    return f

def square(i):
    def f(v):
        return v.item((0, i)) ** 2
    return f

def cube(i):
    def f(v):
        return v.item((0, i)) ** 3
    return f

def gauss(i, s):
    def f(v):
        return np.exp(-(v.item((0, i)) ** 2)/(s ** 2))
    return f

def sin(i, a, b):
    def f(v):
        return np.sin(v.item(0, i) * a + b)
    return f

basis_functions = [
    [id(0), id(1), id(2), id(3), id(4), id(5)],
    [square(0), square(1), square(2), square(3), square(4), square(5)],
    [cube(0), cube(1), cube(2), cube(3), cube(4), cube(5)],
    [mult(0, 0), mult(0, 1), mult(0, 2), mult(0, 3), mult(0, 4), mult(0, 5),
     mult(1, 1), mult(1, 2), mult(1, 3), mult(1, 4), mult(1, 5),
     mult(2, 2), mult(2, 3), mult(2, 4), mult(2, 5),
     mult(3, 3), mult(3, 4), mult(3, 5),
     mult(4, 4), mult(4, 5),
     mult(5, 5)
    ],
    [gauss(0, 0.2), gauss(1, 0.2), gauss(2, 0.2), gauss(3, 0.2), gauss(4, 0.2), gauss(5, 0.2)],
    [gauss(0, 1), gauss(1, 1), gauss(2, 1), gauss(3, 1), gauss(4, 1), gauss(5, 1)],
    [gauss(0, 3), gauss(1, 3), gauss(2, 3), gauss(3, 3), gauss(4, 3), gauss(5, 3)],
    [sin(0,  3.6, -0.2)],
    [sin(0,  3.6, -0.2), sin(5, 3.64, -0.4)],
    [sin(0, 3.6, -0.2), sin(4, 1.33, 1.76), sin(5, 3.64, -0.4)],
]

class LinearRegression:

    def __init__(self, X_train, y_train, X_ver, y_ver, X_test, y_test):
        self.input = np.array([X_train, X_ver, X_test])
        self.output = np.array([y_train, y_ver, y_test])
        self.X = np.copy(self.input)
        self.y = np.copy(self.output)
        self.results = [0] * len(basis_functions)
        self.optimal_reg = [0] * len(basis_functions)
        self.optimal_params = [[]] * len(basis_functions)
        self.results_log_message = [''] * len(basis_functions)
        self.optimal_theta = [[]] * len(basis_functions)
        self.intercept = np.mean(y_train)

    def average_square_cost(self, X, y, Intercept=False):
        theta0 = np.mean(self.y[0])
        if Intercept == False:
            theta0 = 0

        return ((X * self.theta + theta0 - y).T * (X * self.theta + theta0 - y)).item() / X.shape[0]

    def soft_threshold(self, c, l, a):
        if c < -l:
            return (c + l)/a
        elif c >  l:
            return (c - l)/a
        else: 
            return 0

    def coordinate_descent(self, X, y, l):
        theta = np.asmatrix([0] * X.shape[1], dtype=float).T
        n_steps = 20

        for i in range(0, n_steps):
            for j in range(0, theta.shape[0]):
                x_j = np.asmatrix(X[:,j])
                a = (2 * x_j.T * x_j).item()
                theta_j = theta.copy()
                theta_j.itemset((j, 0), 0)
                c = (2 * x_j.T * (y - X * theta_j)).item()
                theta.itemset((j, 0), self.soft_threshold(c, l, a))

        return theta

    def elastic_net_regression(self, X, y):
        I = np.identity(X.shape[1], dtype = float)
        n_lambdas = 50
        max_lambda = 10000
        lambdas = np.arange(0, max_lambda, max_lambda / n_lambdas)
        n_alphas = 10
        alphas = np.arange(0, 1, 1 / n_alphas)

        best_lambda = 0
        best_alpha = 0
        best_ver_cost = 10 ** 10

        for l in lambdas:
            for a in alphas:
                X_ext = np.concatenate((X, np.sqrt(l * a) * I))
                y_ext = np.concatenate((y, np.zeros(X.shape[1]).reshape(-1, 1)))
                self.theta = self.coordinate_descent(X_ext, y_ext, l)
                ver_cost = self.average_square_cost(self.X[1], self.y[1])

                if ver_cost < best_ver_cost:
                    best_ver_cost = ver_cost
                    best_lambda = l
                    best_alpha = a

        X_ext = np.concatenate((X, np.sqrt(best_lambda * best_alpha) * I))
        y_ext = np.concatenate((y, np.zeros(X.shape[1]).reshape(-1, 1)))
        self.theta = self.coordinate_descent(X_ext, y_ext, best_lambda)
        self.last_params = [best_lambda, best_alpha]

    def lasso_regression(self, X, y):
        n_lambdas = 1000
        max_lambda = 1000000
        lambdas = np.arange(0, max_lambda, max_lambda / n_lambdas)
        best_lambda = 0
        best_ver_cost = 10 ** 10

        for l in lambdas:
            self.theta = self.coordinate_descent(X, y, l)
            ver_cost = self.average_square_cost(self.X[1], self.y[1])

            if ver_cost < best_ver_cost:
                best_ver_cost = ver_cost
                best_lambda = l

        self.theta = self.coordinate_descent(X, y, best_lambda)
        self.last_params = [best_lambda]
        
    def ridge_regression(self, X, y):
        I = np.identity((X.T * X).shape[0], dtype = float)
        n_lambdas = 50000
        max_lambda = 10000
        lambdas = np.arange(0, max_lambda, max_lambda / n_lambdas)
        best_lambda = 0
        best_ver_cost = 10 ** 10

        for l in lambdas:
            self.theta = (X.T * X + l * I).I * X.T * y
            ver_cost = self.average_square_cost(self.X[1], self.y[1])
            if ver_cost < best_ver_cost:
                best_ver_cost = ver_cost
                best_lambda = l
        
        self.theta = (X.T * X + best_lambda * I).I * X.T * y
        self.last_params = [best_lambda]
            
    def apply_basis_functions(self, func):
        for k in range(0, 3):
            new_X = np.zeros((len(self.input[k]), len(func)), dtype=float)
            for i in range(0, new_X.shape[0]):
                for j in range(0, new_X.shape[1]):
                    new_X[i][j] = func[j](self.X[k][i:,])
            self.X[k] = np.asmatrix(np.copy(new_X), dtype=float)
        
    def standarize_data(self, X):
        v = np.asarray(np.var(X[0], axis = 0)).flatten()
        m = np.asarray(np.mean(X[0], axis = 0)).flatten()
        for k in range(0, 3):
            for j in range(0, X[k].shape[1]):
                X[k][:, j] = (X[k][:, j] - m[j]) / np.sqrt(v[j])
        
    def find_best_model(self):
        best_func = 0
        best_reg = 0
        best_result = 10 ** 10
        best_theta = []

        regularizations = [self.ridge_regression, self.lasso_regression, self.elastic_net_regression]
        
        for I in range(0, len(basis_functions)):
            for k in range(0, 3):
                self.X[k] = np.asmatrix(np.copy(self.input[k]), dtype=float)
                self.y[k] = np.asmatrix(np.copy(self.output[k]), dtype=float)

            self.apply_basis_functions(basis_functions[I])
            self.standarize_data(self.X)

            self.results[I] = 10 ** 10
            self.optimal_reg[I] = 0
            self.optimal_params[I] = []

            for i in range(0, 3):
                regularizations[i](self.X[0], self.y[0])
                cost = self.average_square_cost(self.X[1], self.y[1], Intercept=True)
                if cost < self.results[I]:
                    self.results[I] = cost
                    self.optimal_reg[I] = i
                    self.optimal_params[I] = np.copy(self.last_params)
                    self.optimal_theta[I] = np.copy(self.theta)

            if self.results[I] < best_result:
                best_func = I
                best_theta = self.optimal_theta[I]
                best_result = self.results[I]

        for k in range(0, 3):
            self.X[k] = np.asmatrix(np.copy(self.input[k]), dtype=float)
            self.y[k] = np.asmatrix(np.copy(self.output[k]), dtype=float)

        self.apply_basis_functions(basis_functions[best_func])
        self.standarize_data(self.X)
        self.theta = best_theta

    def construct_results_log_message(self, I):
        msg = str(I) + "\t\t"
        if self.optimal_reg[I] == 0:
            msg += "ridge\t\t"
        elif self.optimal_reg[I] == 1:
            msg += "lasso\t\t"
        else:
            msg += "elastic net\t\t"
        msg += str(self.optimal_params[I]) + "\t\t"
        msg += str(np.sqrt(self.results[I])) + "\t\t"
        msg += str(np.asarray(self.optimal_theta[I].T).flatten())
        return msg

    def doit(self, logs=False):
        self.standarize_data(self.input)
        self.find_best_model()
   
        if logs == True:
            print('[basis functions set,\toptimal regression,\toptimal regression parameters,\taverage square error,\ttheta]')
            for I in range(0, len(basis_functions)):
                print(self.construct_results_log_message(I))
            print(f'Intercept: {self.intercept}')

        return np.sqrt(self.average_square_cost(self.X[2], self.y[2], Intercept=True))

def train_test_split(data, train_frac, ver_frac):
    train_size = train_frac
    ver_size = ver_frac
    n = data.shape[0]
    m = int(train_size * n)
    h = int((train_size + ver_size) * n)
    return np.split(data, [m, h])

data = np.loadtxt("example.data", dtype = float)

RUNS = 5
train_fractions = np.array([0.01, 0.02, 0.03, 0.125, 0.625, 1])
sum_costs = np.zeros(len(train_fractions))

ax = plt.gca()
plt.xlabel("fraction of training set")
plt.ylabel("average square error on test set")
plt.title("Error as a function of training set fraction")

for run in range(0, RUNS):
    print(f'Run {run}')

    np.random.shuffle(data)
    train_data, ver_data, test_data = train_test_split(data, 0.6, 0.2)

    X, y = np.split(train_data, [6], axis = 1)
    X_ver, y_ver = np.split(ver_data, [6], axis = 1)
    X_test, y_test = np.split(test_data, [6], axis = 1)

    costs = np.zeros(len(train_fractions))
    
    for i in range(0, len(train_fractions)):
        print(f'{train_fractions[i]} of training set')
        n_data = int(train_fractions[i] * X.shape[0])
        X_train = np.copy(X[:n_data,])
        y_train = np.copy(y[:n_data,])
        reg = LinearRegression(X_train, y_train, np.copy(X_ver), np.copy(y_ver), np.copy(X_test), np.copy(y_test))
        costs[i] = reg.doit(logs=True)
        sum_costs[i] += costs[i]

    ax.plot(train_fractions, costs, "c--")

sum_costs /= RUNS
ax.plot(train_fractions, sum_costs, "r-")  
plt.show()
