import numpy
import tensorflow as tf
from tqdm import tqdm


class NeuralNet:
    design = []  # Слои нейронной сети в формате (количество нейронов, функция активации)
    if_compile = False  # Была ли НС скомпилированна
    x = None  # Placeholder для входных данных
    y = None  # Placeholder для обучающих выходных данных
    cost = None  # Переменная ценовой функции
    sess = None  # Сессия
    init = None  # Начальные значения
    tf_matrixes = None  # Матрицы слоев типа tf.Variable в формате (веса слоя, сдвиг) (см. self.__create_tf_matrixes)
    tf_layers = None  # Значения после каждого слоя (см self.compile)
    output = None  # Переменная выхода НС
    train_type = ""  # Тип обучения

    def __init__(self, min_element, max_element, amount_of_outputs):
        # Присваивание значений, сброс к начальным условиям
        self.min_element = min_element
        self.max_element = max_element
        self.amount_of_outputs = amount_of_outputs
        self.design = []
        self.tf_matrixes = []
        self.tf_layers = []

    def add(self, amount_of_units, activation_func):
        # Добавление еще одного слоя в НС
        # Слой добавляется в self.design в необходимом формате
        current_layer = (amount_of_units, activation_func)
        self.design.append(current_layer)

    def add_input_layer(self, amount_of_units):
        # Добавление входного слоя
        # Функция активации входного слоя нигде не используется, но указывается self.linear,
        # как наследие прошлой реализации, хотя можно использовать и None
        self.add(amount_of_units, self.linear)

    def __create_layer_matrixes(self, size):
        # Создаем numpy матрицы слоя
        weight_matrix = numpy.random.uniform(self.min_element, self.max_element, size)
        # Вектор сдвига (bias) должен иметь строк столько же, сколько матрица весов (weight_matrix), и один столбец
        # (см. "Broadcast")
        bias_vector = numpy.random.uniform(self.min_element, self.max_element, (size[0], 1))
        return weight_matrix, bias_vector

    def __create_tf_matrixes(self):
        # Создаем матрицы весов и вектора сдвигов типа tf.Variable
        # Каждый элемент списка tf_matrixes, который представляет из себя (матрица весов, вектор сдвига),
        # отвечает одному слою НС
        tf_matrixes = []
        for index in range(len(self.design) - 1):
            # Цикл идет до длины self.design - 1 потому что, нет ничего после выходного слоя
            current_layer_units = self.design[index][0]
            next_layer_units = self.design[index + 1][0]
            weight_matrix, bias_vector = self.__create_layer_matrixes((next_layer_units, current_layer_units))
            tf_weight_matrix = tf.Variable(weight_matrix, dtype=tf.double)
            tf_bias_vector = tf.Variable(bias_vector, dtype=tf.double)
            tf_matrixes.append((tf_weight_matrix, tf_bias_vector))
        return tf_matrixes

    # Просто тождественная функция
    @staticmethod
    def linear(x):
        return x

    def compile(self):
        # Здесь создается граф вычислений НС
        self.sess = tf.Session()  # Создание сессии
        if self.if_compile:  # Проверка, была ли скомпилированна НС ранее...
            # ...если да - сброс к начальным параметрам
            self.tf_matrixes = []
            self.tf_layers = []
            self.init = None
        else:
            self.add(self.amount_of_outputs, self.linear)  # Добавление выходного слоя
            # Выходной слой добавляется здесь, так как необходиом гарантировать, что он является последним
        self.if_compile = True
        self.x = tf.placeholder(tf.double)  # Создание placeholder для входных данных...
        self.y = tf.placeholder(tf.double)  # ...и обучающих выходов

        self.tf_matrixes = self.__create_tf_matrixes()  # Создание матриц
        self.init = tf.global_variables_initializer()  # Инициализатор переменных
        self.sess.run(self.init)  # Инициализация переменных
        for index, layer in enumerate(self.tf_matrixes):
            weight = layer[0]  # Матрица весов
            bias = layer[1]  # Вектор сдвига
            activation_func = self.design[index + 1][1]  # Функция активации берется из (!!!) СЛЕДУЮЩЕГО (!!!) слоя
            # Этот определенный "костыль", связан с тем, что функция активации является "входным" параметром для слоя =>
            # функция активации, действующая на данные "между" входным и первым слоем хранится в первом слое
            if index == 0:
                current_layer = activation_func(tf.matmul(weight, self.x) + bias)  # Если это первый слой, то действуем
                # на входные данные...
            else:
                prev_layer = self.tf_layers[index - 1]  # ...если нет - то получаем выход с предыдущего слоя...
                current_layer = activation_func(tf.matmul(weight, prev_layer) + bias)  # ...и действуем на него
            self.tf_layers.append(current_layer)
        self.output = self.tf_layers[-1]  # Выход нейронной сети - это последний слой => послдений элемент tf_layers

    def run(self, inputs):
        # Вычисление результата работы НС на выходных данных inputs
        # inputs = numpy.array([[...], ...])
        result = self.sess.run(self.output, {self.x: inputs})
        return result

    def train(self, train_type, x, y, batch_size, max_iters, alpha):
        # Обучение НС
        # Возвращает список cost_list - список значений ценовой функции на каждой итерации,
        # или -1, если произошла ошибка
        # train_type - тип обучения
        # x - входные данные; x = numpy.array([[...], ...])
        # y - обучающие выходные данные; y = numpy.array([[...], ...])
        # batch_size - размер batch (подвыборки), которая на каждой итерации обучения будет выбираться случайным образом
        # max_iters - количество итераций
        # alpha - шаг обучения
        if not self.if_compile:  # Проверка на то, что НС была скомпилировнна
            print('Compile model before calculate cost')
            return -1

        self.sess.run(self.init)  # Сброс матриц к начальным значениям
        self.train_type = train_type
        cost_list = []
        if train_type == 'prediction':  # Задача регрессии - апроксимация какой-либо функции
            self.__prediction_train(alpha, cost_list, batch_size, max_iters, x, y)
        return cost_list

    def continue_train(self, x, y, batch_size, max_iters, alpha):
        # Продолжение обучения
        # Используется тогда, когда цель обучения не была достигнута, но все параметры были выбраны хорошо
        # Олтчается от self.train только тем, что не произоддит сброса и вместо train_type используется self.train_type
        # Параметры - см. self.train
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1
        cost_list = []
        if self.train_type == 'prediction':
            self.__prediction_train(alpha, cost_list, batch_size, max_iters, x, y)
        return cost_list

    def __prediction_train(self, alpha, cost_list, batch_size, max_iters, x, y):
        # Фактически обучение НС
        # Параметры - см. self.train
        self.cost = tf.reduce_mean((self.output - self.y) ** 2)  # Вычисление ценовой функции
        optimizer = tf.train.GradientDescentOptimizer(alpha)  # Инициализация оптимизатора...
        train = optimizer.minimize(self.cost)
        # ...которым в данном случае является обычный градиенты спуск с шагом обучения alpha
        for _ in tqdm(range(max_iters)):
            batch_indexes = numpy.random.randint(0, x.shape[1], batch_size)  # Случайный выбор инедксов, из которых
            # будет состоять подвыборка на данном итерации обучения
            _, cur_cost = self.sess.run([train, self.cost], {self.x: x[:, batch_indexes], self.y: y[:, batch_indexes]})
            cost_list.append(cur_cost)

    def calculate_cost(self, x, y):
        # Вычисление ценовой функции
        # Параметры x и y - аналогично self.train
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1

        if self.cost is None:  # Проверка на то, что self.cost не None...
            self.cost = tf.reduce_mean((self.output - self.y) ** 2)  # ...если None - определяем self.cost

        return self.sess.run(self.cost, {self.x: x, self.y: y})  # Возвращаем вычисленное значение ценовой функции

    def calculate_cost_ex(self, x, y, tf_matrixes):
        # Функция, которая выисляет ценовую функцию, предназначенная для оптимизации параметров НС сторонними
        # оптимизаторами
        # Параметры НС self.tf_matrixes заменяются на параметры из tf_matrixes
        # Параметры x и y - аналогично self.train
        # tf_matrixes - список, формат которго аналогичен self.tf_matrixes
        # Матрица весов каждого слоя weight_matrix = numpy.array([[...], ...])
        # Вектор сдвига каждого слоя bias_vector = numpy.array([[...], ...])
        # Каждый элемент tf_matrixes = (weight_matrix, bias_vector)
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1
        tf_assign = []
        for index, layer in enumerate(self.tf_matrixes):
            tf_assign.append(layer[0].assign(tf_matrixes[index][0]))  # Данная конструкция создает и добавляет в список
            # TensorFlow "присваиватель", который далее запускается через self.sess.run,
            # после чего происходит присваивание
            tf_assign.append(layer[1].assign(tf_matrixes[index][1]))  # Аналогично
        self.sess.run(tf_assign)  # Присваивание значениям self.tf_matrixes tf_matrixes
        return self.calculate_cost(x, y)  # Возвращение вычисленной ценовой функции

    def init_params(self):
        # Инициализация начальных параметров
        self.sess.run(self.init)