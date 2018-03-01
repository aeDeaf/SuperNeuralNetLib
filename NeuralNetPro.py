import tensorflow as tf

import NeuralNet


class NeuralNetPro(NeuralNet.NeuralNet):
    # sess = None
    # tf_matrixes = []
    # tf_layers = []
    # init = None
    # if_compile = False
    # x = None
    # y = None
    placeholders_dict = {}

    def __init__(self, min_element, max_element, amount_of_outputs):
        super().__init__(min_element, max_element, amount_of_outputs)

    def __create_tf_placeholders(self):
        tf_matrixes = []
        for index in range(len(self.design) - 1):
            # Цикл идет до длины self.design - 1 потому что, нет ничего после выходного слоя
            current_layer_units = self.design[index][0]
            next_layer_units = self.design[index + 1][0]
            size = current_layer_units * next_layer_units  # Количество элементов матрицы
            weight_breaker = size + self.unroll_breaks[index][1]  # Индекс конца матрицы весов в unroll векторе -
            # ее размер, плюс сдвиг, связанный с предыдущими матрицами
            bias_breaker = weight_breaker + next_layer_units  # Аналогично
            self.unroll_breaks.append((weight_breaker, bias_breaker))
            tf_weight_matrix = tf.placeholder(tf.double)
            tf_bias_vector = tf.placeholder(tf.double)
            tf_matrixes.append((tf_weight_matrix, tf_bias_vector))
            self.size_list.append(((next_layer_units, current_layer_units), (next_layer_units, 1)))
        return tf_matrixes

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

        self.tf_matrixes = self.__create_tf_placeholders()  # Создание матриц
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

    def set_matrixes(self, matrixes):
        #print(matrixes)
        for index, layer in enumerate(self.tf_matrixes):
            self.placeholders_dict.update([(layer[0], matrixes[index][0]), (layer[1], matrixes[index][1])])

    def set_random_matrixes(self):
        matrixes = []
        for index in range(len(self.design) - 1):
            # Цикл идет до длины self.design - 1 потому что, нет ничего после выходного слоя
            current_layer_units = self.design[index][0]
            next_layer_units = self.design[index + 1][0]
            weight_matrix, bias_vector = self._create_layer_matrixes((next_layer_units, current_layer_units))
            matrixes.append((weight_matrix, bias_vector))
        self.set_matrixes(matrixes)

    def run(self, inputs):
        d = {self.x: inputs}
        d.update(self.placeholders_dict)
        result = self.sess.run(self.output, d)
        return result

    def calculate_cost(self, x, y):
        # Вычисление ценовой функции
        # Параметры x и y - аналогично self.train
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1

        if self.cost is None:  # Проверка на то, что self.cost не None...
            self.cost = tf.reduce_mean((self.output - self.y) ** 2)  # ...если None - определяем self.cost

        d = {self.x: x, self.y: y}
        d.update(self.placeholders_dict)

        return self.sess.run(self.cost, d)  # Возвращаем вычисленное значение ценовой функции

    def roll_matrixes(self, unroll_vector):
        # Противоположно self.unroll_matrixes, roll_matrixes сворачивает матрицы из вектора обратно в нормальный вид
        tf_matrixes = []
        #print(unroll_vector)
        for index, layer in enumerate(self.unroll_breaks[1:]):
            left_weight_break = self.unroll_breaks[index][1]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_weight_break = layer[0]  # Правая граница матрицы весов = левая граница вектора сдвига
            right_bias_break = layer[1]  # Правая граница вектора сдвига
            # Далее мы выделяем нужный нам фрагмент из развернутого вектра, и делаем его нужным размером
            weight_matrix = unroll_vector[left_weight_break:right_weight_break].reshape(self.size_list[index][0])
            bias_vector = unroll_vector[right_weight_break:right_bias_break].reshape(self.size_list[index][1])
            tf_matrixes.append((weight_matrix, bias_vector))
        #print(tf_matrixes)
        self.set_matrixes(tf_matrixes)
        return tf_matrixes
