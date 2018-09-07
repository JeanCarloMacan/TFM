from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor
import tensorflow as tf


class ESNIPCell(
    rnn_cell_impl.RNNCell):  # la clase ESNIPCell vendria a ser como una subclase de rnn_cell_impl.RNNCell, esto ayuda a proporcionar el estado cero
    """Echo State Network Cell.

      Based on http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf
      Only the reservoir, the randomized recurrent layer, is modelled. The readout trainable layer
      which map reservoir output to the target output is not implemented by this cell,
      thus neither are feedback from readout to the reservoir (a quite common technique).

      Here a practical guide to use Echo State Networks:
      http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf

      Since at the moment TF doesn't provide a way to compute spectral radius
      of a matrix the echo state property necessary condition `max(eig(W)) < 1` is approximated
      scaling the norm 2 of the reservoir matrix which is an upper bound of the spectral radius.
      See https://en.wikipedia.org/wiki/Matrix_norm, the section on induced norms.

      Every `RNNCell` must have the properties below and implement `call` with
      the signature `(output, next_state) = call(input, state)`.  The optional
      third input argument, `scope`, is allowed for backwards compatibility
      purposes; but should be left off for new subclasses.
      This definition of cell differs from the definition used in the literature.
      In the literature, 'cell' refers to an object with a single scalar output.
      This definition refers to a horizontal array of such units.
      An RNN cell, in the most abstract setting, is anything that has
      a state and performs some operation that takes a matrix of inputs.

    """

    def __init__(self, num_units, wr2_scale=0.7, connectivity=0.3, leaky=1.0, activation=math_ops.tanh, scope=None,
                 input_size=1, dtype=tf.float32, mean=0., std=0.6, learning_rate=0.001,
                 win_init=init_ops.random_normal_initializer(),
                 # Los parametros input_size y dtype tratar de pasarlos desde fuera
                 wr_init=init_ops.random_normal_initializer(),
                 bias_init=init_ops.random_normal_initializer()):
        """Initialize the Echo State Network Cell.

        Args:
          num_units: Int or 0-D Int Tensor, the number of units in the reservoir
          wr2_scale: desired norm2 of reservoir weight matrix.
            `wr2_scale < 1` is a sufficient condition for echo state property.
          connectivity: connection probability between two reservoir units
          leaky: leaky parameter
          activation: activation function
          win_init: initializer for input weights
          wr_init: used to initialize reservoir weights before applying connectivity mask and scaling
          bias_init: initializer for biases
        """
        self._num_units = num_units
        self._leaky = leaky
        self._activation = activation
        self._input_size = input_size  # Ingresar como argumento cuando se instancia la clase ESN
        self._dtype = dtype
        self.__scope = scope
        self._mean = mean
        self._learning_rate = learning_rate
        self._std = std

        def _wr_initializer(shape, dtype, partition_info=None):
            wr = wr_init(shape, dtype=dtype)

            connectivity_mask = math_ops.cast(
                math_ops.less_equal(
                    random_ops.random_uniform(shape),
                    connectivity),
                dtype)

            wr = math_ops.multiply(wr, connectivity_mask)

            wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(wr)))

            is_norm_0 = math_ops.cast(math_ops.equal(wr_norm2, 0), dtype)

            wr = wr * wr2_scale / (wr_norm2 + 1 * is_norm_0)

            return wr

        self._win_initializer = win_init
        self._bias_initializer = bias_init
        self._wr_initializer = _wr_initializer


    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def get_states(self, y_prev, data_val):
        h_prev=y_prev[0:self._num_units]
        a=y_prev[self._num_units:2*self._num_units]
        b=y_prev[2*self._num_units:3*self._num_units]

        h_prev = tf.reshape(h_prev, [1, self._num_units])  # h_prev seria state
        data_val = tf.reshape(data_val,
                              [1, self._input_size])  # Es necesario xq h_prev y data_val llegan como (30,) y (3,)

        #with vs.variable_scope(self.__scope or type(self).__name__, reuse=tf.AUTO_REUSE):  # "ESNCell"
        #with vs.variable_scope('rnn/ESNIPCell', reuse=tf.AUTO_REUSE):
        win = vs.get_variable("InputMatrix", [self._input_size, self._num_units], dtype=self._dtype,
                              trainable=False, initializer=self._win_initializer)
        wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=self._dtype,
                             trainable=False, initializer=self._wr_initializer)

        diag_a = tf.diag(a)  # dim 30x30

        _wr = math_ops.matmul(diag_a, wr)  # dim 30x30
        _win = math_ops.matmul(win, diag_a)  # dim 3x30

        # h_prev/state comienza en ceros

        in_mat = array_ops.concat([data_val, h_prev], axis=1)  # concat[1x3,1x30] ==> inmat dim = 1x33
        weights_mat = array_ops.concat([_win, _wr], axis=0)  # concat[3x30,30x30] ==> weigmat dim = 33x30

        # Obtencion de nuestros valores x e y para IP
        x_a = math_ops.matmul(in_mat, weights_mat) + b  # dim 1x30 + 1x30

        y_ip = self._activation(x_a)  # dim 1x30

        # Reglas para el caso de tanh, Aquí se debe seleccionar las otras reglas en caso de que sea fermi

        g_b = -self._learning_rate * (-(self._mean / tf.pow(self._std, 2)) + (y_ip / tf.pow(self._std, 2)) * (
            2 * tf.pow(self._std, 2) + 1 - tf.pow(y_ip, 2) + self._mean * y_ip))
        g_a = (self._learning_rate / a) + g_b * x_a

        _a = a + g_a
        _b = b + g_b

        # Asignación de las variables a y b

        c=tf.squeeze(_a)  #Se le debe poner el assing aqui tambien?
        d=tf.squeeze(_b)

        return tf.concat([tf.squeeze(y_ip),c,d],axis=0)


    def optimizeIPscan(self, train_sequence):

        with vs.variable_scope('rnn/ESNIPCell',reuse=tf.AUTO_REUSE):

            # Parametros de IP, ganancia a y bias b
            a = vs.get_variable("GainVector", [self._num_units], dtype=self._dtype,
                                trainable=True, initializer=tf.ones_initializer())

            b = vs.get_variable("Bias", [self._num_units], dtype=self._dtype, trainable=True,
                                initializer=tf.zeros_initializer())

            initial_state = tf.zeros([self._num_units])

            initial_state = tf.concat([initial_state, a, b], axis=0)
            #data = tf.squeeze(train_sequence)
            ip_states = tf.scan(self.get_states, train_sequence, initializer=initial_state)

            states = ip_states[-1,:]  # No se si hacer esto esta bien (aunq esto no es una dynamic rnn, ni la red recurrente)
            h_final = states[0:self._num_units]
            c = a.assign(states[self._num_units:2*self._num_units])
            d = b.assign(states[2*self._num_units: 3*self._num_units])

            #Tomar el ultimo de states, tomar a y b y eso asignar

        return h_final,c,d


    def getSpectralRadius(self):
        with vs.variable_scope('rnn/ESNIPCell', reuse=tf.AUTO_REUSE):
            wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=self._dtype,
                                 trainable=False, initializer=self._wr_initializer)
            a = vs.get_variable("GainVector", [self._num_units], dtype=self._dtype,
                                trainable=True, initializer=tf.ones_initializer())
            b = vs.get_variable("Bias", [self._num_units], dtype=self._dtype, trainable=True,
                                initializer=tf.zeros_initializer())

            diag_a = tf.diag(a)

            _wr = math_ops.matmul(diag_a, wr)+b
            wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(_wr)))


            return wr_norm2

    def getReservoirWeights(self):
        with vs.variable_scope('rnn/ESNIPCell', reuse=tf.AUTO_REUSE):
            wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=self._dtype,
                                 trainable=False, initializer=self._wr_initializer)

            return wr



    def __call__(self, inputs, state, scope=None):
        """ Run one step of ESN Cell

            Args:
              inputs: `2-D Tensor` with shape `[batch_size x input_size]`.
              state: `2-D Tensor` with shape `[batch_size x self.state_size]`.
              scope: VariableScope for the created subgraph; defaults to class `ESNCell`.

            Returns:
              A tuple `(output, new_state)`, computed as
              `output = new_state = (1 - leaky) * state + leaky * activation(Win * input + Wr * state + B)`.

            Raises:
              ValueError: if `inputs` or `state` tensor size mismatch the previously provided dimension.
              """

        inputs = convert_to_tensor(inputs)
        input_size = inputs.get_shape().as_list()[1]
        dtype = inputs.dtype

        # Inicializamos el reservorio (wr), pesos de entrada (win), vector de ganancia (a) y vector de bias (b)

        with vs.variable_scope(self.__scope or type(self).__name__, reuse=tf.AUTO_REUSE):
            win = vs.get_variable("InputMatrix", [input_size, self._num_units], dtype=dtype,
                                  trainable=False, initializer=self._win_initializer)
            wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=dtype,
                                 trainable=False, initializer=self._wr_initializer)
            # Parametros de IP, ganancia a y bias b
            a = vs.get_variable("GainVector", [self._num_units], dtype=dtype,
                                trainable=True, initializer=tf.ones_initializer())

            b = vs.get_variable("Bias", [self._num_units], dtype=dtype, trainable=True,
                                initializer=tf.zeros_initializer())

            diag_a = tf.diag(a)

            _wr = math_ops.matmul(diag_a, wr)
            _win = math_ops.matmul(win, diag_a)  # Se invierte el orden para multiplicar

            in_mat = array_ops.concat([inputs, state], axis=1)
            weights_mat = array_ops.concat([_win, _wr], axis=0)  # Por invertir el orden arriba, esto se mantiene igual
            # bias anterior eliminado

            # Obtencion de nuestros valores x e y para IP
            x_a = math_ops.matmul(in_mat, weights_mat) + b
            y_ip = self._activation(x_a)

        #      output = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat)+b)

        # Reglas de actualizacion de a y b (como escoger la regla si es con fermi o tanh? con un if?)

        # Regla para el caso de tanh

        # g_b = -self._learning_rate * (-(self._mean / tf.pow(self._std, 2)) + (y_ip / tf.pow(self._std, 2)) * (
        # 2 * tf.pow(self._std, 2) + 1 - tf.pow(y_ip, 2) + self._mean * y_ip))
        # g_a = self._learning_rate / a + g_b * x_a

        # _a=a+g_a
        # _b=b+g_b

        # a.assign(tf.squeeze(_a))
        # b.assign(tf.squeeze(_b))

        return y_ip, y_ip
#   el return es output y new_state
