from keras.engine import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K

 

class Attention(Layer):
    def __init__(self, step_dim, **kwargs):
        self.supports_masking = True

        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #self.add_weight()
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],), 
                               initializer='random_normal')
        self.features_dim = input_shape[-1]
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],), 
                               initializer='zeros')    
        #
        super(Attention, self).build(input_shape)
        #self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim