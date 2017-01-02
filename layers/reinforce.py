from apollocaffe.layers.layer_headers import Layer, PyLayer, LossLayer
from IPython import embed

import numpy as np

class Index(PyLayer):
    def forward(self, bottom, top):
        data, indices = bottom
        index_data = indices.data.astype(int)
        top[0].reshape(indices.shape)
        top[0].data[...] = data.data[range(indices.shape[0]), index_data]

    def backward(self, top, bottom):
        data, indices = bottom
        index_data = indices.data.astype(int)
        data.diff[...] = 0
        data.diff[range(indices.shape[0]), index_data] = top[0].diff

class AsLoss(PyLayer):
    def __init__(self, name, **kwargs):
        PyLayer.__init__(self, name, dict(), **kwargs)

    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        top[0].data[...] = bottom[0].data

    def backward(self, top, bottom):
        bottom[0].diff[...] = top[0].data

class Reduction(LossLayer):
    def __init__(self, name, axis, **kwargs):
        kwargs["axis"] = axis
        super(Reduction, self).__init__(self, name, kwargs)

class PyL1Loss(PyLayer):
    def __init__(self, name, loss_weight=1, normalize=2, **kwargs):
        PyLayer.__init__(self, name, dict(), **kwargs)
        self.loss_weight = loss_weight
    def reshape(self, bottom, top):
        top[0].reshape((1,))
    def forward(self, bottom, top):
        top[0].reshape((1,))
        top[0].data[...] = self.loss_weight*np.sum(np.absolute(bottom[0].data))/bottom[0].shape[0]
    def backward(self, top, bottom):
        bottom[0].diff[...] += self.loss_weight*np.sign(bottom[0].data)/bottom[0].shape[0]

class PyL1LossWeighted(PyLayer):
    def __init__(self, name, dim=(1,1), sigma=1.0, loss_weight=1, normalize=2, **kwargs):
        PyLayer.__init__(self, name, dict(), **kwargs)
        self.loss_weight = loss_weight
        self.dim = dim
        self.sigma = sigma
        self.focus = None

    def reshape(self, bottom, top):
        top[0].reshape((1,))

    def forward(self, bottom, top):
        top[0].reshape((1,))
        if not self.focus:
            self.focus = self.get_filter()
        num, channel, height, width = bottom[0].shape
        focus = self.focus.reshape((1,1,height,width))
        focus = np.tile(focus, (num, channel, 1, 1))
    
        top[0].data[...] = self.loss_weight*np.sum(np.absolute(self.focus*bottom[0].data))/bottom[0].shape[0]

    def backward(self, top, bottom):
        bottom[0].diff[...] += self.loss_weight*self.focus*np.sign(bottom[0].data)/bottom[0].shape[0]

    def get_filter(self):
        focus = np.ones(self.dim)
        colc = (self.dim[0]-1)/2
        rowc = (self.dim[1]-1)/2
        for index, val in np.ndenumerate(focus):
            focus[index] = -((index[0] - colc)**2 + (index[1] - rowc)**2 ) / self.sigma**2
        focus = 1-np.exp(focus)
        return focus