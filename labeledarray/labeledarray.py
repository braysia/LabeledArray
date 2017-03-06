import numpy as np
from collections import OrderedDict


class LabeledArray(np.ndarray):
    """
    Each rows corresponds to labels, each columns corresponds to cells.
    Underlying data structure can be N-dimensional array. First dimension will be used for labeled array.
    
    Examples:
        >> arr = np.arange(12).reshape((3, 2, 2))
        >> labelarr = np.array([['a1' ,'b1', ''], 
                                ['a1' ,'b2' , 'c1'], 
                                ['a1' ,'b2' , 'c2']], dtype=object)
        >> darr = DArray(arr, labelarr)
        >> assert darr['a1'].shape
        (3, 2, 2)
        >> darr['a1', 'b1'].shape
        (2, 2)
        >> darr['a1', 'b2', 'c1']
        DArray([[4, 5],
               [6, 7]])
    """

    idx = None
    label = None
    
    def __new__(cls, arr=None, label=None, idx=None):
        obj = np.asarray(arr).view(cls)
        obj.label = label
        obj.idx = idx
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.label = getattr(obj, 'label', None)
        if hasattr(obj, 'idx') and np.any(self.label) and self.ndim > 1:
            self.label = self.label[obj.idx]
            if isinstance(self.label, str):
                return
            if self.label.ndim > 1:
                f_leftshift = lambda a1:all(x>=y for x, y in zip(a1, a1[1:]))
                all_column = np.all(self.label == self.label[0,:], axis=0)
                sl = 0 if not f_leftshift(all_column) else all_column.sum()
                self.label = self.label[:, slice(sl, None)]


    def __getitem__(self, item):
        if isinstance(item, str):
            item = self._label2idx(item)
        if isinstance(item, tuple):
            if isinstance(item[0], str):
                item = self._label2idx(item)
        self.idx = item
        ret = super(LabeledArray, self).__getitem__(item)
        return ret.squeeze()

    def _label2idx(self, item):
        item = (item, ) if not isinstance(item, tuple) else item
        boolarr = np.ones(self.label.shape[0], dtype=bool)
        for num, it in enumerate(item):
            boolarr = boolarr * (self.label[:, num]==it)
        return np.where(boolarr)

    def vstack(self, larr):
        return LabeledArray(np.vstack((self, larr)), np.vstack((self.label, larr.label)))

    def hstack(self, larr):
        if (self.label == larr.label).all():
            return LabeledArray(np.hstack((self, larr)), self.label)

    def save(self, file_name):
        extra_fields = set(dir(self)).difference(set(dir(LabeledArray)))
        data = dict(arr=self, label=self.label)
        for ef in extra_fields:
            data[ef] = getattr(self, ef)
        np.savez_compressed(file_name, **data)

    def load(self, file_name):
        if not file_name.endswith('.npz'):
            file_name = file_name + '.npz'
        f = np.load(file_name)
        arr, label = f['arr'], f['label']
        la = LabeledArray(arr, label)
        for key, value in f.iteritems():
            if not ('arr' == key or 'label' == key):
                setattr(la, key, value)
        return la
        


if __name__ == "__main__":
    # Check 2D.
    arr = np.random.rand(3, 100)
    labelarr = np.array([['a1', 'b1', ''], 
                        ['a1' ,'b2' , 'c1'], 
                        ['a1' ,'b2' , 'c2']], dtype=object)
    darr = LabeledArray(arr, labelarr)
    assert darr['a1'].shape == (3, 100)
    assert darr['a1', 'b1'].shape == (100, )
    assert darr['a1', 'b2'].shape == (2, 100)
    assert darr['a1', 'b2', 'c1'].shape == (100, )

    # check 3D.
    arr = np.arange(12).reshape((3, 2, 2))
    labelarr = np.array([['a1' ,'b1', ''], 
                        ['a1' ,'b2' , 'c1'], 
                        ['a1' ,'b2' , 'c2']], dtype=object)
    darr = LabeledArray(arr, labelarr)
    assert darr['a1'].shape == (3, 2, 2)
    assert darr['a1', 'b1'].shape == (2, 2)
    assert darr['a1', 'b2'].shape == (2, 2, 2)
    assert darr['a1', 'b2', 'c1'].shape == (2, 2)
    assert darr.shape == (3, 2, 2)
    assert np.all(darr['a1', 'b2'].label == np.array([['c1'], ['c2']]))

    # can save and load extra fields. add "time" for example.
    darr.time = np.arange(darr.shape[-1])
    darr.save('test')
    cc = LabeledArray().load('test.npz')
    assert cc.time.shape == (2,)


