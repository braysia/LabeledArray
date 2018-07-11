# LabeledArray

Numpy array subclass for indexing by strings.  
  
Using multi-index in pandas sometimes provides complications in terms of "copies vs views", especially when it has higher than 3 dimensions with complex multi labels. This array is to provide numpy.array's behavior and still enable to slice array by strings.

Underlying data can be 2D, 3D or N-dimensional array. Only the first dimension will be used for labels (multi-index).


### Example 1
| index | | | |
| -----|:---|:---|:---:|
| a A| 1 | 0 | 0 |
| a B| 0 | 1 | 0 |
| b A| 0 | 0 | 1 |

```
>> arr = np.eye(3)
>> labels = np.array([['a' ,'A'],['a' ,'B'],['b' ,'A']], dtype=object)
>> larr = LabeledArray(arr, labels)
>> print larr['a', 'B']
[0. 1. 0.]
>> print larr['a']
[[1. 0. 0.]
 [0. 1. 0.]]
```

### Example 2
```
>> arr = np.zeros((3, 20, 100))
>> labels = np.array([['nuc' ,'area', ''],
                   ['nuc' ,'FITC' , 'min_intensity'],
                   ['nuc' ,'FITC' , 'max_intensity']], dtype=object)
>> larr = LabeledArray(arr, labels)
>> print larr.shape
(3, 20, 100)
>> print larr['nuc', 'FITC'].shape
(2, 20, 100)
>> print larr['nuc', 'FITC', 'max_intensity'].shape
(20, 100)
```

The extra attributes including labels are automatically saved and loaded with the array. 
```
larr = LabeledArray(arr, labels)
larr.time = np.arange(arr.shape[-1])
larr.save('temp')
new_larr = LabeledArray().load('temp')
print new_larr.time
```
