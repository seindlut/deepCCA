
import numpy as np
import leveldb
import random
import caffe

def write_leveldb(set, labels, name=''):
    leveldb.DestroyDB('../data/'+name)
    db = leveldb.LevelDB('../data/'+name, create_if_missing=True, error_if_exists=True, write_buffer_size=268435456)
    wb = leveldb.WriteBatch()
    num_items = set.shape[0]
    print '# samples : ', num_items
    set = set.reshape(num_items, 28, 28)
    count = 0
    for i in range(num_items):
        image = np.expand_dims(set[i,:,:], axis=0)
        label = labels[i]
        # Load image into datum object
        datum = caffe.io.array_to_datum(image, int(label))
        wb.Put('%08d_%s' % (count, file), datum.SerializeToString())
        count = count + 1
        if count % 1000 == 0:
            # Write batch of images to database
            db.Write(wb)
            del wb
            wb = leveldb.WriteBatch()
            print 'Processed %i images.' % count

    if count % 1000 != 0:
        # Write last batch of images
        db.Write(wb)
        print 'Processed a total of %i images.' % count
    else:
        print 'Processed a total of %i images.' % count


if __name__ == '__main__':
    n = 10000
    # ------------------------------------------------------------------ Training set:
    LABEL_FILE = '../data/train-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/train-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255

    write_leveldb(set, labels, 'mnist_train')

    # ------------------------------------------------------------------ Test set:
    LABEL_FILE = '../data/t10k-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/t10k-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255

    write_leveldb(set, labels, 'mnist_test')
